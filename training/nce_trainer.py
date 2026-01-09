import os
# 1. FIX PARALLÉLISME (Indispensable pour éviter les warnings/blocages tokenizers)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# ==========================================
# 2. CLASSES UTILITAIRES (Dataset, Collator)
# ==========================================

class CFDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {"code" : item["code"],
                "feedback" : item["feedback"]}

class CFCollator:
    def __init__(self, tokenizer, max_code_length, max_feedback_length):
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_feedback_length = max_feedback_length
        
    def __call__(self, batch):
        codes = [item["code"] for item in batch]
        feedbacks = [item["feedback"] for item in batch]
        
        # Tokenization
        codes_encoding = self.tokenizer(codes, truncation=True, padding=True, max_length=self.max_code_length, return_tensors="pt")
        feedbacks_encoding = self.tokenizer(feedbacks, truncation=True, padding=True, max_length=self.max_feedback_length, return_tensors="pt")
        
        # Labels factices (requis par l'API Trainer, même si on calcule la loss nous-mêmes)
        dummy_labels = torch.arange(len(codes))
        
        return {"code_input_id" : codes_encoding["input_ids"],
                "feedback_input_id" : feedbacks_encoding["input_ids"],
                "code_attention_mask" : codes_encoding["attention_mask"],
                "feedback_attention_mask" : feedbacks_encoding["attention_mask"],
                "labels": dummy_labels}

# ==========================================
# 3. ARCHITECTURE DU MODÈLE (Bi-Encoder)
# ==========================================

class BiEncoder(nn.Module):
    def __init__(self, base_model_name, lora_config, temperature=0.07):
        super().__init__()
        self.base_model_name = base_model_name
        
        # Chargement du modèle de base
        print(f"📥 Chargement du backbone : {base_model_name}")
        self.encoder = AutoModel.from_pretrained(self.base_model_name, trust_remote_code=True)
        
        # Application de LoRA
        self.encoder = get_peft_model(self.encoder, lora_config)
        
        self.temperature = temperature
        self.encoder.print_trainable_parameters()
        
    def enable_input_require_grads(self):
        self.encoder.enable_input_require_grads()    
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        token_embeddings = outputs.last_hidden_state
        embeddings = self.mean_pooling(token_embeddings, attention_mask)
        # Normalisation L2 (Crucial pour la similarité cosinus)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
        
    def forward(self, code_input_id, code_attention_mask, feedback_input_id, feedback_attention_mask, labels=None):
        code_embeddings = self.encode(code_input_id, code_attention_mask)
        feedback_embeddings = self.encode(feedback_input_id, feedback_attention_mask)
        return code_embeddings, feedback_embeddings

# ==========================================
# 4. CUSTOM TRAINER & METRICS
# ==========================================

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward
        code_emb, feedback_emb = model(**inputs)
        
        # 2. Similarité (Cosinus / Temperature)
        similarity_matrix = torch.matmul(code_emb, feedback_emb.T) / model.temperature
        
        # 3. Labels (La diagonale est la bonne réponse)
        batch_size = code_emb.size(0)
        labels = torch.arange(batch_size).to(code_emb.device)
        
        # 4. Loss Symétrique (InfoNCE)
        loss_c2f = F.cross_entropy(similarity_matrix, labels)
        loss_f2c = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_c2f + loss_f2c) / 2
        
        if return_outputs:
            return (loss, torch.cat((code_emb, feedback_emb), dim=1))
        return loss

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Découpage [Code Embeddings | Feedback Embeddings]
    mid_point = predictions.shape[1] // 2
    code_emb = predictions[:, :mid_point]
    feedback_emb = predictions[:, mid_point:]

    # Matrice de similarité globale sur tout le batch de validation
    similarity_matrix = np.matmul(code_emb, feedback_emb.T)
    labels = np.arange(len(code_emb))
    
    # Ranking
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    hits = (sorted_indices == labels[:, None])
    ranks = np.argwhere(hits)[:, 1] + 1
    
    return {
        "mrr": np.mean(1 / ranks),
        "recall_at_1": np.mean(ranks <= 1),
        "recall_at_5": np.mean(ranks <= 5),
        "recall_at_10": np.mean(ranks <= 10)
    }

# ==========================================
# 5. DIAGNOSTIC AVANT ENTRAINEMENT
# ==========================================
def inspect_global_diagonal_mean(model, collator, dataset, batch_size=8, max_steps=20):
    print(f"\n🔍 Inspection rapide (Pre-training check) sur {max_steps} batchs...")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False)
    device = next(model.parameters()).device
    model.eval()
    
    all_diagonal_probs = []
    
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(dataloader, desc="Inspection")):
            if step >= max_steps: break
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "labels"}
            code_emb, feedback_emb = model(**inputs)
            
            
            similarity_matrix = torch.matmul(code_emb, feedback_emb.T) / model.temperature
            probs = F.softmax(similarity_matrix, dim=1)
            all_diagonal_probs.extend(torch.diagonal(probs).cpu().numpy())

    mean_prob = np.mean(all_diagonal_probs)
    print(f"   => Confiance moyenne initiale sur la bonne réponse : {mean_prob:.2%}")
    model.train()

# ==========================================
# 6. MAIN (EXÉCUTION)
# ==========================================

def main():
    # --- A. Configuration ---
    # Le modèle de base (ex: google/gemma-2b ou embeddinggemma-300m)
    BASE_MODEL = "google/embeddinggemma-300m" 
    # Le dataset sur le Hub HF
    DATASET_ID = "matis35/SYNT_V4"
    # Dossier de sortie final
    FINAL_OUTPUT_DIR = "./final_merged_model"
    
    print("🚀 Démarrage du pipeline d'entraînement...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- B. Chargement des Données ---
    print(f"📚 Chargement du dataset {DATASET_ID}...")
    data_dict = load_dataset(DATASET_ID)
    
    dataset = CFDataset(data_dict["train"].to_list())
    val_dataset = CFDataset(data_dict["validation"].to_list())
    
    # Si 'test' n'existe pas, on utilise validation
    test_data = data_dict["test"].to_list() if "test" in data_dict else data_dict["validation"].to_list()
    test_dataset = CFDataset(test_data)

    collator = CFCollator(tokenizer, max_code_length=512, max_feedback_length=128)

    # --- C. Modèle & LoRA ---
    lora_config = LoraConfig(
        r=32,                
        lora_alpha=64,      
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    model = BiEncoder(base_model_name=BASE_MODEL, lora_config=lora_config, temperature=0.07)
    model.enable_input_require_grads()
    
    # GPU setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps" # Pour Mac
    model.to(device)

    # --- D. Inspection Initiale ---
    inspect_global_diagonal_mean(model, collator, dataset)

    # --- E. Entraînement ---
    training_args = TrainingArguments(
        output_dir="./checkpoints_temp", # Dossier temporaire pour les sauvegardes en cours
        num_train_epochs=5,
        per_device_train_batch_size=64, # Ajuster selon VRAM
        per_device_eval_batch_size=64,
        learning_rate=2e-4,
        bf16=True, # Mettre False si ancien GPU ou erreur
        logging_steps=15,    
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,     
        save_total_limit=1,  
        load_best_model_at_end=True, # Important : recharge le meilleur avant la fusion
        metric_for_best_model="eval_mrr", 
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    print("\n🏋️‍♂️ Début du Training...")
    trainer.train()

    # --- F. FUSION ET SAUVEGARDE FINALE (CRITIQUE) ---
    print("\n💾 FUSION DU MODÈLE (Merge & Unload)...")
    
    # 1. On s'assure que l'encodeur est sur CPU pour éviter les OOM pendant la fusion
    model.encoder.to("cpu")
    
    # 2. On fusionne les poids LoRA dans le modèle de base
    # Cela transforme le modèle "Base + Adapter" en un "Modèle Unique"
    model.encoder = model.encoder.merge_and_unload()
    
    # 3. Sauvegarde propre
    print(f"📦 Sauvegarde du modèle complet dans : {FINAL_OUTPUT_DIR}")
    model.encoder.save_pretrained(FINAL_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_OUTPUT_DIR)
    
    print(f"✅ TERMINÉ ! Le modèle final est prêt dans '{FINAL_OUTPUT_DIR}'.")
    print("👉 Tu peux maintenant le charger directement avec AutoModel.from_pretrained()")

    # --- G. Test Final (Optionnel) ---
    print("\n📊 Évaluation finale sur le Test Set...")
    # Il faut remettre sur GPU pour le test si dispo
    model.encoder.to(device)
    test_output = trainer.predict(test_dataset)
    print(test_output.metrics)

if __name__ == "__main__":
    main()
