import torch
from torch import nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import get_peft_model
import torch.nn.functional as F

# Use cleaned dataset: matis35/RAFT_CLEAN_V1 (40% semantic dedup, threshold 0.40)
# Original: matis35/RAFT (11,806 entries with duplicates)
# Cleaned: matis35/RAFT_CLEAN_V1 (7,024 entries, 73.3% unique patterns)
data_dict = load_dataset('matis35/RAFT_CLEAN_V1')
model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Code-400M_R", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Code-400M_R", trust_remote_code=True)

from torch.utils.data import Dataset
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
        codes_encoding = self.tokenizer(codes, truncation=True, padding=True, max_length=self.max_code_length, return_tensors="pt")
        feedbacks_encoding = self.tokenizer(feedbacks, truncation=True, padding=True, max_length=self.max_feedback_length, return_tensors="pt")
        dummy_labels = torch.arange(len(codes))
        return {"code_input_id" : codes_encoding["input_ids"],
                "feedback_input_id" : feedbacks_encoding["input_ids"],
                "code_attention_mask" : codes_encoding["attention_mask"],
                "feedback_attention_mask" : feedbacks_encoding["attention_mask"],
                 "labels": dummy_labels}



class BiEncoder(nn.Module):
    def __init__(self, base_model_name, lora_config, temperature):
        super().__init__()
        self.base_model_name = base_model_name
        self.encoder = AutoModel.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.temperature = temperature
        print(f"\nModel architecture:")
        self.encoder.print_trainable_parameters()
    def enable_input_require_grads(self):
        """
        Active les gradients sur les embeddings d'entrée.
        Crucial pour faire fonctionner LoRA avec Gradient Checkpointing.
        """
        self.encoder.enable_input_require_grads()    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Le Trainer appelle cette méthode pour activer l'économie de mémoire.
        On passe simplement l'ordre à l'encodeur interne (le modèle HF/PEFT).
        """
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)    
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        token_embeddings = outputs.last_hidden_state
        embeddings = self.mean_pooling(token_embeddings, attention_mask)

        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
        
    def forward(self, code_input_id, code_attention_mask, feedback_input_id, feedback_attention_mask, labels=None):
        code_embeddings = self.encode(code_input_id, code_attention_mask)
        
        feedback_embeddings = self.encode(feedback_input_id, feedback_attention_mask)
        
        return code_embeddings, feedback_embeddings
    


from transformers import Trainer
import torch.nn.functional as F

from transformers import Trainer
import torch.nn.functional as F

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward Pass
        code_emb, feedback_emb = model(**inputs)
        
        # 2. Matrice de Similarité & Labels
        # Note : On garde le calcul ici car on en a besoin pour la Loss
        similarity_matrix = torch.matmul(code_emb, feedback_emb.T) / model.temperature
        batch_size = code_emb.size(0)
        labels = torch.arange(batch_size).to(code_emb.device)
        
        # 3. Calcul de la Loss
        loss_c2f = F.cross_entropy(similarity_matrix, labels)
        loss_f2c = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_c2f + loss_f2c) / 2
        
        # ON A SUPPRIMÉ TOUTE LA PARTIE self.log(...) ICI
        
        if return_outputs:
            # On colle : [Batch, 1024] + [Batch, 1024] -> [Batch, 2048]
            outputs = torch.cat((code_emb, feedback_emb), dim=1)
            return (loss, outputs)
            
        return loss
import numpy as np

def compute_metrics(eval_pred):
    # On reçoit maintenant un seul tableau numpy géant
    predictions = eval_pred.predictions
    
    # Sécurité : si c'est un tuple, on prend le premier élément (au cas où)
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # predictions est de forme (N_samples, 2 * Embedding_Dim)
    # Ex: (1180, 2048)
    
    # On coupe en deux au milieu de la dimension 1
    mid_point = predictions.shape[1] // 2
    
    code_emb = predictions[:, :mid_point]      # La première moitié
    feedback_emb = predictions[:, mid_point:]  # La seconde moitié

    # 1. Calcul de la matrice (Doit donner N_samples x N_samples, ex: 1180x1180)
    similarity_matrix = np.matmul(code_emb, feedback_emb.T)
    
    # 2. Labels (Diagonale)
    labels = np.arange(len(code_emb))
    
    # 3. Calculs
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    hits = (sorted_indices == labels[:, None])
    ranks = np.argwhere(hits)[:, 1] + 1
    
    return {
        "mrr": np.mean(1 / ranks),
        "recall_at_1": np.mean(ranks <= 1),
        "recall_at_5": np.mean(ranks <= 5),
        "recall_at_10": np.mean(ranks <= 10)
    }
def main():
    from transformers import TrainingArguments, AutoTokenizer
from peft import LoraConfig, TaskType

# ==========================================
# 1. CONFIGURATION RAPIDE & TOKENIZER
# ==========================================
model_name = "Salesforce/SFR-Embedding-Code-400M_R"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. DATASETS & COLLATOR
# ==========================================

dataset = CFDataset(data_dict["train"].to_list())
val_dataset = CFDataset(data_dict["validation"].to_list())
collator = CFCollator(tokenizer, max_code_length=512, max_feedback_length=128)

# ==========================================
# 3. PRÉPARATION DU MODÈLE (OPTIMISÉ)
# ==========================================
lora_config = LoraConfig(
    r=32,                
    lora_alpha=64,      
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)

model = BiEncoder(
    base_model_name=model_name,
    lora_config=lora_config,
    temperature=0.07
)
model.enable_input_require_grads()
# ==========================================
# 4. ARGUMENTS D'ENTRAÎNEMENT (OPTIMISÉ)
# ==========================================
training_args = TrainingArguments(
    output_dir="./test_trainer",
    num_train_epochs=10,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    learning_rate=5e-4,
    bf16=True,      
    fp16=False,
    gradient_checkpointing=True,
    # --- Visibilité (TensorBoard) ---
    logging_strategy="steps",
    logging_steps=10,    

    eval_strategy="steps",
    eval_steps=10,     
    
    save_strategy="steps",
    save_steps=200,     
    save_total_limit=1,  
    
    load_best_model_at_end=True,      
    metric_for_best_model="eval_mrr", 
    greater_is_better=True,           
    dataloader_num_workers=8,
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

print("\nDémarrage de l'entraînement optimisé...")
trainer.train()

trainer.save_model("./test_trainer/best_model_final")
print("Entraînement terminé. Meilleur modèle (basé sur MRR) sauvegardé.")

import matplotlib.pyplot as plt

# 1. Extraction des logs de l'historique du Trainer
history = trainer.state.log_history

# On filtre pour séparer les logs d'entraînement (loss) et d'évaluation (mrr, recall)
train_steps = []
train_loss = []

eval_steps = []
eval_loss = []
eval_mrr = []
eval_recall_10 = []

for log in history:
    # Logs d'entraînement (contiennent 'loss' mais pas 'eval_loss')
    if "loss" in log and "eval_loss" not in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])
    
    # Logs d'évaluation (contiennent 'eval_loss')
    if "eval_loss" in log:
        eval_steps.append(log["step"])
        eval_loss.append(log["eval_loss"])
        # On vérifie que la clé existe (au cas où le premier log n'aurait pas tout)
        if "eval_mrr" in log:
            eval_mrr.append(log["eval_mrr"])
        if "eval_recall_at_10" in log:
            eval_recall_10.append(log["eval_recall_at_10"])

# 2. Création du graphique avec Matplotlib
plt.figure(figsize=(15, 5))

# Sous-graphe 1 : Loss (Train vs Eval)
plt.subplot(1, 3, 1)
plt.plot(train_steps, train_loss, label="Training Loss", alpha=0.6)
plt.plot(eval_steps, eval_loss, label="Validation Loss", linewidth=2)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.title("Evolution de la Loss")

# Sous-graphe 2 : MRR
plt.subplot(1, 3, 2)
plt.plot(eval_steps, eval_mrr, color='orange', marker='o')
plt.xlabel("Steps")
plt.ylabel("MRR")
plt.title("Evolution du MRR (Qualité)")
plt.grid(True, linestyle='--', alpha=0.5)

# Sous-graphe 3 : Recall@10
plt.subplot(1, 3, 3)
plt.plot(eval_steps, eval_recall_10, color='green', marker='o')
plt.xlabel("Steps")
plt.ylabel("Recall@10")
plt.title("Evolution du Recall@10")
plt.grid(True, linestyle='--', alpha=0.5)

# 3. Sauvegarde automatique de l'image
plt.tight_layout()
plt.savefig("resultats_entrainement.png")
print("\n📊 Graphique sauvegardé sous : resultats_entrainement.png")

# ==========================================
# 5. ÉVALUATION SUR LE JEU DE TEST (FINAL)
# ==========================================

# 1. On charge le dataset de test (jamais vu par le modèle)
# Vérifie que ton data_dict contient bien une clé "test"
if "test" in data_dict:
    test_dataset = CFDataset(data_dict["test"].to_list())
else:
    # Fallback si pas de split test : on utilise une partie de la validation (déconseillé en prod)
    print("⚠️ Pas de split 'test' trouvé, utilisation de la validation comme test.")
    test_dataset = val_dataset

print("\n" + "="*40)
print("🏁 LANCEMENT DU TEST FINAL")
print("="*40)

# 2. On lance la prédiction
# .predict() est différent de .evaluate() : il ne met pas à jour les gradients
# et retourne les prédictions brutes + les métriques
test_output = trainer.predict(test_dataset)

# 3. Affichage des résultats
metrics = test_output.metrics

# Note : Hugging Face ajoute automatiquement le préfixe "test_" devant tes noms de métriques
print("\nRÉSULTATS OFFICIELS SUR LE TEST SET :")
print(f"🏆 MRR Global      : {metrics.get('test_mrr', 0):.4f}")
print(f"🎯 Recall@1 (Acc)  : {metrics.get('test_recall_at_1', 0):.2%}")
print(f"🔎 Recall@5        : {metrics.get('test_recall_at_5', 0):.2%}")
print(f"🌐 Recall@10       : {metrics.get('test_recall_at_10', 0):.2%}")
print(f"📉 Loss Finale     : {metrics.get('test_loss', 0):.4f}")
print("="*40)


if __name__ == "__main__":
    main()
