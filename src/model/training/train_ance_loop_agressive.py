import os
# Désactivation des warnings de parallélisme (évite les deadlocks)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# ==========================================
# 1. CONFIGURATION
# ==========================================
# On part du modèle chauffé (Phase 1)
PRETRAINED_MODEL_PATH = "google/embeddinggemma-300m" 

# Données Source (Hugging Face)
HF_DATASET_ID = "matis35/cf-synt_V2"
HF_SPLIT = "train" 

OUTPUT_DIR = "./final_model_ance_infonce"

# Paramètres ANCE / InfoNCE
NUM_ANCE_ROUNDS = 3       # Cycles Mine -> Train
EPOCHS_PER_ROUND = 1      # 1 Epoch par cycle est standard pour ANCE
BATCH_SIZE = 32           # Attention à la VRAM : la matrice de loss fait (32 x 64)
NUM_HARD_NEGATIVES = 1    # Combien de pièges on ajoute par exemple

MAX_LENGTH_CODE = 512
MAX_LENGTH_FEEDBACK = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. NETTOYAGE AGRESSIF (Cohérent avec l'Eval)
# ==========================================
def clean_c_code(code_string):
    """
    Supprime tout le contexte (main, includes) pour forcer le modèle 
    à comprendre la fonction isolée.
    """
    if not code_string: return ""
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'#include\s+<.*?>', '', code_string)
    code_string = re.sub(r'#include\s+".*?"', '', code_string)
    # Suppression du main et de son body
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    return code_string.strip()

# ==========================================
# 3. MOTEUR DE MINAGE (Le "Cerveau" de ANCE)
# ==========================================
def mine_hard_negatives(model, tokenizer, raw_data):
    print(f"\n⛏️  MINING STEP : Recherche des erreurs du modèle...")
    model.eval()
    
    # A. Indexation de TOUS les feedbacks uniques (Target Space)
    unique_feedbacks = list(set([item["positive"] for item in raw_data]))
    print(f"   -> Encodage de l'index ({len(unique_feedbacks)} candidats)...")
    
    feedback_embeddings = []
    # On encode par petits batchs pour économiser la VRAM
    for i in tqdm(range(0, len(unique_feedbacks), 64), desc="Index Encoding"):
        batch = unique_feedbacks[i : i + 64]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH_FEEDBACK, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean Pooling
            emb = torch.sum(outputs.last_hidden_state * inputs["attention_mask"].unsqueeze(-1), 1) / torch.clamp(inputs["attention_mask"].sum(1, keepdim=True), min=1e-9)
            # Normalisation L2 critique pour le Cosine Similarity
            emb = F.normalize(emb, p=2, dim=1)
            feedback_embeddings.append(emb.cpu())
    
    index_matrix = torch.cat(feedback_embeddings, dim=0).to(DEVICE) # [N_Feedbacks, Dim]

    # B. Recherche des Négatifs pour chaque Code
    new_triplets = []
    print("   -> Mining sur le dataset d'entraînement...")
    
    batch_size_mine = 32
    for i in tqdm(range(0, len(raw_data), batch_size_mine), desc="Mining"):
        batch_items = raw_data[i : i + batch_size_mine]
        
        # NETTOYAGE : Le modèle voit le code "nu" (sans main)
        batch_codes = [clean_c_code(item["code"]) for item in batch_items]
        batch_positives = [item["positive"] for item in batch_items]
        
        # Filtrage des codes vides (si un étudiant n'avait mis que le main)
        valid_indices = [k for k, c in enumerate(batch_codes) if c]
        if not valid_indices: continue
        
        batch_codes_clean = [batch_codes[k] for k in valid_indices]
        batch_positives_valid = [batch_positives[k] for k in valid_indices]
        
        # Encodage Requête
        inputs = tokenizer(batch_codes_clean, padding=True, truncation=True, max_length=MAX_LENGTH_CODE, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            q_emb = torch.sum(outputs.last_hidden_state * inputs["attention_mask"].unsqueeze(-1), 1) / torch.clamp(inputs["attention_mask"].sum(1, keepdim=True), min=1e-9)
            q_emb = F.normalize(q_emb, p=2, dim=1)
        
        # Similarité [Batch, N_Feedbacks]
        scores = torch.matmul(q_emb, index_matrix.T)
        
        # On prend les Top-K résultats
        # K = Négatifs voulus + 10 (marge de sécurité au cas où le modèle trouve le bon feedback)
        top_k = torch.topk(scores, k=NUM_HARD_NEGATIVES + 10, dim=1)
        
        for j, (indices, vals) in enumerate(zip(top_k.indices, top_k.values)):
            true_feedback = batch_positives_valid[j]
            cleaned_code = batch_codes_clean[j]
            
            hard_negatives_found = []
            for idx in indices:
                candidate = unique_feedbacks[idx.item()]
                # Si le candidat trouvé n'est PAS le vrai feedback, c'est un Hard Negative
                if candidate != true_feedback:
                    hard_negatives_found.append(candidate)
                    if len(hard_negatives_found) >= NUM_HARD_NEGATIVES:
                        break
            
            # Création des triplets pour InfoNCE
            for neg in hard_negatives_found:
                new_triplets.append({
                    "anchor": cleaned_code, 
                    "positive": true_feedback,
                    "negative": neg
                })

    print(f"   -> Fin du Mining. {len(new_triplets)} Hard Negatives identifiés.")
    return new_triplets

# ==========================================
# 4. DATASET & COLLATOR
# ==========================================
class DynamicTripletDataset(Dataset):
    def __init__(self, triplets):
        self.samples = triplets
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

class TripletCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        anchors = [x["anchor"] for x in batch]
        positives = [x["positive"] for x in batch]
        negatives = [x["negative"] for x in batch]
        
        a_enc = self.tokenizer(anchors, padding=True, truncation=True, max_length=MAX_LENGTH_CODE, return_tensors="pt")
        p_enc = self.tokenizer(positives, padding=True, truncation=True, max_length=MAX_LENGTH_FEEDBACK, return_tensors="pt")
        n_enc = self.tokenizer(negatives, padding=True, truncation=True, max_length=MAX_LENGTH_FEEDBACK, return_tensors="pt")
        
        return {
            "anchor_input_ids": a_enc["input_ids"],
            "anchor_attention_mask": a_enc["attention_mask"],
            "pos_input_ids": p_enc["input_ids"],
            "pos_attention_mask": p_enc["attention_mask"],
            "neg_input_ids": n_enc["input_ids"],
            "neg_attention_mask": n_enc["attention_mask"]
        }

# ==========================================
# 5. MODÈLE & INFONCE TRAINER
# ==========================================
class BiEncoderModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        base_model.gradient_checkpointing_enable()
        
        # Hack pour LoRA + Gradient Checkpointing
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            base_model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.encoder = get_peft_model(base_model, peft_config)

    def forward(self, anchor_input_ids, anchor_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask):
        # Encodage partagé (Siamese Network)
        # Helper pooling local
        def get_emb(ids, mask):
            out = self.encoder(ids, mask)
            # Mean Pooling manuel robuste
            emb = torch.sum(out.last_hidden_state * mask.unsqueeze(-1), 1) / torch.clamp(mask.sum(1, keepdim=True), min=1e-9)
            return F.normalize(emb, p=2, dim=1)

        a_emb = get_emb(anchor_input_ids, anchor_attention_mask)
        p_emb = get_emb(pos_input_ids, pos_attention_mask)
        n_emb = get_emb(neg_input_ids, neg_attention_mask)
        
        return a_emb, p_emb, n_emb
    
    def merge_and_save(self, output_dir):
        self.encoder.eval()
        merged = self.encoder.merge_and_unload()
        merged.save_pretrained(output_dir)

class InfoNCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        anchor_emb, pos_emb, neg_emb = model(
            inputs["anchor_input_ids"], inputs["anchor_attention_mask"],
            inputs["pos_input_ids"], inputs["pos_attention_mask"],
            inputs["neg_input_ids"], inputs["neg_attention_mask"]
        )
        
        # --- LOGIQUE INFONCE ---
        # 1. On concatène les candidats : [Positifs du batch, Négatifs du batch]
        # Taille : [2 * Batch_Size, Hidden_Dim]
        target_emb = torch.cat([pos_emb, neg_emb], dim=0)
        
        # 2. Matrice de Similarité : [Batch_Size, 2 * Batch_Size]
        # Anchor[i] est comparé à (Pos[0]...Pos[N]) ET (Neg[0]...Neg[N])
        scores = torch.mm(anchor_emb, target_emb.transpose(0, 1))
        
        # 3. Temperature Scaling (Crucial pour InfoNCE)
        scores = scores * 20.0 
        
        # 4. Labels
        # L'Anchor[i] doit matcher Target[i] (qui est Pos[i])
        labels = torch.arange(anchor_emb.size(0), device=scores.device)
        
        # 5. Cross Entropy
        loss = nn.CrossEntropyLoss()(scores, labels)
        
        return (loss, (anchor_emb, pos_emb, neg_emb)) if return_outputs else loss

# ==========================================
# 6. MAIN LOOP
# ==========================================
def main():
    print(f"🚀 Démarrage ANCE (InfoNCE + Clean Code Agressif)...")
    
    # 1. Chargement des données brutes
    print("📥 Téléchargement Dataset HF...")
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    raw_data = []
    for item in tqdm(ds, desc="Conversion"):
        # Mapping des colonnes
        code = item.get("code", item.get("anchor"))
        pos = item.get("feedback", item.get("positive"))
        if code and pos:
            raw_data.append({"code": code, "positive": pos})
    
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    model = BiEncoderModel(PRETRAINED_MODEL_PATH)
    collator = TripletCollator(tokenizer)
    
    # 2. Boucle ANCE
    for round_idx in range(NUM_ANCE_ROUNDS):
        print(f"\n" + "="*40)
        print(f"🔄 ANCE ROUND {round_idx + 1}/{NUM_ANCE_ROUNDS}")
        print("="*40)
        
        # PHASE A : MINING
        triplets = mine_hard_negatives(model.encoder, tokenizer, raw_data)
        if not triplets: break
        
        train_dataset = DynamicTripletDataset(triplets)
        
        # PHASE B : TRAINING (InfoNCE)
        print(f"\n🏋️‍♂️ TRAINING : Optimisation InfoNCE sur {len(triplets)} triplets...")
        model.train()
        
        training_args = TrainingArguments(
            output_dir=f"./checkpoints_ance_infonce_rd{round_idx}",
            num_train_epochs=EPOCHS_PER_ROUND,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=2,
            learning_rate=3e-5, # LR doux pour ne pas casser le modèle
            warmup_ratio=0.1,
            bf16=True,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False
        )
        
        trainer = InfoNCETrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator
        )
        
        trainer.train()
        
        # Nettoyage VRAM
        torch.cuda.empty_cache()

    # 3. Finalisation
    print(f"\n💾 Sauvegarde Modèle Final dans {OUTPUT_DIR}...")
    model.merge_and_save(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Terminé. Prêt pour l'évaluation.")

if __name__ == "__main__":
    main()