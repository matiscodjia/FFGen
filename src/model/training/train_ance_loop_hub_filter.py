import os
import json
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import re
import numpy as np
import shutil

# ==========================================
# 1. CONFIGURATION "POWER USER"
# ==========================================
PRETRAINED_MODEL_PATH = "matis35/feedbacker-2" 
HF_DATASET_ID = "matis35/cf-synt_V2"
HF_SPLIT = "train" 

OUTPUT_DIR = "./final_model_ance_logged"
LOG_DIR = "./ance_logs"

# Paramètres ANCE
NUM_ANCE_ROUNDS = 3       
EPOCHS_PER_ROUND = 1      
BATCH_SIZE = 32           
NUM_HARD_NEGATIVES = 1    

# Paramètres Avancés
HUB_THRESHOLD_PCT = 0.005  # 0.5% (Plus strict pour éviter les mini-hubs)
MAX_LENGTH_CODE = 512
MAX_LENGTH_FEEDBACK = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Création du dossier de logs
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR)

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================
def clean_c_code(code_string):
    if not code_string: return ""
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'#include\s+<.*?>', '', code_string)
    code_string = re.sub(r'#include\s+".*?"', '', code_string)
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    return code_string.strip()

def write_log(filename, content):
    with open(os.path.join(LOG_DIR, filename), "a", encoding="utf-8") as f:
        f.write(content + "\n")

# ==========================================
# 3. MODÈLE BLINDÉ (SAFE FP32)
# ==========================================
class BiEncoderModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # Chargement en FP16 pour la VRAM
        base_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        base_model.gradient_checkpointing_enable()
        
        if hasattr(base_model, "enable_input_require_grads"): base_model.enable_input_require_grads()
        else: base_model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.encoder = get_peft_model(base_model, peft_config)

    def safe_pool_and_normalize(self, last_hidden_state, attention_mask):
        """Le cœur de la sécurité : Cast en FP32 pour les maths"""
        # 1. Cast FP32
        last_hidden = last_hidden_state.to(torch.float32)
        mask = attention_mask.to(torch.float32)
        
        # 2. Pooling
        sum_mask = torch.clamp(mask.sum(1, keepdim=True), min=1e-9)
        emb = torch.sum(last_hidden * mask.unsqueeze(-1), 1) / sum_mask
        
        # 3. Normalisation L2
        return F.normalize(emb, p=2, dim=1)

    def forward(self, anchor_input_ids, anchor_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask):
        # Anchor
        out_a = self.encoder(anchor_input_ids, anchor_attention_mask)
        emb_a = self.safe_pool_and_normalize(out_a.last_hidden_state, anchor_attention_mask)
        
        # Positive
        out_p = self.encoder(pos_input_ids, pos_attention_mask)
        emb_p = self.safe_pool_and_normalize(out_p.last_hidden_state, pos_attention_mask)
        
        # Negative
        out_n = self.encoder(neg_input_ids, neg_attention_mask)
        emb_n = self.safe_pool_and_normalize(out_n.last_hidden_state, neg_attention_mask)
        
        return emb_a, emb_p, emb_n
    
    def encode_inference(self, input_ids, attention_mask):
        """Helper pour le mining"""
        out = self.encoder(input_ids, attention_mask)
        return self.safe_pool_and_normalize(out.last_hidden_state, attention_mask)

    def merge_and_save(self, output_dir):
        self.encoder.eval()
        merged = self.encoder.merge_and_unload()
        merged.save_pretrained(output_dir)

# ==========================================
# 4. MINING AVEC LOGGING AVANCÉ
# ==========================================
def mine_hard_negatives(model, tokenizer, raw_data, round_idx):
    log_file = f"round_{round_idx}_mining.txt"
    write_log(log_file, f"=== RAPPORT DE MINING : ROUND {round_idx} ===\n")
    print(f"\n⛏️  MINING ROUND {round_idx}...")
    model.eval()
    
    # --- A. Indexation ---
    unique_feedbacks = list(set([item["positive"] for item in raw_data]))
    # Filtre anti-vide par sécurité
    unique_feedbacks = [f for f in unique_feedbacks if len(f.strip()) > 5]
    
    feedback_embeddings = []
    for i in tqdm(range(0, len(unique_feedbacks), 64), desc="Index Encoding"):
        batch = unique_feedbacks[i : i + 64]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH_FEEDBACK, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = model.encode_inference(inputs["input_ids"], inputs["attention_mask"])
            feedback_embeddings.append(emb.cpu())
    
    index_matrix = torch.cat(feedback_embeddings, dim=0).to(DEVICE)

    # --- B. Recherche & Hubness ---
    temp_candidates = []
    all_negative_indices = []
    
    # Pour les stats de similarité
    sim_stats_pos = []
    sim_stats_neg = []
    
    batch_size_mine = 32
    for i in tqdm(range(0, len(raw_data), batch_size_mine), desc="Scanning"):
        batch_items = raw_data[i : i + batch_size_mine]
        batch_codes = [clean_c_code(item["code"]) for item in batch_items]
        batch_positives = [item["positive"] for item in batch_items]
        
        valid_indices = [k for k, c in enumerate(batch_codes) if c and len(c) > 10]
        if not valid_indices: continue
        
        batch_codes_clean = [batch_codes[k] for k in valid_indices]
        batch_positives_valid = [batch_positives[k] for k in valid_indices]
        
        inputs = tokenizer(batch_codes_clean, padding=True, truncation=True, max_length=MAX_LENGTH_CODE, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            q_emb = model.encode_inference(inputs["input_ids"], inputs["attention_mask"])
        
        # Score Matrix [Batch, N_Feedbacks]
        scores = torch.matmul(q_emb, index_matrix.T)
        
        # Récupération des Top Candidats
        top_k = torch.topk(scores, k=20, dim=1)
        
        for j, (indices, vals) in enumerate(zip(top_k.indices, top_k.values)):
            indices_cpu = indices.cpu().tolist()
            vals_cpu = vals.cpu().tolist()
            
            # On cherche l'index du vrai positif pour comparer les scores
            try:
                # C'est un peu lent, mais utile pour les stats
                true_idx = unique_feedbacks.index(batch_positives_valid[j])
                true_score = scores[j, true_idx].item()
                sim_stats_pos.append(true_score)
            except:
                true_score = -1.0 # Pas trouvé (rare)

            temp_candidates.append({
                "code": batch_codes_clean[j],
                "positive": batch_positives_valid[j],
                "positive_score": true_score,
                "candidate_indices": indices_cpu,
                "candidate_scores": vals_cpu
            })
            all_negative_indices.extend(indices_cpu)

    # --- C. Analyse Hubness & Logging ---
    counts = Counter(all_negative_indices)
    threshold_count = len(raw_data) * HUB_THRESHOLD_PCT
    banned_indices = {idx for idx, count in counts.items() if count > threshold_count}
    
    # LOGGING DES HUBS
    write_log(log_file, f"--- HUB ANALYSIS ---")
    write_log(log_file, f"Threshold (0.5%): {int(threshold_count)} citations.")
    write_log(log_file, f"Banned Feedbacks: {len(banned_indices)}")
    
    if len(banned_indices) > 0:
        write_log(log_file, "TOP 5 BANNED HUBS:")
        top_banned = counts.most_common(5)
        for idx, cnt in top_banned:
            if idx in banned_indices:
                write_log(log_file, f"   [{cnt}x] : {unique_feedbacks[idx][:100]}...")

    # --- D. Sélection Finale ---
    new_triplets = []
    examples_to_log = [] # On en garde 5 pour voir
    
    for item in temp_candidates:
        hard_negatives_found = []
        neg_scores = []
        
        for idx, score in zip(item["candidate_indices"], item["candidate_scores"]):
            candidate_text = unique_feedbacks[idx]
            # Filtre strict : Pas le vrai positif, Pas un Hub
            if candidate_text != item["positive"] and idx not in banned_indices:
                hard_negatives_found.append(candidate_text)
                neg_scores.append(score)
                sim_stats_neg.append(score)
                
                if len(hard_negatives_found) >= NUM_HARD_NEGATIVES:
                    break
        
        for neg, n_score in zip(hard_negatives_found, neg_scores):
            new_triplets.append({
                "anchor": item["code"],
                "positive": item["positive"],
                "negative": neg
            })
            
            # On stocke quelques exemples intéressants (où le score est haut)
            if len(examples_to_log) < 10 and n_score > 0.4:
                examples_to_log.append({
                    "code": item["code"][:100],
                    "pos": item["positive"][:100],
                    "neg": neg[:100],
                    "pos_score": item["positive_score"],
                    "neg_score": n_score
                })

    # LOGGING DES STATS
    avg_pos = np.mean(sim_stats_pos) if sim_stats_pos else 0
    avg_neg = np.mean(sim_stats_neg) if sim_stats_neg else 0
    
    write_log(log_file, f"\n--- SIMILARITY STATS ---")
    write_log(log_file, f"Average Similarity (Anchor <-> Positive) : {avg_pos:.4f}")
    write_log(log_file, f"Average Similarity (Anchor <-> Negative) : {avg_neg:.4f}")
    write_log(log_file, f"Margin (Gap) : {avg_pos - avg_neg:.4f}")
    
    write_log(log_file, f"\n--- TRIPLET EXAMPLES (Hardest Found) ---")
    for i, ex in enumerate(examples_to_log):
        write_log(log_file, f"Example #{i+1}:")
        write_log(log_file, f"   Code: {ex['code']}...")
        write_log(log_file, f"   (+) : {ex['pos']}... (Score: {ex['pos_score']:.3f})")
        write_log(log_file, f"   (-) : {ex['neg']}... (Score: {ex['neg_score']:.3f})")
        write_log(log_file, f"   -----------------------------")

    print(f"   -> Fin du Mining. {len(new_triplets)} Triplets générés.")
    print(f"   -> Rapport complet écrit dans {LOG_DIR}/{log_file}")
    return new_triplets

# ==========================================
# 5. DATASET & TRAINER
# ==========================================
class DynamicTripletDataset(Dataset):
    def __init__(self, triplets): self.samples = triplets
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class TripletCollator:
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def __call__(self, batch):
        a = self.tokenizer([x["anchor"] for x in batch], padding=True, truncation=True, max_length=MAX_LENGTH_CODE, return_tensors="pt")
        p = self.tokenizer([x["positive"] for x in batch], padding=True, truncation=True, max_length=MAX_LENGTH_FEEDBACK, return_tensors="pt")
        n = self.tokenizer([x["negative"] for x in batch], padding=True, truncation=True, max_length=MAX_LENGTH_FEEDBACK, return_tensors="pt")
        return {
            "anchor_input_ids": a["input_ids"], "anchor_attention_mask": a["attention_mask"],
            "pos_input_ids": p["input_ids"], "pos_attention_mask": p["attention_mask"],
            "neg_input_ids": n["input_ids"], "neg_attention_mask": n["attention_mask"]
        }

class InfoNCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        a_emb, p_emb, n_emb = model(
            inputs["anchor_input_ids"], inputs["anchor_attention_mask"],
            inputs["pos_input_ids"], inputs["pos_attention_mask"],
            inputs["neg_input_ids"], inputs["neg_attention_mask"]
        )
        target_emb = torch.cat([p_emb, n_emb], dim=0)
        scores = torch.mm(a_emb, target_emb.transpose(0, 1)) * 20.0 
        labels = torch.arange(a_emb.size(0), device=scores.device)
        loss = nn.CrossEntropyLoss()(scores, labels)
        return (loss, (a_emb, p_emb, n_emb)) if return_outputs else loss

# ==========================================
# 6. MAIN LOOP
# ==========================================
def main():
    print(f"🚀 Démarrage ANCE 'Power Logger'...")
    
    # Chargement Données
    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    raw_data = [{"code": item.get("code", item.get("anchor")), "positive": item.get("feedback", item.get("positive"))} for item in ds if item.get("code") or item.get("anchor")]
    
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    model = BiEncoderModel(PRETRAINED_MODEL_PATH)
    collator = TripletCollator(tokenizer)
    
    for round_idx in range(1, NUM_ANCE_ROUNDS + 1):
        print(f"\n" + "="*40)
        print(f"🔄 ANCE ROUND {round_idx}/{NUM_ANCE_ROUNDS}")
        print("="*40)
        
        # --- PHASE 1 : MINING AVEC LOGGING ---
        triplets = mine_hard_negatives(model, tokenizer, raw_data, round_idx)
        if not triplets: break
        
        train_dataset = DynamicTripletDataset(triplets)
        
        # --- PHASE 2 : TRAINING ---
        print(f"\n🏋️‍♂️ TRAINING ROUND {round_idx}...")
        model.train()
        
        training_args = TrainingArguments(
            output_dir=f"./checkpoints_ance_rd{round_idx}",
            num_train_epochs=EPOCHS_PER_ROUND,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,
            warmup_ratio=0.1,
            bf16=True, # Safe car on a blindé le modèle manuellement
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False
        )
        
        trainer = InfoNCETrainer(
            model=model, args=training_args, train_dataset=train_dataset, data_collator=collator
        )
        trainer.train()
        
        # Sauvegarde intermédiaire des logs de loss
        loss_log = trainer.state.log_history
        with open(f"{LOG_DIR}/round_{round_idx}_loss.json", "w") as f:
            json.dump(loss_log, f, indent=2)

        torch.cuda.empty_cache()

    print(f"\n💾 Sauvegarde Modèle Final dans {OUTPUT_DIR}...")
    model.merge_and_save(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Terminé. Vérifie le dossier 'ance_logs' pour voir la magie !")

if __name__ == "__main__":
    main()