import os
# Fix pour éviter les deadlocks tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import re

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
# On n'a plus besoin de PEFT/LoRA car on va fine-tuner tous les poids (ou recharger LoRA si besoin)
# Ici, on recharge le modèle fusionné, donc c'est un modèle standard.

# ==========================================
# 1. CONFIGURATION
# ==========================================
# IMPORTANT : On part de ton modèle DÉJÀ entrainé (Step 0)
PRETRAINED_MODEL_PATH = "matis35/feedbacker-2" 
# Le fichier généré par le script de mining
HARD_NEGATIVES_FILE = "train_hard_negatives.json" 
# Où sauvegarder le modèle "Expert" final
OUTPUT_DIR = "./final_model_hard_negatives"

# Paramètres
MAX_LENGTH_CODE = 512      # On garde ta config corrigée
MAX_LENGTH_FEEDBACK = 128
TRIPLET_MARGIN = 0.5       # Marge standard pour Cosine Distance

# ==========================================
# 2. DATASET (Triplet Parsing)
# ==========================================
class TripletDataset(Dataset):
    def __init__(self, json_file):
        print(f"📂 Chargement des triplets depuis {json_file}...")
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
            
        self.samples = []
        # APLATISSEMENT : On transforme {code, pos, [neg1, neg2]} en plusieurs triplets
        for item in raw_data:
            anchor = item["code"]
            positive = item["positive"]
            negatives = item["negatives"]
            
            for neg in negatives:
                self.samples.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": neg
                })
        
        print(f"   -> {len(raw_data)} entrées brutes converties en {len(self.samples)} triplets d'entrainement.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ==========================================
# 3. COLLATOR (Tokenization A, P, N)
# ==========================================
class TripletCollator:
    def __init__(self, tokenizer, max_code_len, max_feed_len):
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_feed_len = max_feed_len
        
    def __call__(self, batch):
        anchors = [x["anchor"] for x in batch]
        positives = [x["positive"] for x in batch]
        negatives = [x["negative"] for x in batch]
        
        # Tokenization séparée
        a_enc = self.tokenizer(anchors, padding=True, truncation=True, max_length=self.max_code_len, return_tensors="pt")
        p_enc = self.tokenizer(positives, padding=True, truncation=True, max_length=self.max_feed_len, return_tensors="pt")
        n_enc = self.tokenizer(negatives, padding=True, truncation=True, max_length=self.max_feed_len, return_tensors="pt")
        
        return {
            "anchor_input_ids": a_enc["input_ids"],
            "anchor_attention_mask": a_enc["attention_mask"],
            "pos_input_ids": p_enc["input_ids"],
            "pos_attention_mask": p_enc["attention_mask"],
            "neg_input_ids": n_enc["input_ids"],
            "neg_attention_mask": n_enc["attention_mask"]
        }

# ==========================================
# 4. MODÈLE & LOSS
# ==========================================
class BiEncoderTriplet(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # On charge le modèle déjà entrainé
        print(f"Chargement du backbone expert : {model_path}")
        self.encoder = AutoModel.from_pretrained(model_path)
        # Gradient Checkpointing pour économiser la VRAM (Important car on encode 3x plus)
        self.encoder.gradient_checkpointing_enable() 

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        emb = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(emb, p=2, dim=1) # Toujours normaliser pour Cosine Sim

    def forward(self, anchor_input_ids, anchor_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask):
        # On encode les 3 parties
        anchor_emb = self.encode(anchor_input_ids, anchor_attention_mask)
        pos_emb = self.encode(pos_input_ids, pos_attention_mask)
        neg_emb = self.encode(neg_input_ids, neg_attention_mask)
        
        return anchor_emb, pos_emb, neg_emb

class TripletTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward Pass
        anchor_emb, pos_emb, neg_emb = model(
            inputs["anchor_input_ids"], inputs["anchor_attention_mask"],
            inputs["pos_input_ids"], inputs["pos_attention_mask"],
            inputs["neg_input_ids"], inputs["neg_attention_mask"]
        )
        
        # 2. Calcul des similarités (Cosinus)
        # Note : Les embeddings sont déjà normalisés, donc dot product = cosine similarity
        sim_pos = torch.sum(anchor_emb * pos_emb, dim=-1) # Proche de 1 idéalement
        sim_neg = torch.sum(anchor_emb * neg_emb, dim=-1) # Proche de -1 ou 0 idéalement
        
        # 3. Triplet Loss avec Marge
        # On veut: sim_pos > sim_neg + margin
        # Loss = ReLU(sim_neg - sim_pos + margin)
        loss = torch.mean(torch.relu(sim_neg - sim_pos + TRIPLET_MARGIN))
        
        if return_outputs:
            return (loss, (anchor_emb, pos_emb, neg_emb))
        return loss


# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("🚀 Démarrage du Fine-Tuning 'Hard Negatives'...")
    
    # A. Chargement
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    train_dataset = TripletDataset(HARD_NEGATIVES_FILE)
    collator = TripletCollator(tokenizer, MAX_LENGTH_CODE, MAX_LENGTH_FEEDBACK)
    
    model = BiEncoderTriplet(PRETRAINED_MODEL_PATH)
    
    # B. Config Entrainement
    # Triplet Loss converge vite car on part déjà d'un bon modèle
    # 1 ou 2 époques suffisent souvent pour corriger les erreurs sans overfitter
    training_args = TrainingArguments(
        output_dir="./checkpoints_triplet",
        num_train_epochs=2,              # Court mais intense
        per_device_train_batch_size=32,  # 32 triplets = 96 encodages ! Attention VRAM
        learning_rate=1e-5,              # LR très faible pour ne pas casser le modèle existant
        warmup_ratio=0.1,
        bf16=True,                       # Accélération A40
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False
    )
    
    trainer = TripletTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator
    )
    
    # C. Train
    trainer.train()
    
    # D. Sauvegarde Finale
    print(f"\nSauvegarde du modèle expert dans {OUTPUT_DIR}...")
    model.encoder.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()