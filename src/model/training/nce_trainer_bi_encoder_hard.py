import os
# Fix pour éviter les deadlocks tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import re
# ==========================================
# 1. CONFIGURATION
# ==========================================
PRETRAINED_MODEL_PATH = "google/embeddinggemma-300m" 
HARD_NEGATIVES_FILE = "train_hard_negatives_cleaned.json"
OUTPUT_DIR = "./final_model_infonce"

# Paramètres
MAX_LENGTH_CODE = 512
MAX_LENGTH_FEEDBACK = 128


# ==========================================
# 1. NETTOYAGE CODE C
# ==========================================
def clean_c_code(code_string):
    if not code_string: return ""
    
    # 1. Supprimer les commentaires blocs /* ... */
    # [\s\S] permet de matcher aussi les sauts de ligne
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    
    # 2. Supprimer les commentaires ligne // ...
    code_string = re.sub(r'//.*', '', code_string)
    
    # 3. Supprimer la fonction main et tout ce qui suit
    # On cherche "int main(...){" ou "void main(...){" et on coupe tout jusqu'à la fin
    # C'est une heuristique robuste car le main sert souvent de runner de test à la fin du fichier
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    
    # 4. (Optionnel) Supprimer les directives #include si tu veux vraiment juste la logique
    # code_string = re.sub(r'#include.*', '', code_string)
    
    # 5. Nettoyage des espaces vides excessifs
    return code_string.strip()

# ==========================================
# 2. DATASET
# ==========================================
class TripletDataset(Dataset):
    def __init__(self, json_file):
        print(f"📂 Chargement des triplets depuis {json_file}...")
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
            
        self.samples = []
        for item in raw_data:
            anchor = clean_c_code(item["code"])
            positive = item["positive"]
            negatives = item["negatives"]
            
            for neg in negatives:
                self.samples.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": neg
                })
        
        print(f"   -> {len(raw_data)} entrées converties en {len(self.samples)} triplets.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ==========================================
# 3. COLLATOR
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
# 4. MODÈLE AVEC LORA & FUSION
# ==========================================
class BiEncoderInfoNCE(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print(f"Chargement du backbone : {model_path}")
        
        base_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Gradient Checkpointing pour économiser la VRAM pendant le train
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # Config LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.encoder = get_peft_model(base_model, peft_config)
        self.encoder.print_trainable_parameters()

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask)
        emb = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, anchor_input_ids, anchor_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask):
        anchor_emb = self.encode(anchor_input_ids, anchor_attention_mask)
        pos_emb = self.encode(pos_input_ids, pos_attention_mask)
        neg_emb = self.encode(neg_input_ids, neg_attention_mask)
        return anchor_emb, pos_emb, neg_emb
    
    def merge_and_save(self, output_dir):
        """Fusionne LoRA dans le modèle de base et sauvegarde le tout."""
        print("Fusion des poids LoRA dans le modèle de base...")
        
        # 1. On repasse le modèle en mode eval pour être propre
        self.encoder.eval()
        
        # 2. Fusion (Merge) : Les poids LoRA sont ajoutés aux poids du backbone
        # merge_and_unload() renvoie le modèle de base standard (AutoModel)
        merged_model = self.encoder.merge_and_unload()
        
        print(f"Sauvegarde du modèle complet dans {output_dir}...")
        merged_model.save_pretrained(output_dir)

# ==========================================
# 5. TRAINER INFONCE
# ==========================================
class InfoNCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        anchor_emb, pos_emb, neg_emb = model(
            inputs["anchor_input_ids"], inputs["anchor_attention_mask"],
            inputs["pos_input_ids"], inputs["pos_attention_mask"],
            inputs["neg_input_ids"], inputs["neg_attention_mask"]
        )
        
        # Concaténation Positifs + Négatifs Difficiles (Hard Negatives)
        target_emb = torch.cat([pos_emb, neg_emb], dim=0)
        
        # Similarité (Batch x 2*Batch)
        scores = torch.mm(anchor_emb, target_emb.transpose(0, 1))
        
        # Temperature Scaling
        logit_scale = 20.0 
        scores = scores * logit_scale
        
        # Labels : L'Ancre i doit matcher la Cible i
        labels = torch.arange(anchor_emb.size(0), device=scores.device)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(scores, labels)
        
        if return_outputs:
            return (loss, (anchor_emb, pos_emb, neg_emb))
        return loss

# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("Démarrage du Fine-Tuning InfoNCE (Hard Negs + In-Batch)...")
    
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    train_dataset = TripletDataset(HARD_NEGATIVES_FILE)
    collator = TripletCollator(tokenizer, MAX_LENGTH_CODE, MAX_LENGTH_FEEDBACK)
    
    model = BiEncoderInfoNCE(PRETRAINED_MODEL_PATH)
    
    training_args = TrainingArguments(
        output_dir="./checkpoints_infonce_merged",
        num_train_epochs=3,              
        per_device_train_batch_size=256, 
        gradient_accumulation_steps=2,   
        learning_rate=5e-5,             
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
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
    
    print("\nEntraînement terminé. Début de la fusion...")
    # Sauvegarde Finale : Modèle fusionné + Tokenizer
    model.merge_and_save(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modèle complet (Stand-alone) sauvegardé dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()