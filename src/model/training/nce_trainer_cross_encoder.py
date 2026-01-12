import os
import torch
import random
import numpy as np
import json
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME = "microsoft/codebert-base"
DATASET_ID = "matis35/cf-synt_V2"
OUTPUT_DIR = "./cross_encoder_codebert_reranker"
MAX_LENGTH = 512
NUM_NEGATIVES = 4  # Ratio 1:4 (1 Positif pour 4 Négatifs)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ==========================================
# 2. NETTOYAGE
# ==========================================
def clean_c_code(code_string):
    if not code_string: return ""
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    return code_string.strip()

# ==========================================
# 3. DATASET POUR CROSS-ENCODER
# ==========================================
class CrossEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_length=512, num_negatives=1, is_eval=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        all_feedbacks = [item["feedback"] for item in data_list]
        print(f"🔨 Construction du dataset ({'Eval' if is_eval else 'Train'})...")
        
        for idx, item in enumerate(tqdm(data_list)):
            code = clean_c_code(item["code"])
            feedback = item["feedback"]
            
            # 1. Exemple POSITIF
            self.samples.append({
                "text_a": code,
                "text_b": feedback,
                "label": 1.0
            })
            
            # 2. Exemples NEGATIFS (Training seulement)
            # Pour l'évaluation on garde quelques négatifs pour calculer l'accuracy
            if not is_eval or True: 
                neg_indices = random.sample([i for i in range(len(all_feedbacks)) if i != idx], num_negatives)
                for neg_idx in neg_indices:
                    self.samples.append({
                        "text_a": code,
                        "text_b": all_feedbacks[neg_idx],
                        "label": 0.0
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Tokenization Cross-Encoder
        # [CLS] Code [SEP] Feedback [SEP]
        # CRUCIAL : truncation="only_first"
        # Si ça dépasse 512 tokens, on coupe le CODE, mais on garde le FEEDBACK entier.
        encoding = self.tokenizer(
            item["text_a"],
            item["text_b"],
            truncation="only_first", 
            max_length=self.max_length,
            padding="max_length"
        )
        
        item_dict = {key: torch.tensor(val) for key, val in encoding.items()}
        item_dict["labels"] = torch.tensor(item["label"]).float()
        return item_dict

# ==========================================
# 4. MÉTRIQUES & EVALUATION
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    probs = 1 / (1 + np.exp(-preds))
    pred_labels = (probs > 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average='binary')
    acc = accuracy_score(labels, pred_labels)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def evaluate_reranking(model, tokenizer, test_data, num_distractors=19, device="cuda"):
    print(f"\nÉvaluation Reranking (1 Positif vs {num_distractors} Négatifs)...")
    model.eval()
    model.to(device)
    
    all_feedbacks = [d["feedback"] for d in test_data]
    ranks = []
    
    # Éval sur 500 exemples pour aller vite (ou len(test_data) pour tout)
    eval_subset = test_data
    
    for i, item in enumerate(tqdm(eval_subset)):
        code = clean_c_code(item["code"])
        true_feedback = item["feedback"]
        
        # Candidats : Le vrai + des faux
        candidates = [true_feedback]
        neg_indices = random.sample([x for x in range(len(all_feedbacks)) if all_feedbacks[x] != true_feedback], num_distractors)
        candidates.extend([all_feedbacks[idx] for idx in neg_indices])
        
        pairs = [[code, cand] for cand in candidates]
        
        # CRUCIAL : truncation="only_first" ici aussi
        inputs = tokenizer(
            [p[0] for p in pairs], 
            [p[1] for p in pairs], 
            padding=True, 
            truncation="only_first", 
            max_length=512, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            
        # Le vrai feedback est à l'index 0. Quel est son rang ?
        sorted_indices = np.argsort(scores)[::-1]
        rank = np.where(sorted_indices == 0)[0][0] + 1
        ranks.append(rank)

    metrics = {
        "mrr": np.mean([1/r for r in ranks]),
        "recall_at_1": np.mean([r <= 1 for r in ranks]),
        "recall_at_5": np.mean([r <= 5 for r in ranks])
    }
    
    print(f"   => MRR: {metrics['mrr']:.4f}")
    print(f"   => Recall@1: {metrics['recall_at_1']:.4f}")
    return metrics

# ==========================================
# 5. MAIN
# ==========================================
def main():
    # A. Setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # B. Data
    print(f" Chargement {DATASET_ID}...")
    dataset_dict = load_dataset(DATASET_ID)
    train_data = dataset_dict["train"].to_list()
    val_data = dataset_dict["validation"].to_list()
    
    train_dataset = CrossEncoderDataset(train_data, tokenizer, num_negatives=NUM_NEGATIVES, is_eval=False)
    val_dataset = CrossEncoderDataset(val_data, tokenizer, num_negatives=1, is_eval=True)
    
    # C. Training Config
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16, # Petit batch car Cross-Encoder lourd
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps", # Correction du nom du paramètre
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, compute_metrics=compute_metrics
    )
    
    # D. Train
    print("\n Début de l'entraînement Cross-Encoder...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # E. Eval Finale
    test_data = dataset_dict["test"].to_list() if "test" in dataset_dict else val_data
    rerank_metrics = evaluate_reranking(model, tokenizer, test_data, num_distractors=19, device=device)
    
    with open(os.path.join(OUTPUT_DIR, "reranking_metrics.json"), "w") as f:
        json.dump(rerank_metrics, f, indent=4)
        
    print(f"\nModèle sauvegardé dans {OUTPUT_DIR}")

if __name__ == "__main__":
    main()