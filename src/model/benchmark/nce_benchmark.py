import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import json
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType

# ==========================================
# 0. CONFIGURATION GLOBALE
# ==========================================
RESULTS_FILE = "benchmark_data_final.json"
IMG_DIR = "./benchmark_plots_HD"
os.makedirs(IMG_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-paper')
# Palette étendue pour distinguer 10 modèles
MODEL_COLORS = sns.color_palette("tab10", 10) 
BATCH_STYLES = {64: ':', 128: '--', 256: '-'}
BATCH_MARKERS = {64: 'o', 128: '^', 256: 's'}

# ==========================================
# 1. LISTE "STATE OF THE ART" (9 Modèles)
# ==========================================
MODELS_TO_TEST = [
    "Salesforce/SFR-Embedding-Code-400M_R",
    "microsoft/graphcodebert-base",
    "jinaai/jina-embeddings-v2-base-code",
    "google/embeddinggemma-300m",
    "Snowflake/snowflake-arctic-embed-m",
    "BAAI/bge-base-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1"
]

BATCH_SIZES = [64, 128, 256]

# ==========================================
# 2. CLASSES UTILITAIRES
# ==========================================

class CFDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # On renvoie les données brutes
        return {"code": self.data[idx]["code"], "feedback": self.data[idx]["feedback"]}

class CFCollator:
    def __init__(self, tokenizer, max_code_len, max_feedback_len):
        self.tokenizer = tokenizer
        self.max_code_len = max_code_len
        self.max_feedback_len = max_feedback_len
        
    def __call__(self, batch):
        # Extraction des textes bruts (qui n'ont pas été supprimés grâce à remove_unused_columns=False)
        codes = [item["code"] for item in batch]
        feedbacks = [item["feedback"] for item in batch]
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        c_enc = self.tokenizer(codes, truncation=True, padding=True, max_length=self.max_code_len, return_tensors="pt")
        f_enc = self.tokenizer(feedbacks, truncation=True, padding=True, max_length=self.max_feedback_len, return_tensors="pt")
        
        dummy_labels = torch.arange(len(codes))
        
        return {
            "code_input_id": c_enc["input_ids"],
            "feedback_input_id": f_enc["input_ids"],
            "code_attention_mask": c_enc["attention_mask"],
            "feedback_attention_mask": f_enc["attention_mask"],
            "labels": dummy_labels
        }

class BiEncoder(nn.Module):
    def __init__(self, base_model_name, lora_config, temperature):
        super().__init__()
        self.base_model_name = base_model_name
        self.base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.temperature = temperature
        
    def enable_input_require_grads(self):
        self.base_model.enable_input_require_grads()    
    
    # --- CORRECTION 1 : Ajout de la méthode manquante pour le Trainer ---
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    # --------------------------------------------------------------------

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            emb = outputs.last_hidden_state
        else:
            emb = outputs[0]
        
        embeddings = self.mean_pooling(emb, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)
        
    def forward(self, code_input_id, code_attention_mask, feedback_input_id, feedback_attention_mask, labels=None):
        c_emb = self.encode(code_input_id, code_attention_mask)
        f_emb = self.encode(feedback_input_id, feedback_attention_mask)
        return c_emb, f_emb

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        c_emb, f_emb = model(**inputs)
        sim_matrix = torch.matmul(c_emb, f_emb.T) / model.temperature
        labels = torch.arange(c_emb.size(0)).to(c_emb.device)
        loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
        return (loss, torch.cat((c_emb, f_emb), dim=1)) if return_outputs else loss

def compute_metrics(eval_pred):
    preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    mid = preds.shape[1] // 2
    c_emb, f_emb = preds[:, :mid], preds[:, mid:]
    
    sim_matrix = np.matmul(c_emb, f_emb.T)
    labels = np.arange(len(c_emb))
    
    sorted_indices = np.argsort(-sim_matrix, axis=1)
    ranks = np.argwhere(sorted_indices == labels[:, None])[:, 1] + 1
    
    return {
        "mrr": np.mean(1 / ranks),
        "recall_at_1": np.mean(ranks <= 1),
        "recall_at_5": np.mean(ranks <= 5),
        "recall_at_10": np.mean(ranks <= 10)
    }

# ==========================================
# 3. PLOTTING
# ==========================================

def update_plots(all_results):
    metrics = ["eval_mrr", "eval_recall_at_5", "eval_recall_at_10"]
    unique_models = list(set([res['model_base'] for res in all_results]))
    color_map = {model: MODEL_COLORS[i % len(MODEL_COLORS)] for i, model in enumerate(unique_models)}

    for metric in metrics:
        plt.figure(figsize=(18, 12), dpi=300) 
        
        for res in all_results:
            model_base = res['model_base']
            bs = res['batch_size']
            
            short_name = model_base.split('/')[-1]
            if "embedding" in short_name: short_name = short_name.replace("embedding", "")
            
            label = f"{short_name} (BS={bs})"
            color = color_map[model_base]
            linestyle = BATCH_STYLES.get(bs, '-')
            
            if len(res['eval_steps']) > 0:
                plt.plot(
                    res['eval_steps'], res[metric], 
                    label=label, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8
                )
        
        metric_name = metric.replace("eval_", "").upper().replace("_", " ")
        plt.title(f"BENCHMARK: {metric_name}", fontsize=20, fontweight='bold')
        plt.xlabel("Steps", fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        
        plt.savefig(f"{IMG_DIR}/PROGRESS_{metric}.png")
        plt.close()

# ==========================================
# 4. ENGINE
# ==========================================

def run_single_experiment(model_name, batch_size, data_dict):
    run_name = f"{model_name.split('/')[-1]}_BS{batch_size}"
    print(f"\n{'='*60}")
    print(f" RUN: {run_name}")
    print(f"{'='*60}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
        
        max_len = 1024 if "jina" in model_name or "nomic" in model_name else 512
        collator = CFCollator(tokenizer, max_len, 128)
        
        lora_config = LoraConfig(
            r=32, lora_alpha=64, target_modules="all-linear", 
            lora_dropout=0.05, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )
        
        temp = 0.05 if "gemma" in model_name else 0.07
        model = BiEncoder(model_name, lora_config, temperature=temp)
        model.enable_input_require_grads()

        args = TrainingArguments(
            output_dir=f"./bench_ckpt/{run_name}",
            num_train_epochs=15,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=2e-4,
            lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.01,
            bf16=True, gradient_checkpointing=True,
            logging_steps=10, eval_strategy="steps", eval_steps=25, save_strategy="steps", save_steps=25,
            load_best_model_at_end=True, metric_for_best_model="eval_recall_at_10", greater_is_better=True,
            save_total_limit=1, dataloader_num_workers=4, report_to="none",
            label_names=["labels"],
            
            # --- CORRECTION 2 : CRUCIAL ! Empêche le trainer de supprimer 'code' et 'feedback' ---
            remove_unused_columns=False 
            # -------------------------------------------------------------------------------------
        )

        trainer = ContrastiveTrainer(
            model=model, args=args, 
            # --- CORRECTION 3 : .to_list() pour sécurité Multiprocessing ---
            train_dataset=CFDataset(data_dict["train"].to_list()),
            eval_dataset=CFDataset(data_dict["validation"].to_list()),
            # ---------------------------------------------------------------
            data_collator=collator, compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )

        trainer.train()
        
        print("  Evaluation finale sur le TEST SET...")
        if "test" in data_dict:
            test_dataset = CFDataset(data_dict["test"].to_list())
        else:
            test_dataset = CFDataset(data_dict["validation"].to_list())
            
        test_output = trainer.predict(test_dataset)
        t_metrics = test_output.metrics
        
        print(f" RESULTATS TEST ({run_name}): MRR={t_metrics['test_mrr']:.4f} | R@10={t_metrics['test_recall_at_10']:.4f}")

        history = trainer.state.log_history
        eval_steps, eval_mrr, eval_r5, eval_r10 = [], [], [], []
        for log in history:
            if "eval_mrr" in log:
                eval_steps.append(log["step"])
                eval_mrr.append(log["eval_mrr"])
                eval_r5.append(log["eval_recall_at_5"])
                eval_r10.append(log["eval_recall_at_10"])

        del model, trainer, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "name": run_name,
            "model_base": model_name,
            "batch_size": batch_size,
            "eval_steps": eval_steps,
            "eval_mrr": eval_mrr,
            "eval_recall_at_5": eval_r5,
            "eval_recall_at_10": eval_r10,
            "test_mrr": t_metrics['test_mrr'],
            "test_recall_at_10": t_metrics['test_recall_at_10'],
            "test_recall_at_5": t_metrics['test_recall_at_5']
        }

    except Exception as e:
        print(f" CRASH {run_name}: {e}")
        torch.cuda.empty_cache()
        return None

# ==========================================
# 5. MAIN
# ==========================================

def main():
    print(" Chargement Data...")
    data_dict = load_dataset('matis35/RAFT_CLEAN_V1')
    
    all_results = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f: all_results = json.load(f)
        except: pass

    for model_name in MODELS_TO_TEST:
        for bs in BATCH_SIZES:
            run_id = f"{model_name.split('/')[-1]}_BS{bs}"
            
            if any(r['name'] == run_id for r in all_results):
                print(f"⏩ {run_id} déjà fait.")
                continue
            
            res = run_single_experiment(model_name, bs, data_dict)
            if res:
                all_results.append(res)
                with open(RESULTS_FILE, 'w') as f: json.dump(all_results, f, indent=4)
                update_plots(all_results)

    print("\n" + "="*50)
    print("  LEADERBOARD FINAL (SUR TEST SET)")
    print("="*50)
    df = pd.DataFrame(all_results)
    if not df.empty:
        summary = df[['name', 'test_mrr', 'test_recall_at_5', 'test_recall_at_10']].sort_values(by='test_recall_at_10', ascending=False)
        print(summary.to_string(index=False))
        summary.to_csv("benchmark_leaderboard.csv", index=False)
        print(" Sauvegardé dans benchmark_leaderboard.csv")

if __name__ == "__main__":
    main()