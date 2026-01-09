import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# Configuration esthétique style "Publication Scientifique"
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl", 6)

# ==========================================
# 1. CLASSES UTILITAIRES
# ==========================================

class CFDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {"code" : item["code"], "feedback" : item["feedback"]}

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
        
        return {
            "code_input_id" : codes_encoding["input_ids"],
            "feedback_input_id" : feedbacks_encoding["input_ids"],
            "code_attention_mask" : codes_encoding["attention_mask"],
            "feedback_attention_mask" : feedbacks_encoding["attention_mask"],
            "labels": dummy_labels
        }

class BiEncoder(nn.Module):
    def __init__(self, base_model_name, lora_config, temperature):
        super().__init__()
        self.base_model_name = base_model_name
        self.encoder = AutoModel.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.temperature = temperature
        
    def enable_input_require_grads(self):
        self.encoder.enable_input_require_grads()    
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
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

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        code_emb, feedback_emb = model(**inputs)
        similarity_matrix = torch.matmul(code_emb, feedback_emb.T) / model.temperature
        batch_size = code_emb.size(0)
        labels = torch.arange(batch_size).to(code_emb.device)
        loss_c2f = F.cross_entropy(similarity_matrix, labels)
        loss_f2c = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_c2f + loss_f2c) / 2
        if return_outputs:
            outputs = torch.cat((code_emb, feedback_emb), dim=1)
            return (loss, outputs)
        return loss

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    mid_point = predictions.shape[1] // 2
    code_emb = predictions[:, :mid_point]
    feedback_emb = predictions[:, mid_point:]

    similarity_matrix = np.matmul(code_emb, feedback_emb.T)
    labels = np.arange(len(code_emb))
    
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
# 2. FONCTIONS DE VISUALISATION (DATA ORIENTED)
# ==========================================

def plot_curves(all_results, metric_key, title, filename):
    """Trace les courbes d'évolution"""
    plt.figure(figsize=(12, 8), dpi=300)
    for res in all_results:
        linestyle = '-' if "SFR" in res['name'] else '--'
        marker = 'o' if "SFR" in res['name'] else '^'
        plt.plot(res['eval_steps'], res[metric_key], label=res['name'], 
                 linewidth=2.5, linestyle=linestyle, marker=marker, markersize=6, alpha=0.8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel(metric_key.replace('_', ' ').upper(), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f" Courbes sauvegardées : {filename}")

def plot_loss_grid(all_results):
    """Grille comparative Train vs Val Loss"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=300)
    fig.suptitle('Dynamique: Training Loss vs Validation Loss', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    for i, res in enumerate(all_results):
        if i >= 6: break
        ax = axes[i]
        loss_smooth = np.convolve(res['train_loss'], np.ones(5)/5, mode='valid')
        steps_smooth = res['train_steps'][:len(loss_smooth)]
        
        ax.plot(steps_smooth, loss_smooth, label='Train Loss', color='#2ecc71', alpha=0.6)
        ax.plot(res['eval_steps'], res['eval_loss'], label='Val Loss', color='#e74c3c', linewidth=2.5)
        
        ax.set_title(res['name'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("DYNAMIQUE_LOSS.png")
    plt.close()
    print(f" Grille Loss sauvegardée : DYNAMIQUE_LOSS.png")

def plot_final_bar_chart(all_results):
    """
    Génère un histogramme comparatif des résultats finaux sur le TEST SET.
    C'est l'image qui 'renvoie les données'.
    """
    # Préparation des données pour Pandas
    data = []
    for res in all_results:
        data.append({
            "Model": res['name'],
            "Metric": "MRR",
            "Score": res['test_mrr']
        })
        data.append({
            "Model": res['name'],
            "Metric": "Recall@10",
            "Score": res['test_recall_10']
        })
        # On peut ajouter Recall@5 si besoin
        
    df = pd.DataFrame(data)

    plt.figure(figsize=(14, 8), dpi=300)
    
    # Création du Bar Chart groupé
    chart = sns.barplot(data=df, x="Model", y="Score", hue="Metric", palette="viridis")
    
    # Ajout des valeurs sur les barres (Data Labels)
    for container in chart.containers:
        chart.bar_label(container, fmt='%.3f', padding=3, fontweight='bold')

    plt.title("Performance Finale sur le Test Set (Bigger Batch is Better)", fontsize=18, fontweight='bold')
    plt.ylabel("Score", fontsize=14)
    plt.xlabel("Configuration Modèle", fontsize=14)
    plt.ylim(0, df['Score'].max() * 1.15) # Un peu de marge en haut
    plt.legend(title="Métrique", title_fontsize='13', fontsize='12')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig("RESULTATS_FINAUX_TEST.png")
    plt.close()
    print(f" Bar Chart Final sauvegardé : RESULTATS_FINAUX_TEST.png")

# ==========================================
# 3. MOTEUR D'EXPÉRIENCE
# ==========================================

def run_experiment(config, data_dict):
    print("\n" + "#"*60)
    print(f" RUN: {config['name']} | Model: {config['model_name']} | BS: {config['batch_size']}")
    print("#"*60 + "\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], trust_remote_code=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        collator = CFCollator(tokenizer, max_code_length=512, max_feedback_length=128)

        # Datasets
        train_dataset = CFDataset(data_dict["train"].to_list())
        val_dataset = CFDataset(data_dict["validation"].to_list())
        # Gestion du Test Set
        if "test" in data_dict:
            test_dataset = CFDataset(data_dict["test"].to_list())
        else:
            print(" Pas de 'test' split, usage validation comme test.")
            test_dataset = val_dataset

        lora_config = LoraConfig(
            r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            lora_dropout=0.05, bias="none", task_type=TaskType.FEATURE_EXTRACTION
        )

        model = BiEncoder(config['model_name'], lora_config, temperature=0.07)
        model.enable_input_require_grads()

        # Config Entraînement
        training_args = TrainingArguments(
            output_dir=f"./experiments/{config['name']}",
            num_train_epochs=config['epochs'],
            per_device_train_batch_size=config['batch_size'],
            per_device_eval_batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            bf16=True, fp16=False, gradient_checkpointing=True,
            logging_strategy="steps", logging_steps=10,    
            eval_strategy="steps", eval_steps=20,     
            save_strategy="steps", save_steps=20,               
            load_best_model_at_end=True, 
            metric_for_best_model="eval_mrr", 
            greater_is_better=True,      
            save_total_limit=1,          
            dataloader_num_workers=4,
            remove_unused_columns=False,
            report_to="none"
        )

        trainer = ContrastiveTrainer(
            model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=collator, compute_metrics=compute_metrics
        )

        # 1. Train
        trainer.train()
        
        # 2. Sauvegarde Best Model
        final_path = f"./final_best_models/{config['name']}"
        os.makedirs(final_path, exist_ok=True)
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f" Modèle sauvegardé : {final_path}")

        # 3. Evaluation Finale sur Test Set
        print(f" Evaluation Test Set...")
        test_output = trainer.predict(test_dataset)
        test_metrics = test_output.metrics
        
        # 4. Extraction Historique pour les courbes
        history = trainer.state.log_history
        train_steps, train_loss = [], []
        eval_steps, eval_mrr, eval_recall_5, eval_recall_10, eval_loss = [], [], [], [], []

        for log in history:
            if "loss" in log and "eval_loss" not in log:
                train_steps.append(log["step"])
                train_loss.append(log["loss"])
            if "eval_loss" in log:
                eval_steps.append(log["step"])
                eval_loss.append(log["eval_loss"])
                if "eval_mrr" in log: eval_mrr.append(log["eval_mrr"])
                if "eval_recall_at_5" in log: eval_recall_5.append(log["eval_recall_at_5"])
                if "eval_recall_at_10" in log: eval_recall_10.append(log["eval_recall_at_10"])

        # Nettoyage
        del model, trainer, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "name": config['name'],
            # Données pour les courbes
            "train_steps": train_steps, "train_loss": train_loss,
            "eval_steps": eval_steps, "eval_loss": eval_loss,
            "eval_mrr": eval_mrr, "eval_recall_at_5": eval_recall_5, "eval_recall_at_10": eval_recall_10,
            # Données pour le Bar Chart final
            "test_mrr": test_metrics.get('test_mrr', 0),
            "test_recall_10": test_metrics.get('test_recall_at_10', 0),
            "test_recall_5": test_metrics.get('test_recall_at_5', 0)
        }

    except Exception as e:
        print(f" ERREUR CRITIQUE sur {config['name']}: {e}")
        torch.cuda.empty_cache()
        return None

# ==========================================
# 4. MAIN & CONFIGURATIONS
# ==========================================

def main():
    print("Chargement du dataset...")
    data_dict = load_dataset('matis35/RAFT_CLEAN_V1')

    EXPERIMENTS = [
        # --- SFR (400M) ---
        {"name": "SFR_BS64",  "model_name": "Salesforce/SFR-Embedding-Code-400M_R", "batch_size": 64,  "learning_rate": 2e-4, "epochs": 10},
        {"name": "SFR_BS128", "model_name": "Salesforce/SFR-Embedding-Code-400M_R", "batch_size": 128, "learning_rate": 5e-4, "epochs": 20},
        {"name": "SFR_BS256", "model_name": "Salesforce/SFR-Embedding-Code-400M_R", "batch_size": 256, "learning_rate": 1e-3, "epochs": 40},
        
        # --- GEMMA (300M/2B) ---
        # Note: J'utilise google/embeddinggemma-300m comme demandé
        {"name": "Gemma_BS64", "model_name": "google/embeddinggemma-300m", "batch_size": 64, "learning_rate": 2e-4, "epochs": 10},
        {"name": "Gemma_BS128", "model_name": "google/embeddinggemma-300m", "batch_size": 128, "learning_rate": 5e-4, "epochs": 20},
        # Si BS256 plante pour Gemma, le script l'ignorera grâce au try/except
        {"name": "Gemma_BS256", "model_name": "google/embeddinggemma-300m", "batch_size": 256, "learning_rate": 1e-3, "epochs": 40},
    ]

    all_results = []

    for config in EXPERIMENTS:
        res = run_experiment(config, data_dict)
        if res is not None:
            all_results.append(res)

    if len(all_results) > 0:
        print("\n Génération des rendus finaux...")
        
        # 1. Les courbes dynamiques
        plot_curves(all_results, "eval_mrr", "DYNAMIQUE MRR", "DYNAMIQUE_MRR.png")
        plot_curves(all_results, "eval_recall_at_10", "DYNAMIQUE RECALL@10", "DYNAMIQUE_RECALL.png")
        plot_loss_grid(all_results)
        
        # 2. Le Bar Chart Final (Le Juge de Paix)
        plot_final_bar_chart(all_results)
        
    else:
        print("Aucun résultat valide.")

if __name__ == "__main__":
    main()