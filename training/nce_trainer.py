import torch
from torch import nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from peft import get_peft_model
import torch.nn.functional as F

data_dict = load_dataset('matis35/RAFT')
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

        return {"code_input_id" : codes_encoding["input_ids"],
                "feedback_input_id" : feedbacks_encoding["input_ids"],
                "code_attention_mask" : codes_encoding["attention_mask"],
                "feedback_attention_mask" : feedbacks_encoding["attention_mask"] }



class BiEncoder(nn.Module):
    def __init__(self, base_model_name, lora_config, temperature):
        super().__init__()
        self.base_model_name = base_model_name
        self.encoder = AutoModel.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.temperature = temperature
        print(f"\nModel architecture:")
        self.encoder.print_trainable_parameters()
    
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
        
    def forward(self, code_input_id, code_attention_mask, feedback_input_id, feedback_attention_mask):
        code_embeddings = self.encode(code_input_id, code_attention_mask)
        
        feedback_embeddings = self.encode(feedback_input_id, feedback_attention_mask)
        
        return code_embeddings, feedback_embeddings
    


from transformers import Trainer
import torch.nn.functional as F

from transformers import Trainer
import torch.nn.functional as F

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        code_emb, feedback_emb = model(**inputs)
        temperature = model.temperature
        similarity_matrix = torch.matmul(code_emb, feedback_emb.T) / temperature
        
        batch_size = code_emb.size(0)
        labels = torch.arange(batch_size).to(code_emb.device)
        
        # 2. Loss (Comme avant)
        loss_c2f = F.cross_entropy(similarity_matrix, labels)
        loss_f2c = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_c2f + loss_f2c) / 2
        
        # 3. METRIQUES AVANCÉES (MRR, Recall)
        # On le fait dans un bloc 'no_grad' pour ne pas exploser la mémoire
        with torch.no_grad():
            self.log_ranking_metrics(similarity_matrix, labels)

        return (loss, (code_emb, feedback_emb)) if return_outputs else loss

    def log_ranking_metrics(self, similarity_matrix, labels):
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
        hits = (sorted_indices == labels.view(-1, 1))
        
        # C. On récupère le rang (l'index où hits est True)
        # nonzero() renvoie les coordonnées [ligne, colonne]. La colonne EST le rang (0-indexed)
        # On ajoute +1 car le rang commence humainement à 1
        ranks = hits.nonzero(as_tuple=True)[1].float() + 1
        
        # D. Calcul des Scores
        
        # MRR : Moyenne des inverses des rangs (1/1, 1/2, 1/3...)
        mrr = (1 / ranks).mean().item()
        
        # Recall@1 : Combien de rangs valent 1 ? (C'est ton accuracy)
        r1 = (ranks <= 1).float().mean().item()
        
        # Recall@5 : Combien de rangs sont <= 5 ?
        r5 = (ranks <= 5).float().mean().item()
        
        # Recall@10 : Combien de rangs sont <= 10 ?
        r10 = (ranks <= 10).float().mean().item()
        
        self.log({
            "mrr": mrr,
            "recall_at_1": r1,
            "recall_at_5": r5,
            "recall_at_10": r10
        })


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

# ==========================================
# 4. ARGUMENTS D'ENTRAÎNEMENT (OPTIMISÉ)
# ==========================================
training_args = TrainingArguments(
    output_dir="./test_trainer",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-4,
    
    # --- Visibilité (TensorBoard) ---
    logging_dir='./logs',
    report_to="tensorboard",
    logging_strategy="steps",
    logging_steps=50,    

    eval_strategy="steps",
    eval_steps=200,     
    
    save_strategy="steps",
    save_steps=200,     
    save_total_limit=2,  
    
    load_best_model_at_end=True,      
    metric_for_best_model="eval_mrr", 
    greater_is_better=True,           
    
    remove_unused_columns=False
)

trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    data_collator=collator
)

print("\nDémarrage de l'entraînement optimisé...")
trainer.train()

trainer.save_model("./test_trainer/best_model_final")
print("Entraînement terminé. Meilleur modèle (basé sur MRR) sauvegardé.")


if __name__ == "__main__":
    main()