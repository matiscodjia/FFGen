import torch
from torch import nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel, Trainer
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

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Forward Pass
        # Le "**inputs" déballe le dictionnaire du collator directement dans les arguments du forward
        code_emb, feedback_emb = model(**inputs)
        
        # 2. Calcul de la Matrice de Similarité
        # On multiplie Code (NxD) par Feedback Transposé (DxN) -> Matrice (NxN)
        # On divise par la température (ex: 0.07) pour "picoter" la distribution
        temperature = model.temperature
        similarity_matrix = torch.matmul(code_emb, feedback_emb.T) / temperature
        
        # 3. Création des Labels (La Vérité est sur la diagonale)
        # Si batch_size = 4, labels = [0, 1, 2, 3]
        batch_size = code_emb.size(0)
        labels = torch.arange(batch_size).to(code_emb.device)
        
        # 4. Calcul de la Loss (Symétrique pour plus de robustesse)
        # Code -> Feedback : Est-ce que le Code A retrouve le Feedback A ?
        loss_c2f = F.cross_entropy(similarity_matrix, labels)
        
        # Feedback -> Code : Est-ce que le Feedback A retrouve le Code A ?
        loss_f2c = F.cross_entropy(similarity_matrix.T, labels)
        
        loss = (loss_c2f + loss_f2c) / 2
        
        # 5. (Optionnel mais recommandé) Nos Métriques Custom
        # On veut savoir si le modèle devine juste, pas juste la valeur de la loss.
        with torch.no_grad():
            # Quelle colonne a le score le plus haut pour chaque ligne ?
            predicted_indices = torch.argmax(similarity_matrix, dim=1)
            accuracy = (predicted_indices == labels).float().mean()
            
            # On loggue directement dans le système de tracking de HF
            self.log({"batch_accuracy": accuracy.item()})

        # Le Trainer attend (loss, outputs) si return_outputs est True
        return (loss, (code_emb, feedback_emb)) if return_outputs else loss
    


def main():
    from transformers import TrainingArguments, AutoTokenizer
    from peft import LoraConfig, TaskType

    # ==========================================
    # 1. CONFIGURATION RAPIDE
    # ==========================================
    model_name = "Salesforce/SFR-Embedding-Code-400M_R"



    # On charge le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # On instancie nos classes
    dataset = CFDataset(data_dict["train"].to_list())
    val_dataset = CFDataset(data_dict["validation"].to_list())
    collator = CFCollator(tokenizer, max_code_length=512, max_feedback_length=128)

    # ==========================================
    # 3. PRÉPARATION DU MODÈLE
    # ==========================================
    # Config LoRA (Allège le modèle pour l'entraînement)
    lora_config = LoraConfig(
        r=128,                 # Rang de la matrice (plus petit = moins de paramètres)
        lora_alpha=16,       # Facteur d'échelle
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Adapte selon le modèle !
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # Attention : pour SFR-Embedding, les target_modules peuvent varier.
    # Si tu as une erreur, essaie juste ["query", "value"] ou regarde le nom des couches.

    model = BiEncoder(
        base_model_name=model_name,
        lora_config=lora_config,
        temperature=0.07
    )

    # ==========================================
    # 4. LANCEMENT DE L'ENTRAÎNEMENT
    # ==========================================
    training_args = TrainingArguments(
        output_dir="./test_trainer",
        num_train_epochs=10,              # Juste 2 époques pour tester
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,   # Petit batch pour éviter l'OOM (Out of Memory)
        learning_rate=1e-4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1000,
        eval_steps=1000,  
        logging_dir='./logs',              
        remove_unused_columns=False,     
        report_to="tensorboard",                        
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=collator
    )

    print("\nDémarrage de l'entraînement test...")
    trainer.train()
    print("✅ Entraînement terminé avec succès !")


if __name__ == "__main__":
    main()