import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_ID = "matis35/cf-synt_V2"
SPLIT = "train"  # On analyse le train, c'est lui qui forge le modèle

# Configuration Bi-Encoder (Gemma)
BI_MODEL_ID = "google/embeddinggemma-300m"
BI_MAX_CODE = 512
BI_MAX_FEEDBACK = 128

# Configuration Cross-Encoder (CodeBERT)
CROSS_MODEL_ID = "microsoft/codebert-base"
CROSS_MAX_TOTAL = 512

# ==========================================
# 1. NETTOYAGE (LE MÊME QUE L'ENTRAINEMENT)
# ==========================================
def clean_c_code(code_string):
    if not code_string: return ""
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    return code_string.strip()

# ==========================================
# 2. ANALYSEUR
# ==========================================
def analyze_bi_encoder(dataset):
    print(f"\n🔵 --- ANALYSE BI-ENCODER ({BI_MODEL_ID}) ---")
    tokenizer = AutoTokenizer.from_pretrained(BI_MODEL_ID)
    
    code_lengths = []
    feedback_lengths = []
    truncated_codes = 0
    truncated_feedbacks = 0
    severe_truncation = 0 # Plus de 50% du contenu perdu
    
    print("   Tokenization en cours...")
    for item in tqdm(dataset):
        c = clean_c_code(item["code"])
        f = item["feedback"]
        
        # On tokenise sans tronquer pour avoir la vraie longueur
        len_c = len(tokenizer(c, truncation=False)["input_ids"])
        len_f = len(tokenizer(f, truncation=False)["input_ids"])
        
        code_lengths.append(len_c)
        feedback_lengths.append(len_f)
        
        if len_c > BI_MAX_CODE: 
            truncated_codes += 1
            if len_c > BI_MAX_CODE * 2: severe_truncation += 1
            
        if len_f > BI_MAX_FEEDBACK: 
            truncated_feedbacks += 1

    total = len(dataset)
    
    print(f"\n📊 RÉSULTATS BI-ENCODER (Limites: Code={BI_MAX_CODE}, Feedback={BI_MAX_FEEDBACK})")
    print(f"   ► Codes Tronqués      : {truncated_codes}/{total} ({truncated_codes/total:.2%})")
    print(f"   ► Feedbacks Tronqués  : {truncated_feedbacks}/{total} ({truncated_feedbacks/total:.2%})")
    print(f"   ► Codes > 2x la limite: {severe_truncation}/{total} ({severe_truncation/total:.2%}) ⚠️ DANGER")
    print(f"   ► Longueur Moyenne Code : {int(np.mean(code_lengths))} tokens (Max: {np.max(code_lengths)})")
    print(f"   ► Longueur Moyenne FB   : {int(np.mean(feedback_lengths))} tokens (Max: {np.max(feedback_lengths)})")
    
    return code_lengths, feedback_lengths

def analyze_cross_encoder(dataset):
    print(f"\n🟣 --- ANALYSE CROSS-ENCODER ({CROSS_MODEL_ID}) ---")
    tokenizer = AutoTokenizer.from_pretrained(CROSS_MODEL_ID)
    
    # CodeBERT a besoin de tokens spéciaux : [CLS] Code [SEP] Feedback [SEP]
    # Cela consomme 3 tokens. Donc la place réelle est 512 - 3 = 509.
    effective_limit = CROSS_MAX_TOTAL - 3
    
    pair_lengths = []
    truncated_pairs = 0
    lost_feedbacks = 0 # Cas où le feedback est touché (si on coupe la fin)
    
    print("   Tokenization en cours...")
    for item in tqdm(dataset):
        c = clean_c_code(item["code"])
        f = item["feedback"]
        
        # Tokenization individuelle pour estimer la composition
        len_c = len(tokenizer(c, add_special_tokens=False)["input_ids"])
        len_f = len(tokenizer(f, add_special_tokens=False)["input_ids"])
        total_len = len_c + len_f
        
        pair_lengths.append(total_len)
        
        if total_len > effective_limit:
            truncated_pairs += 1
            # Si on tronque, et qu'on utilise truncation="only_first" (on coupe le code),
            # le feedback est sauvé.
            # MAIS si le code est immense (ex: 1000 tokens), il ne restera plus de place pour lui.
            # Estimons si le code restant est trop petit (< 50 tokens)
            if (effective_limit - len_f) < 50:
                lost_feedbacks += 1 # On considère que le couple est cassé car code quasi inexistant
    
    total = len(dataset)
    
    print(f"\n📊 RÉSULTATS CROSS-ENCODER (Limite Totale={CROSS_MAX_TOTAL})")
    print(f"   ► Paires Tronquées        : {truncated_pairs}/{total} ({truncated_pairs/total:.2%})")
    print(f"   ► Situations Critiques    : {lost_feedbacks}/{total} ({lost_feedbacks/total:.2%})")
    print(f"     (Cas où le feedback est si long ou le code si long que l'un écrase l'autre)")
    print(f"   ► Longueur Moyenne Paire  : {int(np.mean(pair_lengths))} tokens (Max: {np.max(pair_lengths)})")
    
    return pair_lengths

# ==========================================
# 3. PLOTTING
# ==========================================
def plot_distributions(bi_c, bi_f, cross_p):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot Code (Bi)
    sns.histplot(bi_c, ax=axes[0], color='skyblue', bins=50)
    axes[0].axvline(BI_MAX_CODE, color='red', linestyle='--', label=f'Limit ({BI_MAX_CODE})')
    axes[0].set_title(f"Distribution Longueur Codes (Gemma)")
    axes[0].legend()
    
    # Plot Feedback (Bi)
    sns.histplot(bi_f, ax=axes[1], color='orange', bins=50)
    axes[1].axvline(BI_MAX_FEEDBACK, color='red', linestyle='--', label=f'Limit ({BI_MAX_FEEDBACK})')
    axes[1].set_title(f"Distribution Longueur Feedbacks (Gemma)")
    axes[1].legend()
    
    # Plot Paires (Cross)
    sns.histplot(cross_p, ax=axes[2], color='purple', bins=50)
    axes[2].axvline(CROSS_MAX_TOTAL, color='red', linestyle='--', label=f'Limit ({CROSS_MAX_TOTAL})')
    axes[2].set_title(f"Distribution Paires Concaténées (CodeBERT)")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("truncation_analysis.png")
    print("\n💾 Graphique sauvegardé : truncation_analysis.png")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    dataset = load_dataset(DATASET_ID, split=SPLIT)
    
    # Analyse
    bi_codes, bi_feeds = analyze_bi_encoder(dataset)
    cross_pairs = analyze_cross_encoder(dataset)
    
    # Visualisation
    plot_distributions(bi_codes, bi_feeds, cross_pairs)