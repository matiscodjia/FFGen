#!/usr/bin/env python3
"""
Push cleaned dataset to HuggingFace Hub with Stratified Split (90/5/5)
Includes filtering of rare classes to prevent stratification errors.
"""

from datasets import load_dataset, DatasetDict
from collections import Counter
import os

# --- CONFIGURATION ---
INPUT_FILE = "dataset_c_piscine_semantic_validated.jsonl"
STRATIFY_COLUMN = "error_category" 
REPO_ID = "matis35/cf-synt"
MIN_SAMPLES_PER_CLASS = 20 # Minimum requis pour éviter le crash (conseillé : 10+)
# ---------------------

def main():
    print("="*70)
    print("LOADING, FILTERING AND STRATIFYING (90/5/5)")
    print("="*70)

    # 1. Chargement
    print(f"Loading data from: {INPUT_FILE}")
    full_dataset = load_dataset('json', data_files=INPUT_FILE, split="train")
    original_len = len(full_dataset)
    print(f"Loaded: {original_len:,} samples")

    # --- ÉTAPE DE NETTOYAGE (NOUVEAU) ---
    print(f"\nAnalyzing class distribution for '{STRATIFY_COLUMN}'...")
    
    # On compte combien de fois chaque erreur apparaît
    labels = full_dataset[STRATIFY_COLUMN]
    counts = Counter(labels)
    
    # On identifie les classes trop rares
    rare_classes = [cls for cls, count in counts.items() if count < MIN_SAMPLES_PER_CLASS]
    
    if rare_classes:
        print(f"⚠️  Found {len(rare_classes)} rare categories (freq < {MIN_SAMPLES_PER_CLASS}).")
        print(f"   Examples: {rare_classes[:3]}...")
        print(f"   Action: Removing rows with these rare classes to allow stratification.")
        
        # On filtre : on ne garde que les lignes dont la classe n'est PAS dans rare_classes
        full_dataset = full_dataset.filter(lambda x: x[STRATIFY_COLUMN] not in rare_classes)
        
        removed_count = original_len - len(full_dataset)
        print(f"   Removed {removed_count} samples. New total: {len(full_dataset):,}")
    else:
        print("✓ All classes have enough samples.")

    # --- CASTING ---
    print(f"\nCasting column '{STRATIFY_COLUMN}' to ClassLabel...")
    full_dataset = full_dataset.class_encode_column(STRATIFY_COLUMN)
    # ------------------------------

    # 2. Premier Split : Train (90%) / Reste (10%)
    print(f"Splitting Train (90%) / Rest (10%)...")
    train_test_split = full_dataset.train_test_split(
        test_size=0.1, 
        stratify_by_column=STRATIFY_COLUMN,
        seed=42
    )
    
    train_dataset = train_test_split['train']
    rest_dataset = train_test_split['test']

    # 3. Second Split : Validation (5%) / Test (5%)
    print(f"Splitting Rest into Validation (5%) / Test (5%)...")
    
    # Note: Si le dataset "rest" est très petit, certaines classes rares peuvent encore bloquer ici.
    # Si ça plante ici, augmentez MIN_SAMPLES_PER_CLASS à 10 ou 20.
    val_test_split = rest_dataset.train_test_split(
        test_size=0.5,
        stratify_by_column=STRATIFY_COLUMN,
        seed=42
    )
    
    # 4. Finalisation
    final_dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })

    # Stats
    total = len(full_dataset)
    print(f"\nFinal Split Statistics:")
    print(f"  Train:      {len(final_dataset['train']):,} ({len(final_dataset['train'])/total:.1%})")
    print(f"  Validation: {len(final_dataset['validation']):,} ({len(final_dataset['validation'])/total:.1%})")
    print(f"  Test:       {len(final_dataset['test']):,} ({len(final_dataset['test'])/total:.1%})")

    # Push
    print(f"\nPushing to {REPO_ID}...")
    final_dataset.push_to_hub(REPO_ID, private=False)

    print(f"\n✓ Dataset pushed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()