#!/usr/bin/env python3
"""
Merge existing Hugging Face dataset with new local data,
Then apply the ORIGINAL logic: Filter -> Class Encode -> Stratify (90/5/5) -> Push.
"""

from datasets import load_dataset, concatenate_datasets, DatasetDict, Value, ClassLabel
from collections import Counter
import os

# --- CONFIGURATION ---
EXISTING_REPO_ID = "matis35/cf-synt"
NEW_DATA_FILE = "dataset_c_piscine_semantic_chunk.jsonl"
TARGET_REPO_ID = "matis35/cf-synt_V2"
STRATIFY_COLUMN = "error_category"
MIN_SAMPLES_PER_CLASS = 30 # Sécurité pour le split 5%/5%
# ---------------------

def main():
    print("="*70)
    print("MERGE + ORIGINAL STRATIFICATION LOGIC")
    print("="*70)

    # ---------------------------------------------------------
    # ÉTAPE 1 : FUSION (La seule nouveauté)
    # ---------------------------------------------------------
    
    # A. Charger l'ancien dataset (Train + Val + Test)
    print(f"\n1. Loading existing dataset from '{EXISTING_REPO_ID}'...")
    try:
        existing_ds_dict = load_dataset(EXISTING_REPO_ID)
        existing_full = concatenate_datasets([
            existing_ds_dict['train'], 
            existing_ds_dict['validation'], 
            existing_ds_dict['test']
        ])
        
        # IMPORTANT : Pour fusionner, il faut que tout soit du texte brut au départ.
        # Si l'ancien est déjà un ClassLabel (chiffres), on le remet en string pour la fusion.
        if isinstance(existing_full.features[STRATIFY_COLUMN], ClassLabel):
            print("   (Converting existing ClassLabel back to string for clean merging...)")
            feat = existing_full.features[STRATIFY_COLUMN]
            existing_full = existing_full.map(lambda x: {STRATIFY_COLUMN: feat.int2str(x[STRATIFY_COLUMN])})
        
        # On force le type string pour être sûr
        existing_full = existing_full.cast_column(STRATIFY_COLUMN, Value("string"))
        print(f"   ✅ Existing data: {len(existing_full):,} samples.")

    except Exception as e:
        print(f"   ❌ Error loading existing dataset (Starting fresh?): {e}")
        existing_full = None

    # B. Charger le nouveau fichier local
    print(f"\n2. Loading new local data from '{NEW_DATA_FILE}'...")
    new_ds = load_dataset('json', data_files=NEW_DATA_FILE, split="train")
    new_ds = new_ds.cast_column(STRATIFY_COLUMN, Value("string"))
    print(f"   ✅ New data: {len(new_ds):,} samples.")

    # C. Concaténer
    if existing_full:
        full_dataset = concatenate_datasets([existing_full, new_ds])
    else:
        full_dataset = new_ds

    original_len = len(full_dataset)
    print(f"\n   ➡️  TOTAL MERGED: {original_len:,} samples.")

    # ---------------------------------------------------------
    # ÉTAPE 2 : TA LOGIQUE ORIGINALE (Filtrage + Class Encode)
    # ---------------------------------------------------------

    print(f"\n3. Analyzing class distribution for '{STRATIFY_COLUMN}'...")
    labels = full_dataset[STRATIFY_COLUMN]
    counts = Counter(labels)
    
    # Filtrage des classes rares
    rare_classes = [cls for cls, count in counts.items() if count < MIN_SAMPLES_PER_CLASS]
    
    if rare_classes:
        print(f"⚠️  Found {len(rare_classes)} rare categories.")
        full_dataset = full_dataset.filter(lambda x: x[STRATIFY_COLUMN] not in rare_classes)
        removed_count = original_len - len(full_dataset)
        print(f"   Removed {removed_count} samples. New total: {len(full_dataset):,}")
    else:
        print("✓ All classes have enough samples.")

    # --- LE RETOUR DE TA LOGIQUE PRÉFÉRÉE ---
    print(f"\n4. Casting column '{STRATIFY_COLUMN}' to ClassLabel (Original Logic)...")
    # C'est cette ligne qui transforme le String en Int (ClassLabel) et génère la map interne
    full_dataset = full_dataset.class_encode_column(STRATIFY_COLUMN)
    # ----------------------------------------

    # ---------------------------------------------------------
    # ÉTAPE 3 : STRATIFICATION & PUSH
    # ---------------------------------------------------------

    print(f"5. Splitting Train (90%) / Rest (10%)...")
    train_test_split = full_dataset.train_test_split(
        test_size=0.1, 
        stratify_by_column=STRATIFY_COLUMN,
        seed=42
    )
    
    train_dataset = train_test_split['train']
    rest_dataset = train_test_split['test']

    print(f"6. Splitting Rest into Validation (5%) / Test (5%)...")
    val_test_split = rest_dataset.train_test_split(
        test_size=0.5,
        stratify_by_column=STRATIFY_COLUMN,
        seed=42
    )
    
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
    print(f"\nPushing to {TARGET_REPO_ID}...")
    final_dataset.push_to_hub(TARGET_REPO_ID, private=False)

    print(f"\n✓ Dataset pushed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()