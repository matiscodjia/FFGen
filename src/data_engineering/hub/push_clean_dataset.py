#!/usr/bin/env python3
"""
Push cleaned dataset to HuggingFace Hub with Stratified Split (90/5/5)
"""

from datasets import load_dataset, DatasetDict
import os

# --- CONFIGURATION ---
# REMPLACEZ CECI par le nom de la colonne qui contient vos classes/labels
# (ex: 'category', 'label', 'intent', etc.)
STRATIFY_COLUMN = "label" 
REPO_ID = "matis35/cf-synt"
# ---------------------

def main():
    print("="*70)
    print("LOADING AND STRATIFYING DATASET (90/5/5)")
    print("="*70)
    
    # split="train" permet de charger le tout comme un seul Dataset, pas un DatasetDict
    full_dataset = load_dataset('json', data_files=data_files, split="train")

    print(f"\nTotal samples loaded: {len(full_dataset):,}")

    # 2. Premier Split : Séparer Train (90%) du Reste (10%)
    # test_size=0.1 correspond aux 10% restants (qui deviendront Val + Test)
    print(f"Splitting Train (90%) / Rest (10%) stratifying on '{STRATIFY_COLUMN}'...")
    
    train_test_split = full_dataset.train_test_split(
        test_size=0.1, 
        stratify_by_column=STRATIFY_COLUMN,
        seed=42 # Pour la reproductibilité
    )
    
    train_dataset = train_test_split['train']
    rest_dataset = train_test_split['test'] # Ce sont les 10% temporaires

    # 3. Second Split : Séparer le Reste en Validation (5%) et Test (5%)
    # On divise les 10% restants en deux parts égales (0.5), soit 5% du total global chacun
    print(f"Splitting Rest into Validation (5%) / Test (5%)...")
    
    val_test_split = rest_dataset.train_test_split(
        test_size=0.5,
        stratify_by_column=STRATIFY_COLUMN,
        seed=42
    )
    
    # 4. Reconstitution de l'objet DatasetDict final
    final_dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_test_split['train'], # La première moitié du reste
        'test': val_test_split['test']         # La seconde moitié du reste
    })

    print(f"\nFinal Split Statistics:")
    print(f"  Train:      {len(final_dataset['train']):,} ({len(final_dataset['train'])/len(full_dataset):.1%})")
    print(f"  Validation: {len(final_dataset['validation']):,} ({len(final_dataset['validation'])/len(full_dataset):.1%})")
    print(f"  Test:       {len(final_dataset['test']):,} ({len(final_dataset['test'])/len(full_dataset):.1%})")

    # Push to hub
    print(f"\nPushing to {REPO_ID}...")
    final_dataset.push_to_hub(REPO_ID, private=False)

    print(f"\n✓ Dataset pushed successfully!")
    print(f"  URL: https://huggingface.co/datasets/{REPO_ID}")
    print("="*70)

if __name__ == "__main__":
    main()