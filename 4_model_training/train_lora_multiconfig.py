#!/usr/bin/env python3
"""
Multi-Configuration LoRA Training Script

Ce script permet de lancer plusieurs entraînements LoRA avec des configurations différentes.
Suffit d'éditer le dictionnaire TRAINING_CONFIGS pour définir les paramètres.

Features:
- Support de multiples configurations d'entraînement
- Sauvegarde automatique des meilleurs modèles uniquement
- Logs et graphiques organisés dans ./logs
- Modèles finaux dans ./lora_output
- Nettoyage automatique des checkpoints intermédiaires
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import shutil
from datetime import datetime
from typing import Dict, List
import torch

# Import du script d'entraînement LoRA existant
from train_embedding_lora import train_lora
from argparse import Namespace


TRAINING_CONFIGS = [
    {
        "name": "gemma_300m_r16_8bit",
        "model": "google/embeddinggemma-300m",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        "use_8bit": True,
        "gradient_checkpointing": True,
        "batch_size": 8,
        "epochs": 15,
        "lr": 3e-5,
        "margin": 0.5,
        "max_length": 512,
        "patience": 5,
    },
    {
        "name": "sfr_400m_r16_8bit",
        "model": "Salesforce/SFR-Embedding-Code-400M_R",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        "use_8bit": True,
        "gradient_checkpointing": True,
        "batch_size": 8,
        "epochs": 15,
        "lr": 3e-5,
        "margin": 0.5,
        "max_length": 512,
        "patience": 5,
    },
    {
        "name": "minilm_r16_8bit",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        "use_8bit": True,
        "gradient_checkpointing": True,
        "batch_size": 8,
        "epochs": 15,
        "lr": 3e-5,
        "margin": 0.5,
        "max_length": 512,
        "patience": 5,
    },
    
  
]

# Dataset par défaut (multi_neg_10)
DEFAULT_DATASET = "data/Exp-002-llama3B_v2_multi_neg_10.jsonl"

# Nombre maximum d'exemples (None = tout le dataset)
MAX_SAMPLES = None

# Dossiers de sortie
OUTPUT_ROOT = Path("lora_output")
LOGS_ROOT = Path("logs")


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def create_experiment_id(config_name: str) -> str:
    """Créer un ID unique pour l'expérience"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config_name}_{timestamp}"


def save_config_summary(exp_id: str, config: Dict, output_dir: Path):
    """Sauvegarder la configuration de l'expérience"""
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "config": config
        }, f, indent=2)
    print(f"   Configuration saved to {config_file}")


def cleanup_intermediate_checkpoints(output_dir: Path):
    """
    Nettoyer les checkpoints intermédiaires, ne garder que best_model
    """
    for item in output_dir.iterdir():
        if item.is_dir() and item.name != "best_model":
            print(f"   Cleaning up intermediate checkpoint: {item}")
            shutil.rmtree(item)


def move_best_model_to_output(temp_dir: Path, final_dir: Path, exp_id: str):
    """
    Déplacer le meilleur modèle vers le dossier de sortie final
    """
    best_model_src = temp_dir / "best_model"
    best_model_dst = final_dir / exp_id

    if best_model_src.exists():
        # Créer le dossier de destination
        best_model_dst.mkdir(parents=True, exist_ok=True)

        # Copier tous les fichiers du meilleur modèle
        for item in best_model_src.iterdir():
            if item.is_file():
                shutil.copy2(item, best_model_dst / item.name)
            elif item.is_dir():
                shutil.copytree(item, best_model_dst / item.name, dirs_exist_ok=True)

        print(f"   Best model moved to {best_model_dst}")
        return best_model_dst
    else:
        print(f"   Warning: No best_model found in {temp_dir}")
        return None


def move_logs_to_logs_dir(temp_dir: Path, logs_dir: Path, exp_id: str):
    """
    Déplacer les fichiers de logs vers le dossier logs/
    """
    training_logs_src = temp_dir / "training_logs"

    if training_logs_src.exists():
        logs_exp_dir = logs_dir / exp_id
        logs_exp_dir.mkdir(parents=True, exist_ok=True)

        # Copier tous les .png et autres fichiers de log
        for item in training_logs_src.iterdir():
            if item.is_file() and (item.suffix == ".png" or item.suffix == ".json"):
                shutil.copy2(item, logs_exp_dir / item.name)

        print(f"   Training logs moved to {logs_exp_dir}")
        return logs_exp_dir
    else:
        print(f"   Warning: No training_logs found in {temp_dir}")
        return None


def get_device_info():
    """Obtenir des informations sur le device disponible"""
    if torch.cuda.is_available():
        device = "CUDA"
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{device} - {device_name} ({memory:.1f}GB)"
    elif torch.backends.mps.is_available():
        return "MPS (Apple Silicon)"
    else:
        return "CPU"


# ============================================================================
# FONCTION PRINCIPALE D'ENTRAÎNEMENT MULTI-CONFIG
# ============================================================================

def run_multi_config_training(
    configs: List[Dict],
    dataset_path: str,
    max_samples: int = None,
    output_root: Path = OUTPUT_ROOT,
    logs_root: Path = LOGS_ROOT
):
    """
    Lancer plusieurs entraînements LoRA avec différentes configurations

    Args:
        configs: Liste de dictionnaires de configuration
        dataset_path: Chemin vers le dataset JSONL
        max_samples: Nombre max d'exemples (None = tous)
        output_root: Dossier racine pour les modèles
        logs_root: Dossier racine pour les logs
    """

    # Vérifier que le dataset existe
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Créer les dossiers de sortie
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    # Informations générales
    print("\n" + "="*80)
    print(" MULTI-CONFIGURATION LORA TRAINING")
    print("="*80)
    print(f"  Device: {get_device_info()}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Max samples: {max_samples if max_samples else 'ALL'}")
    print(f"  Number of configurations: {len(configs)}")
    print(f"  Output directory: {output_root}")
    print(f"  Logs directory: {logs_root}")
    print("="*80 + "\n")

    # Résultats globaux
    all_results = []

    # Lancer chaque configuration
    for idx, config in enumerate(configs, 1):
        config_name = config.get("name", f"config_{idx}")
        exp_id = create_experiment_id(config_name)

        print("\n" + "="*80)
        print(f" EXPERIMENT {idx}/{len(configs)}: {config_name}")
        print("="*80)

        # Afficher la configuration
        print("\nConfiguration:")
        for key, value in config.items():
            if key != "name":
                print(f"  {key}: {value}")
        print("")

        # Créer un dossier temporaire pour cette expérience
        temp_output_dir = Path(f"temp_training_{exp_id}")
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Créer les arguments pour train_lora
            args = Namespace(
                data=dataset_path,
                max_samples=max_samples,
                model=config.get("model", "google/embeddinggemma-300m"),
                max_length=config.get("max_length", 512),
                lora_r=config.get("lora_r", 8),
                lora_alpha=config.get("lora_alpha", 16),
                lora_dropout=config.get("lora_dropout", 0.05),
                target_modules=config.get("target_modules", None),
                batch_size=config.get("batch_size", 4),
                epochs=config.get("epochs", 10),
                lr=config.get("lr", 2e-5),
                margin=config.get("margin", 0.5),
                patience=config.get("patience", 5),
                use_8bit=config.get("use_8bit", False),
                gradient_checkpointing=config.get("gradient_checkpointing", False),
                output_dir=str(temp_output_dir)
            )

            # Sauvegarder la configuration
            save_config_summary(exp_id, config, temp_output_dir)

            # Lancer l'entraînement
            print(f"\nStarting training for {exp_id}...\n")
            result = train_lora(args, custom_save_dir=str(temp_output_dir))

            # Ajouter des métadonnées au résultat
            result['experiment_id'] = exp_id
            result['config_name'] = config_name
            result['config'] = config

            # Nettoyer les checkpoints intermédiaires
            cleanup_intermediate_checkpoints(temp_output_dir)

            # Déplacer le meilleur modèle vers lora_output/
            final_model_path = move_best_model_to_output(
                temp_output_dir,
                output_root,
                exp_id
            )
            if final_model_path:
                result['final_model_path'] = str(final_model_path)

            # Déplacer les logs vers logs/
            logs_path = move_logs_to_logs_dir(
                temp_output_dir,
                logs_root,
                exp_id
            )
            if logs_path:
                result['logs_path'] = str(logs_path)

            # Nettoyer le dossier temporaire
            shutil.rmtree(temp_output_dir)

            all_results.append(result)

            print(f"\n Experiment {exp_id} completed successfully!")
            print(f"  Best val loss: {result.get('final_test_loss', 'N/A'):.4f}")
            print(f"  Model saved to: {result.get('final_model_path', 'N/A')}")
            print(f"  Logs saved to: {result.get('logs_path', 'N/A')}")

        except Exception as e:
            print(f"\n ERROR in experiment {exp_id}: {str(e)}")
            import traceback
            traceback.print_exc()

            # Nettoyer le dossier temporaire en cas d'erreur
            if temp_output_dir.exists():
                shutil.rmtree(temp_output_dir)

            all_results.append({
                'experiment_id': exp_id,
                'config_name': config_name,
                'config': config,
                'error': str(e),
                'status': 'failed'
            })

            # Continuer avec la configuration suivante
            continue

    # Sauvegarder le résumé final
    save_final_summary(all_results, output_root, logs_root)

    print("\n" + "="*80)
    print(" ALL EXPERIMENTS COMPLETED")
    print("="*80)
    print(f"  Total experiments: {len(configs)}")
    print(f"  Successful: {sum(1 for r in all_results if r.get('status') != 'failed')}")
    print(f"  Failed: {sum(1 for r in all_results if r.get('status') == 'failed')}")
    print(f"  Results saved to: {output_root}")
    print("="*80 + "\n")

    return all_results


def save_final_summary(results: List[Dict], output_root: Path, logs_root: Path):
    """
    Sauvegarder un résumé final de tous les entraînements
    """
    summary_file = logs_root / "training_summary.json"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r.get('status') != 'failed'),
        "failed": sum(1 for r in results if r.get('status') == 'failed'),
        "experiments": []
    }

    for result in results:
        summary["experiments"].append({
            "experiment_id": result.get('experiment_id'),
            "config_name": result.get('config_name'),
            "status": result.get('status', 'success'),
            "final_test_loss": result.get('final_test_loss'),
            "final_model_path": result.get('final_model_path'),
            "logs_path": result.get('logs_path'),
            "error": result.get('error'),
        })

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n Final summary saved to {summary_file}")

    # Trouver le meilleur modèle
    successful_results = [r for r in results if r.get('status') != 'failed']
    if successful_results:
        best_result = min(successful_results, key=lambda x: x.get('final_test_loss', float('inf')))
        print(f"\n BEST MODEL:")
        print(f"  Experiment: {best_result.get('experiment_id')}")
        print(f"  Config: {best_result.get('config_name')}")
        print(f"  Val Loss: {best_result.get('final_test_loss'):.4f}")
        print(f"  Path: {best_result.get('final_model_path')}")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Multi-configuration LoRA training script'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET,
        help=f'Path to JSONL dataset (default: {DEFAULT_DATASET})'
    )

    parser.add_argument(
        '--max-samples',
        type=int,
        default=MAX_SAMPLES,
        help='Maximum number of samples to use (default: None = all)'
    )

    parser.add_argument(
        '--configs-only',
        action='store_true',
        help='Only show configurations without training'
    )

    args = parser.parse_args()

    # Afficher les configurations si demandé
    if args.configs_only:
        print("\n" + "="*80)
        print(" CONFIGURED EXPERIMENTS")
        print("="*80)
        for idx, config in enumerate(TRAINING_CONFIGS, 1):
            print(f"\n{idx}. {config.get('name', f'config_{idx}')}")
            for key, value in config.items():
                if key != "name":
                    print(f"   {key}: {value}")
        print("\n" + "="*80 + "\n")
    else:
        # Lancer les entraînements
        results = run_multi_config_training(
            configs=TRAINING_CONFIGS,
            dataset_path=args.dataset,
            max_samples=args.max_samples,
            output_root=OUTPUT_ROOT,
            logs_root=LOGS_ROOT
        )
