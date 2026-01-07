#!/usr/bin/env python3
"""
Monitoring en temps réel de la génération du dataset (avec métriques)
Usage: python scripts/monitor_generation.py
"""

import os
import json
import time
import statistics
from collections import Counter
from datetime import datetime

DATASET_FILE = "dataset_c_piscine_semantic_validated.jsonl"
METRICS_FILE = "generation_metrics.jsonl"
SUMMARY_FILE = "generation_summary.json"
REFRESH_INTERVAL = 5  # secondes

def analyze_dataset():
    """Analyse le dataset actuel"""
    if not os.path.exists(DATASET_FILE):
        return None

    total = 0
    difficulties = []
    error_categories = []
    tags_all = []

    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                total += 1
                try:
                    data = json.loads(line)
                    difficulties.append(data.get('difficulty', 'unknown'))
                    error_categories.append(data.get('error_category', 'unknown'))
                    tags_all.extend(data.get('tags', []))
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        return {"error": str(e)}

    return {
        "total": total,
        "difficulties": Counter(difficulties),
        "error_categories": Counter(error_categories),
        "tags": Counter(tags_all)
    }

def load_metrics():
    """Charge les métriques de génération"""
    if not os.path.exists(METRICS_FILE):
        return None

    metrics = []
    try:
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                metrics.append(json.loads(line))
    except Exception:
        return None
    return metrics

def load_summary():
    """Charge le résumé global"""
    if not os.path.exists(SUMMARY_FILE):
        return None

    try:
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def print_stats(stats, metrics_data=None, summary=None):
    """Affiche les statistiques de manière lisible"""
    if stats is None:
        print("⚠️  Fichier dataset non trouvé")
        return

    if "error" in stats:
        print(f"❌ Erreur: {stats['error']}")
        return

    print("=" * 70)
    print(f"📊 STATISTIQUES DU DATASET")
    print("=" * 70)
    print(f"\n📈 Total samples : {stats['total']}")
    print(f"🎯 Progression : {stats['total']}/15000 ({100*stats['total']/15000:.1f}%)")

    # Métriques de performance
    if metrics_data:
        successes = sum(1 for m in metrics_data if m.get("success", False))
        failures = sum(1 for m in metrics_data if not m.get("success", False))
        total_attempts = sum(m.get("attempts", 0) for m in metrics_data)

        if successes + failures > 0:
            success_rate = 100 * successes / (successes + failures)
            print(f"\n⚡ Métriques de Génération:")
            print(f"   Success Rate : {success_rate:.1f}% ({successes}/{successes + failures})")
            print(f"   Total Retries: {total_attempts}")
            print(f"   Avg Attempts : {total_attempts/(successes + failures):.2f}")

        # Tokens
        total_tokens = sum(m.get("total_tokens_total", 0) for m in metrics_data)
        total_prompt = sum(m.get("total_tokens_prompt", 0) for m in metrics_data)
        total_completion = sum(m.get("total_tokens_completion", 0) for m in metrics_data)

        if total_tokens > 0:
            cost = (total_prompt / 1_000_000) * 0.14 + (total_completion / 1_000_000) * 0.28
            print(f"\n🎫 Tokens Consommés:")
            print(f"   Total  : {total_tokens:,}")
            print(f"   Prompt : {total_prompt:,}")
            print(f"   Output : {total_completion:,}")
            print(f"   💰 Coût : ${cost:.2f}")

    # Difficulties
    print(f"\n🎓 Répartition par difficulté:")
    for diff, count in stats['difficulties'].most_common():
        pct = 100 * count / stats['total']
        bar = "█" * int(pct / 2)
        print(f"  {diff:12} : {count:4} ({pct:5.1f}%) {bar}")

    # Top 5 error categories
    print(f"\n🐛 Top 5 catégories d'erreurs:")
    for cat, count in stats['error_categories'].most_common(5):
        pct = 100 * count / stats['total']
        print(f"  {count:3}× ({pct:4.1f}%) {cat}")

    # Top 5 tags
    print(f"\n🏷️  Top 5 tags:")
    for tag, count in stats['tags'].most_common(5):
        print(f"  {count:3}× {tag}")

    print("=" * 70)

def monitor_loop():
    """Boucle de monitoring"""
    print("🚀 Monitoring démarré (Ctrl+C pour arrêter)")
    print(f"⏱️  Rafraîchissement toutes les {REFRESH_INTERVAL}s")
    print()

    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')

            # Charger les données
            stats = analyze_dataset()
            metrics = load_metrics()
            summary = load_summary()

            # Afficher
            print_stats(stats, metrics, summary)

            print(f"\n⏳ Prochaine mise à jour dans {REFRESH_INTERVAL}s...")
            time.sleep(REFRESH_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n👋 Monitoring arrêté")

if __name__ == "__main__":
    monitor_loop()
