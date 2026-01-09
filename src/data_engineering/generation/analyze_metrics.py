#!/usr/bin/env python3
"""
Analyse avancée des métriques de génération
Usage: python scripts/analyze_metrics.py
"""

import json
import statistics
from collections import Counter, defaultdict

METRICS_FILE = "generation_metrics.jsonl"
SUMMARY_FILE = "generation_summary.json"

def load_metrics():
    """Charge toutes les métriques"""
    metrics = []
    try:
        with open(METRICS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                metrics.append(json.loads(line))
    except FileNotFoundError:
        print(f" Fichier {METRICS_FILE} introuvable")
        return []
    return metrics

def load_summary():
    """Charge le résumé global"""
    try:
        with open(SUMMARY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def analyze_tokens(metrics):
    """Analyse de consommation de tokens"""
    print("=" * 70)
    print(" ANALYSE DES TOKENS")
    print("=" * 70)

    if not metrics:
        print("Aucune donnée disponible")
        return

    # Statistiques par tentative
    all_prompts = []
    all_completions = []

    for m in metrics:
        all_prompts.append(m["total_tokens_prompt"])
        all_completions.append(m["total_tokens_completion"])

    print(f"\n Tokens Prompt (input):")
    print(f"   Total    : {sum(all_prompts):,}")
    print(f"   Moyenne  : {statistics.mean(all_prompts):.0f}")
    print(f"   Médiane  : {statistics.median(all_prompts):.0f}")
    print(f"   Min/Max  : {min(all_prompts)}/{max(all_prompts)}")

    print(f"\n Tokens Completion (output):")
    print(f"   Total    : {sum(all_completions):,}")
    print(f"   Moyenne  : {statistics.mean(all_completions):.0f}")
    print(f"   Médiane  : {statistics.median(all_completions):.0f}")
    print(f"   Min/Max  : {min(all_completions)}/{max(all_completions)}")

    # Tokens par succès vs échec
    success_tokens = [m["total_tokens_total"] for m in metrics if m["success"]]
    failure_tokens = [m["total_tokens_total"] for m in metrics if not m["success"]]

    if success_tokens:
        print(f"\n Tokens moyens (succès) : {statistics.mean(success_tokens):.0f}")
    if failure_tokens:
        print(f" Tokens moyens (échec)  : {statistics.mean(failure_tokens):.0f}")

    # Coût estimé
    total_tokens = sum(all_prompts) + sum(all_completions)
    cost_input = (sum(all_prompts) / 1_000_000) * 0.14  # DeepSeek pricing
    cost_output = (sum(all_completions) / 1_000_000) * 0.28
    total_cost = cost_input + cost_output

    print(f"\n COÛT ESTIMÉ (DeepSeek):")
    print(f"   Input  : ${cost_input:.2f}")
    print(f"   Output : ${cost_output:.2f}")
    print(f"   TOTAL  : ${total_cost:.2f}")

def analyze_retries(metrics):
    """Analyse des tentatives et retries"""
    print("\n" + "=" * 70)
    print(" ANALYSE DES RETRIES")
    print("=" * 70)

    attempts_distribution = Counter(m["attempts"] for m in metrics)

    print(f"\n Distribution des tentatives:")
    for num_attempts in sorted(attempts_distribution.keys()):
        count = attempts_distribution[num_attempts]
        pct = 100 * count / len(metrics)
        bar = "" * int(pct / 2)
        print(f"   {num_attempts} tentative(s) : {count:4} ({pct:5.1f}%) {bar}")

    # Taux de succès au premier coup
    first_try_success = sum(1 for m in metrics if m["success"] and m["attempts"] == 1)
    print(f"\n Succès au 1er coup : {first_try_success} ({100*first_try_success/len(metrics):.1f}%)")

    # Exercices problématiques (beaucoup de retries)
    exercise_retries = defaultdict(list)
    for m in metrics:
        exercise_retries[m["exercise"]].append(m["attempts"])

    print(f"\n  Top 5 exercices difficiles (moyenne de tentatives) :")
    exercise_avg_retries = {ex: statistics.mean(attempts) for ex, attempts in exercise_retries.items()}
    for ex, avg in sorted(exercise_avg_retries.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {avg:.2f} tentatives - {ex}")

def analyze_failures(metrics):
    """Analyse des échecs"""
    print("\n" + "=" * 70)
    print(" ANALYSE DES ÉCHECS")
    print("=" * 70)

    failures = [m for m in metrics if not m["success"]]

    if not failures:
        print(" Aucun échec !")
        return

    print(f"\n Total d'échecs : {len(failures)} ({100*len(failures)/len(metrics):.1f}%)")

    # Raisons d'échec
    failure_reasons = Counter(m.get("failure_reason", "unknown") for m in failures)
    print(f"\n Raisons d'échec :")
    for reason, count in failure_reasons.most_common():
        pct = 100 * count / len(failures)
        print(f"   {count:4}× ({pct:5.1f}%) {reason}")

    # Analyse détaillée des retries sur échecs
    print(f"\n Détail des échecs de validation (dernière tentative) :")
    validation_issues = []
    compilation_issues = []

    for m in failures:
        if m["retry_details"]:
            last_retry = m["retry_details"][-1]

            if last_retry.get("validation_result"):
                val_msg = last_retry["validation_result"].get("message", "")
                if not last_retry["validation_result"]["valid"]:
                    validation_issues.append(val_msg)

            if last_retry.get("compilation_result"):
                comp_msg = last_retry["compilation_result"].get("message", "")
                if not last_retry["compilation_result"]["compiled"]:
                    compilation_issues.append(comp_msg[:50])  # Tronquer

    if validation_issues:
        print(f"\n    Problèmes de validation sémantique :")
        for msg, count in Counter(validation_issues).most_common(5):
            print(f"      {count:3}× {msg}")

    if compilation_issues:
        print(f"\n    Problèmes de compilation :")
        for msg, count in Counter(compilation_issues).most_common(5):
            print(f"      {count:3}× {msg}...")

def analyze_timing(metrics):
    """Analyse des temps de génération"""
    print("\n" + "=" * 70)
    print("⏱  ANALYSE DES TEMPS")
    print("=" * 70)

    times = [m["generation_time_seconds"] for m in metrics]

    print(f"\n Temps de génération (secondes) :")
    print(f"   Total    : {sum(times):.0f}s ({sum(times)/3600:.1f}h)")
    print(f"   Moyenne  : {statistics.mean(times):.2f}s")
    print(f"   Médiane  : {statistics.median(times):.2f}s")
    print(f"   Min/Max  : {min(times):.2f}s / {max(times):.2f}s")

    # Temps par succès vs échec
    success_times = [m["generation_time_seconds"] for m in metrics if m["success"]]
    failure_times = [m["generation_time_seconds"] for m in metrics if not m["success"]]

    if success_times:
        print(f"\n Temps moyen (succès) : {statistics.mean(success_times):.2f}s")
    if failure_times:
        print(f" Temps moyen (échec)  : {statistics.mean(failure_times):.2f}s")

def analyze_error_types(metrics):
    """Analyse des types d'erreurs générés"""
    print("\n" + "=" * 70)
    print(" ANALYSE DES TYPES D'ERREURS")
    print("=" * 70)

    error_types = Counter(m["error_type"] for m in metrics)

    print(f"\n Distribution des catégories d'erreurs (top 10) :")
    for error, count in error_types.most_common(10):
        pct = 100 * count / len(metrics)
        print(f"   {count:3}× ({pct:4.1f}%) {error}")

    # Taux de succès par type d'erreur
    error_success_rate = defaultdict(lambda: {"success": 0, "total": 0})
    for m in metrics:
        error = m["error_type"]
        error_success_rate[error]["total"] += 1
        if m["success"]:
            error_success_rate[error]["success"] += 1

    print(f"\n Taux de succès par type d'erreur (top 5 plus difficiles) :")
    error_rates = {
        error: 100 * stats["success"] / stats["total"]
        for error, stats in error_success_rate.items()
        if stats["total"] >= 3  # Au moins 3 tentatives
    }

    for error, rate in sorted(error_rates.items(), key=lambda x: x[1])[:5]:
        total = error_success_rate[error]["total"]
        print(f"   {rate:5.1f}% ({error_success_rate[error]['success']}/{total}) {error}")

def print_summary(summary):
    """Affiche le résumé global"""
    if not summary:
        return

    print("\n" + "=" * 70)
    print(" RÉSUMÉ GLOBAL")
    print("=" * 70)

    print(f"\n Succès : {summary['total_successes']}")
    print(f" Échecs : {summary['total_failures']}")
    print(f" Success Rate : {summary.get('success_rate', 0)}%")
    print(f"⏱  Durée totale : {summary.get('total_duration_human', 'N/A')}")
    print(f" Coût total : ${summary.get('estimated_cost_usd', 0)}")

def main():
    print(" ANALYSE DÉTAILLÉE DES MÉTRIQUES DE GÉNÉRATION")
    print()

    metrics = load_metrics()

    if not metrics:
        print(" Aucune métrique à analyser")
        return

    print(f" {len(metrics)} entrées de métriques chargées")
    print()

    # Analyses
    summary = load_summary()
    if summary:
        print_summary(summary)

    analyze_tokens(metrics)
    analyze_retries(metrics)
    analyze_failures(metrics)
    analyze_timing(metrics)
    analyze_error_types(metrics)

    print("\n" + "=" * 70)
    print(" ANALYSE TERMINÉE")
    print("=" * 70)

if __name__ == "__main__":
    main()
