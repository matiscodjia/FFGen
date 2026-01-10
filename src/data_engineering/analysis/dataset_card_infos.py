import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION DES PRIX (Basé sur ta capture) ---
# Prix pour 1 Million de tokens
PRICE_INPUT_MISS = 0.28   # $0.28 / 1M (Prompt - Standard)
PRICE_OUTPUT = 0.42       # $0.42 / 1M (Completion)

# Nom du fichier de logs
LOG_FILE = "generation_metrics.jsonl"
OUTPUT_DIR = "dataset_card_assets"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def load_data(filepath):
    """Charge le JSONL dans un DataFrame Pandas"""
    if not Path(filepath).exists():
        print(f"❌ Erreur : Le fichier '{filepath}' est introuvable.")
        exit(1)
        
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def analyze_economics(df):
    print_header("💰 ECONOMICS (Precise Cost Calculation)")
    
    total_samples = len(df)
    
    # On sépare les tokens d'entrée et de sortie car ils n'ont pas le même prix
    # Note : On utilise 'total_tokens_prompt' et 'total_tokens_completion' qui sont dans tes logs
    if 'total_tokens_prompt' in df.columns and 'total_tokens_completion' in df.columns:
        total_input = df['total_tokens_prompt'].sum()
        total_output = df['total_tokens_completion'].sum()
    else:
        # Fallback si les colonnes sont mal nommées (basé sur total / 2 pour estimer)
        total_input = df['total_tokens_total'].sum() * 0.45 
        total_output = df['total_tokens_total'].sum() * 0.55

    total_tokens = total_input + total_output

    # Calcul du coût précis
    cost_input = (total_input / 1_000_000) * PRICE_INPUT_MISS
    cost_output = (total_output / 1_000_000) * PRICE_OUTPUT
    total_cost = cost_input + cost_output
    
    print(f"- Total Samples:             {total_samples:,}")
    print(f"- Total Input Tokens:        {total_input:,.0f}")
    print(f"- Total Output Tokens:       {total_output:,.0f}")
    print(f"- Total Combined Tokens:     {total_tokens:,.0f}")
    print("-" * 30)
    print(f"- Cost (Inputs @ ${PRICE_INPUT_MISS}):   ${cost_input:.2f}")
    print(f"- Cost (Outputs @ ${PRICE_OUTPUT}):   ${cost_output:.2f}")
    print(f"- TOTAL REPRODUCTION COST:   ${total_cost:.2f}")
    
    # Markdown snippet pour le README
    print("\n📋 [Copy to Markdown]:")
    print(f"| Metric | Value |")
    print(f"| :--- | :--- |")
    print(f"| **Samples** | {total_samples:,} |")
    print(f"| **Input Tokens** | {total_input/1e6:.2f}M |")
    print(f"| **Output Tokens** | {total_output/1e6:.2f}M |")
    print(f"| **Est. Value** | **${total_cost:.2f}** |")

def analyze_green_ai(df):
    print_header("🌱 GREEN AI & COMPUTE TIME")
    
    total_seconds = df['generation_time_seconds'].sum()
    total_hours = total_seconds / 3600
    avg_time = df['generation_time_seconds'].mean()
    
    print(f"- Total Compute Time: {total_hours:.2f} hours")
    print(f"- Average Time per Sample: {avg_time:.2f} seconds")
    
    # Estimation CO2 (Hypothèse GPU moderne ~300W-400W en charge)
    # 0.3 kgCO2e par kWh (Moyenne mix énergétique global)
    kwh = (400 * total_hours) / 1000 
    co2 = kwh * 0.3
    
    print(f"- Est. Energy Consumption: {kwh:.2f} kWh")
    print(f"- Est. Carbon Footprint: {co2:.2f} kgCO2e")

def analyze_quality(df):
    print_header("🛡️ QUALITY & ROBUSTNESS")
    
    if 'attempts' not in df.columns:
        print("⚠️ Pas de colonne 'attempts' trouvée.")
        return

    total_attempts = df['attempts'].sum()
    retries = total_attempts - len(df)
    retry_rate = (retries / total_attempts) * 100
    
    first_try = df[df['attempts'] == 1]
    success_rate_first_pass = (len(first_try) / len(df)) * 100
    
    print(f"- Total Generations Triggered: {total_attempts}")
    print(f"- Discarded/Retried Samples:   {retries}")
    print(f"- Pipeline Rejection Rate:     {retry_rate:.2f}%")
    print(f"- First-Pass Success Rate:     {success_rate_first_pass:.2f}%")

def plot_complexity(df):
    print_header("🧠 COMPLEXITY ANALYSIS")
    
    # On utilise les tokens de sortie comme proxy de complexité du code généré
    if 'total_tokens_completion' in df.columns:
        col_target = 'total_tokens_completion'
        x_label = "Average Output Tokens (Code + Explanation)"
    else:
        col_target = 'total_tokens_total'
        x_label = "Average Total Tokens"

    complexity = df.groupby('error_type')[col_target].mean().sort_values(ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=complexity.values, y=complexity.index, hue=complexity.index, palette="viridis", legend=False)
    plt.title("Top 15 Most Verbose Error Corrections (Token Count)", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Error Category", fontsize=12)
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/complexity_chart.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Graphique sauvegardé : {filename}")

def plot_diversity_heatmap(df):
    print_header("🎨 DIVERSITY HEATMAP")
    
    top_exercises = df['exercise'].value_counts().head(20).index
    top_errors = df['error_type'].value_counts().head(20).index
    
    filtered_df = df[df['exercise'].isin(top_exercises) & df['error_type'].isin(top_errors)]
    
    matrix = pd.crosstab(filtered_df['error_type'], filtered_df['exercise'])
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(matrix, cmap="YlGnBu", annot=True, fmt='d', cbar_kws={'label': 'Count'})
    plt.title("Dataset Diversity: Error Types vs Exercises", fontsize=16)
    plt.xlabel("Exercise Theme", fontsize=12)
    plt.ylabel("Error Category", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/diversity_heatmap.png"
    plt.savefig(filename, dpi=300)
    print(f"✅ Heatmap sauvegardée : {filename}")

def main():
    print(f"Lecture du fichier : {LOG_FILE} ...")
    df = load_data(LOG_FILE)
    
    # Configuration du style graphiques
    sns.set_theme(style="whitegrid")

    analyze_economics(df)
    analyze_green_ai(df)
    analyze_quality(df)
    plot_complexity(df)
    plot_diversity_heatmap(df)
    
    print("\n" + "="*60)
    print(f"🎉 ANALYSE TERMINÉE. Résultats dans '{OUTPUT_DIR}'")
    print("="*60)

if __name__ == "__main__":
    main()