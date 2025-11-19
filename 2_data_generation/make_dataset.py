import pandas as pd

dataset = pd.read_json('/Users/matiscodjia/Dev/06_research/FFGen/data/Exp-002-llama3B_v2_multi_neg_10.jsonl', lines=True)


dataset_exploded = dataset.explode('negative_feedbacks')

# 2. Renommer la nouvelle colonne si nécessaire pour un traitement uniforme
# Si vous voulez l'utiliser comme un simple 'negative_feedback' (singulier)
dataset_exploded = dataset_exploded.rename(columns={'negative_feedbacks': 'negative_feedback'})

# Afficher les premières lignes du nouveau DataFrame
print(dataset_exploded.head())

dic = dataset_exploded.to_dict(orient='records')

print(len(dic))

