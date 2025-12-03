"""
Diagnostic script to analyze train/val/test split distributions
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (same as training script)
print("Loading data...")
hf_dataset = load_dataset("matis35/RAFT")
all_dfs = []
for split_name, split_data in hf_dataset.items():
    split_df = split_data.to_pandas()
    all_dfs.append(split_df)

df_feedback = pd.concat(all_dfs, ignore_index=True)
df_clustered = pd.read_csv("./data/dataset_clustered_no_tests.csv")
df_clustered = df_clustered.rename(columns={'code_snippet': 'code'})

df = df_feedback.merge(
    df_clustered[['code', 'cluster_kmeans']],
    on='code',
    how='inner'
)

df['cluster_kmeans'] = df['cluster_kmeans'].fillna(-1).astype(int)

# Same split as training
cluster_counts = df['cluster_kmeans'].value_counts()
min_cluster_size = cluster_counts.min()

if min_cluster_size >= 3:
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['cluster_kmeans'])
    train_df, val_df = train_test_split(train_df, test_size=0.111, random_state=42, stratify=train_df['cluster_kmeans'])
    print("✓ Using stratified splits")
else:
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.111, random_state=42)
    print("✓ Using random splits")

print(f"\nDataset sizes:")
print(f"  Train: {len(train_df):,}")
print(f"  Val:   {len(val_df):,}")
print(f"  Test:  {len(test_df):,}")

# Analyze cluster distributions
print("\n" + "="*80)
print("CLUSTER DISTRIBUTION ANALYSIS")
print("="*80)

train_clusters = set(train_df['cluster_kmeans'].unique())
val_clusters = set(val_df['cluster_kmeans'].unique())
test_clusters = set(test_df['cluster_kmeans'].unique())

print(f"\nUnique clusters:")
print(f"  Train: {len(train_clusters)}")
print(f"  Val:   {len(val_clusters)}")
print(f"  Test:  {len(test_clusters)}")

only_in_test = test_clusters - train_clusters
only_in_val = val_clusters - train_clusters

print(f"\n⚠️  Clusters ONLY in test (not in train): {len(only_in_test)}")
if only_in_test:
    print(f"     {only_in_test}")
    test_only_samples = test_df[test_df['cluster_kmeans'].isin(only_in_test)]
    print(f"     → {len(test_only_samples)} samples ({len(test_only_samples)/len(test_df)*100:.1f}% of test set)")

print(f"\n⚠️  Clusters ONLY in val (not in train): {len(only_in_val)}")
if only_in_val:
    print(f"     {only_in_val}")
    val_only_samples = val_df[val_df['cluster_kmeans'].isin(only_in_val)]
    print(f"     → {len(val_only_samples)} samples ({len(val_only_samples)/len(val_df)*100:.1f}% of val set)")

# Cluster size distribution per split
print("\n" + "="*80)
print("CLUSTER SIZE DISTRIBUTION")
print("="*80)

train_cluster_counts = train_df['cluster_kmeans'].value_counts().sort_index()
val_cluster_counts = val_df['cluster_kmeans'].value_counts().sort_index()
test_cluster_counts = test_df['cluster_kmeans'].value_counts().sort_index()

comparison_df = pd.DataFrame({
    'train': train_cluster_counts,
    'val': val_cluster_counts,
    'test': test_cluster_counts
}).fillna(0).astype(int)

print("\nCluster counts per split:")
print(comparison_df)

print("\n" + "="*80)
print("STATISTICS")
print("="*80)

for split_name, split_counts in [('Train', train_cluster_counts),
                                   ('Val', val_cluster_counts),
                                   ('Test', test_cluster_counts)]:
    print(f"\n{split_name}:")
    print(f"  Mean samples/cluster: {split_counts.mean():.1f}")
    print(f"  Median samples/cluster: {split_counts.median():.1f}")
    print(f"  Min samples/cluster: {split_counts.min()}")
    print(f"  Max samples/cluster: {split_counts.max()}")
    print(f"  Std: {split_counts.std():.1f}")

# Check for severely underrepresented clusters in test
print("\n" + "="*80)
print("POTENTIAL ISSUES")
print("="*80)

# Clusters with very different proportions
comparison_df['train_pct'] = (comparison_df['train'] / len(train_df) * 100)
comparison_df['test_pct'] = (comparison_df['test'] / len(test_df) * 100)
comparison_df['val_pct'] = (comparison_df['val'] / len(val_df) * 100)
comparison_df['test_vs_train_ratio'] = comparison_df['test_pct'] / (comparison_df['train_pct'] + 1e-6)

print("\nClusters over-represented in test (ratio > 2.0):")
overrep = comparison_df[comparison_df['test_vs_train_ratio'] > 2.0]
if len(overrep) > 0:
    print(overrep[['train', 'test', 'train_pct', 'test_pct', 'test_vs_train_ratio']])
else:
    print("  None")

print("\nClusters under-represented in test (ratio < 0.5):")
underrep = comparison_df[comparison_df['test_vs_train_ratio'] < 0.5]
if len(underrep) > 0:
    print(underrep[['train', 'test', 'train_pct', 'test_pct', 'test_vs_train_ratio']])
else:
    print("  None")

# Save detailed report
comparison_df.to_csv('./data/cluster_split_analysis.csv')
print(f"\n✓ Detailed analysis saved to: ./data/cluster_split_analysis.csv")

# Analyze code/feedback characteristics per split
print("\n" + "="*80)
print("CODE/FEEDBACK CHARACTERISTICS")
print("="*80)

for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    code_lengths = split_df['code'].str.len()
    feedback_lengths = split_df['feedback'].str.len()

    print(f"\n{split_name}:")
    print(f"  Code length - mean: {code_lengths.mean():.0f}, median: {code_lengths.median():.0f}")
    print(f"  Feedback length - mean: {feedback_lengths.mean():.0f}, median: {feedback_lengths.median():.0f}")

print("\n" + "="*80)
print("Done!")
print("="*80)
