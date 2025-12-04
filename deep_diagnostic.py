"""
Deep diagnostic script to understand the train/val/test gap
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("="*80)
print("DEEP DIAGNOSTIC: Train/Val/Test Distribution Analysis")
print("="*80)

# Load data (same as training script)
print("\n1. Loading data...")
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
else:
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.111, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# =============================================================================
# DIAGNOSTIC 1: Cluster overlap
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 1: Cluster Overlap")
print("="*80)

train_clusters = set(train_df['cluster_kmeans'].unique())
val_clusters = set(val_df['cluster_kmeans'].unique())
test_clusters = set(test_df['cluster_kmeans'].unique())

only_in_test = test_clusters - train_clusters
only_in_val = val_clusters - train_clusters
overlap_all = train_clusters & val_clusters & test_clusters

print(f"\nTotal unique clusters: {len(set(df['cluster_kmeans']))}")
print(f"  Train: {len(train_clusters)}")
print(f"  Val:   {len(val_clusters)}")
print(f"  Test:  {len(test_clusters)}")
print(f"  Overlap (all 3): {len(overlap_all)}")

print(f"\n🔴 Clusters ONLY in test: {len(only_in_test)}")
if only_in_test:
    print(f"   Clusters: {sorted(only_in_test)}")
    test_unseen = test_df[test_df['cluster_kmeans'].isin(only_in_test)]
    print(f"   Affected samples: {len(test_unseen)} ({len(test_unseen)/len(test_df)*100:.1f}% of test)")

print(f"\n🟠 Clusters ONLY in val: {len(only_in_val)}")
if only_in_val:
    print(f"   Clusters: {sorted(only_in_val)}")
    val_unseen = val_df[val_df['cluster_kmeans'].isin(only_in_val)]
    print(f"   Affected samples: {len(val_unseen)} ({len(val_unseen)/len(val_df)*100:.1f}% of val)")

# =============================================================================
# DIAGNOSTIC 2: Cluster representation per split
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 2: Cluster Size Distribution")
print("="*80)

train_cluster_counts = train_df['cluster_kmeans'].value_counts()
val_cluster_counts = val_df['cluster_kmeans'].value_counts()
test_cluster_counts = test_df['cluster_kmeans'].value_counts()

comparison_df = pd.DataFrame({
    'train': train_cluster_counts,
    'val': val_cluster_counts,
    'test': test_cluster_counts
}).fillna(0).astype(int)

comparison_df['train_pct'] = comparison_df['train'] / len(train_df) * 100
comparison_df['val_pct'] = comparison_df['val'] / len(val_df) * 100
comparison_df['test_pct'] = comparison_df['test'] / len(test_df) * 100

print("\nTop 10 clusters by total size:")
comparison_df['total'] = comparison_df['train'] + comparison_df['val'] + comparison_df['test']
print(comparison_df.nlargest(10, 'total')[['train', 'val', 'test', 'train_pct', 'val_pct', 'test_pct']])

print("\nClusters with severe imbalance (test >> train):")
comparison_df['test_train_ratio'] = comparison_df['test_pct'] / (comparison_df['train_pct'] + 1e-6)
imbalanced = comparison_df[comparison_df['test_train_ratio'] > 3.0].sort_values('test_train_ratio', ascending=False)
if len(imbalanced) > 0:
    print(imbalanced[['train', 'test', 'train_pct', 'test_pct', 'test_train_ratio']])
else:
    print("None found")

# =============================================================================
# DIAGNOSTIC 3: Code/Feedback length distribution
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 3: Text Length Distribution")
print("="*80)

for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    code_lengths = split_df['code'].str.len()
    feedback_lengths = split_df['feedback'].str.len()

    print(f"\n{split_name}:")
    print(f"  Code length:     mean={code_lengths.mean():.0f}, median={code_lengths.median():.0f}, std={code_lengths.std():.0f}")
    print(f"  Feedback length: mean={feedback_lengths.mean():.0f}, median={feedback_lengths.median():.0f}, std={feedback_lengths.std():.0f}")
    print(f"  Code > 512 chars: {(code_lengths > 512).sum()} ({(code_lengths > 512).mean()*100:.1f}%)")
    print(f"  Feedback > 256 chars: {(feedback_lengths > 256).sum()} ({(feedback_lengths > 256).mean()*100:.1f}%)")

# =============================================================================
# DIAGNOSTIC 4: Cluster quality metrics
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 4: Cluster Characteristics")
print("="*80)

# For each cluster, compute average code/feedback length
cluster_stats = []
for cluster_id in sorted(df['cluster_kmeans'].unique()):
    cluster_data = df[df['cluster_kmeans'] == cluster_id]

    train_count = len(train_df[train_df['cluster_kmeans'] == cluster_id])
    val_count = len(val_df[val_df['cluster_kmeans'] == cluster_id])
    test_count = len(test_df[test_df['cluster_kmeans'] == cluster_id])

    cluster_stats.append({
        'cluster': cluster_id,
        'total_size': len(cluster_data),
        'train': train_count,
        'val': val_count,
        'test': test_count,
        'avg_code_len': cluster_data['code'].str.len().mean(),
        'avg_feedback_len': cluster_data['feedback'].str.len().mean()
    })

cluster_stats_df = pd.DataFrame(cluster_stats)

print("\nClusters with most test samples:")
print(cluster_stats_df.nlargest(10, 'test')[['cluster', 'train', 'val', 'test', 'avg_code_len', 'avg_feedback_len']])

print("\nClusters with imbalanced splits:")
cluster_stats_df['test_ratio'] = cluster_stats_df['test'] / (cluster_stats_df['train'] + 1)
imbalanced_clusters = cluster_stats_df[cluster_stats_df['test_ratio'] > 0.5].sort_values('test_ratio', ascending=False)
if len(imbalanced_clusters) > 0:
    print(imbalanced_clusters.head(10)[['cluster', 'train', 'val', 'test', 'test_ratio']])
else:
    print("None found")

# =============================================================================
# DIAGNOSTIC 5: Sample some test examples
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 5: Sample Test Examples")
print("="*80)

print("\nRandom sample from test set:")
sample_test = test_df.sample(3, random_state=42)
for idx, row in sample_test.iterrows():
    print(f"\nCluster {row['cluster_kmeans']}:")
    print(f"  Code (first 100 chars): {row['code'][:100]}...")
    print(f"  Feedback (first 100 chars): {row['feedback'][:100]}...")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print(f"\n1. Cluster coverage:")
print(f"   - {len(only_in_test)} clusters appear ONLY in test")
print(f"   - This affects {len(test_df[test_df['cluster_kmeans'].isin(only_in_test)])} samples")

print(f"\n2. Distribution imbalance:")
imbalanced_count = len(cluster_stats_df[cluster_stats_df['test_ratio'] > 0.5])
print(f"   - {imbalanced_count} clusters have test_ratio > 0.5 (more test than train)")

print(f"\n3. Length distribution:")
test_code_mean = test_df['code'].str.len().mean()
train_code_mean = train_df['code'].str.len().mean()
diff_pct = abs(test_code_mean - train_code_mean) / train_code_mean * 100
print(f"   - Code length diff (test vs train): {diff_pct:.1f}%")

if len(only_in_test) > 0:
    print(f"\n⚠️  ROOT CAUSE: Clusters in test that were never seen in training!")
    print(f"   The model cannot generalize to these unseen clusters.")
    print(f"   → Solution: Re-split the data to ensure all clusters appear in train")

# Save detailed report
cluster_stats_df.to_csv('./data/cluster_analysis_detailed.csv', index=False)
comparison_df.to_csv('./data/split_comparison.csv')
print(f"\n✓ Detailed reports saved to:")
print(f"   - ./data/cluster_analysis_detailed.csv")
print(f"   - ./data/split_comparison.csv")
