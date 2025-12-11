"""
Export RAFT dataset to JSONL format for use in viewer app
"""

import pandas as pd
import json
from datasets import load_dataset
from pathlib import Path

print("=" * 80)
print("EXPORT RAFT DATASET TO JSONL")
print("=" * 80)

# Load RAFT dataset
print("\n1. Loading RAFT dataset from HuggingFace...")
hf_dataset = load_dataset("matis35/RAFT")

# Combine all splits
all_dfs = []
for split_name, split_data in hf_dataset.items():
    split_df = split_data.to_pandas()
    all_dfs.append(split_df)
    print(f"   - {split_name}: {len(split_df)} samples")

df = pd.concat(all_dfs, ignore_index=True)
print(f"\n   Total: {len(df)} samples")

# Check columns
print(f"\n2. Dataset columns: {df.columns.tolist()}")

# Expected columns: 'code', 'feedback'
if 'code' not in df.columns or 'feedback' not in df.columns:
    print("\n   ⚠️  Warning: Expected columns 'code' and 'feedback' not found")
    print(f"   Available columns: {df.columns.tolist()}")
    exit(1)

# Add IDs if not present
if 'id' not in df.columns:
    df['id'] = [f"raft_{i}" for i in range(len(df))]
    print("\n   ✓ Added 'id' column")

# Create output directory
output_dir = Path("./data/viewer_app")
output_dir.mkdir(parents=True, exist_ok=True)

# Export to JSONL
output_file = output_dir / "raft_dataset.jsonl"

print(f"\n3. Exporting to JSONL: {output_file}")

with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        item = {
            "id": row['id'],
            "code": row['code'],
            "feedback": row['feedback']
        }
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"   ✓ Exported {len(df)} samples")

# Sample output
print(f"\n4. Sample entries:")
for i in range(min(3, len(df))):
    print(f"\n   Sample {i+1}:")
    print(f"   ID: {df.iloc[i]['id']}")
    print(f"   Code (first 100 chars): {df.iloc[i]['code'][:100]}...")
    print(f"   Feedback (first 100 chars): {df.iloc[i]['feedback'][:100]}...")

print("\n" + "=" * 80)
print("✓ SUCCESS!")
print("=" * 80)
print(f"\nDataset exported to: {output_file}")
print(f"\nTo use in viewer app:")
print(f"  1. cd viewer_app")
print(f"  2. streamlit run app.py")
print(f"  3. Go to 'RAG Testing' tab")
print(f"  4. Upload the JSONL file: {output_file.absolute()}")
print("\n" + "=" * 80)
