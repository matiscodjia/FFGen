#!/usr/bin/env python3
import pandas as pd
import json

def parquet_to_jsonl(parquet_path: str, jsonl_path: str):
    """Convert parquet file to JSONL format"""
    df = pd.read_parquet(parquet_path)
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for record in df.to_dict('records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"Converted {len(df)} records: {parquet_path} -> {jsonl_path}")
