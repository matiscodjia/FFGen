#!/usr/bin/env python3
"""
Extract all code from codes directory and merge with existing processed_collection.parquet
"""
import os
import pandas as pd
import uuid
import re
from tqdm import tqdm
from pathlib import Path

def extract_info_from_path(full_path, root_dir):
    """Extract author, cpoolday, and site from file path"""
    relative_path = os.path.relpath(full_path, root_dir)
    student_dir = os.path.dirname(relative_path).split(os.sep)[0]

    try:
        parts = student_dir.split('-')
        author_id_raw = student_dir.split('cpoolday')[1].split('-')[1]
        cpoolday = next((p for p in parts if p.startswith('cpoolday')), 'unknown')
        site = parts[3]
        return author_id_raw, cpoolday, site
    except (IndexError, StopIteration):
        print(f"Warning: Could not parse path: {student_dir}")
        return "ERROR", "ERROR", "ERROR"

def extract_codes_from_directory(codes_dir: str):
    """
    Extract all C code files from codes directory following stage 1 logic

    Args:
        codes_dir: Path to codes directory

    Returns:
        List of dictionaries containing code records
    """
    print(f"Extracting code from '{codes_dir}'...")

    data_records = []

    # Collect all .c files
    c_files_paths = []
    for dirpath, _, files in os.walk(codes_dir):
        for file in files:
            if file.endswith('.c'):
                c_files_paths.append(os.path.join(dirpath, file))

    if not c_files_paths:
        print(f"No .c files found in '{codes_dir}'")
        return []

    print(f"Found {len(c_files_paths)} .c files")

    # Process files
    for full_path in tqdm(c_files_paths, desc="Processing C files"):
        author_id_raw, cpoolday, site = extract_info_from_path(full_path, codes_dir)
        exercise_name = os.path.basename(full_path)

        if author_id_raw == "ERROR":
            continue

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
        except Exception as e:
            print(f"Warning: Error reading {full_path}: {e}")
            continue

        # Separate Header / Code (same logic as ingest_code.py)
        header_match = re.search(r'^\s*/\*(.*?)\*/(.*)$', full_content, re.DOTALL)

        if header_match:
            header = header_match.group(1).strip()
            code_snippet = header_match.group(2).strip()
        else:
            header = ""
            code_snippet = full_content.strip()

        if not code_snippet:
            continue

        # Create record with same structure as stage 1
        record = {
            'code_id': str(uuid.uuid4()),
            'author_id': str(uuid.uuid5(uuid.NAMESPACE_DNS, author_id_raw)),
            'code_snippet': code_snippet,
        }

        data_records.append(record)

    print(f"Extracted {len(data_records)} valid code records")
    return data_records

def merge_with_existing(new_records, existing_parquet_path):
    """
    Merge new code records with existing parquet file

    Args:
        new_records: List of new code records
        existing_parquet_path: Path to existing parquet file

    Returns:
        Merged DataFrame
    """
    # Create DataFrame from new records
    new_df = pd.DataFrame(new_records)
    print(f"\nNew records: {len(new_df)}")

    # Load existing data if it exists
    if os.path.exists(existing_parquet_path):
        existing_df = pd.read_parquet(existing_parquet_path)
        print(f"Existing records: {len(existing_df)}")

        # Remove duplicates based on code_snippet
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Check for duplicates
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['code_snippet'], keep='first')
        duplicates_removed = initial_count - len(combined_df)

        print(f"Total records after merge: {len(combined_df)}")
        print(f"Duplicates removed: {duplicates_removed}")

        return combined_df
    else:
        print(f"No existing parquet file found at {existing_parquet_path}")
        print(f"Creating new parquet with {len(new_df)} records")
        return new_df

def main():
    # Paths
    project_root = Path(__file__).parent
    codes_dir = project_root / "codes"
    output_parquet = project_root / "data" / "collections.parquet"
    existing_parquet = project_root / "data" / "processed_collection.parquet"

    print("="*70)
    print("Extract and Merge Codes - Stage 1 Data Extraction")
    print("="*70)

    # Extract codes from codes directory
    new_records = extract_codes_from_directory(str(codes_dir))

    if not new_records:
        print("\nNo records extracted. Exiting.")
        return

    # Merge with existing data
    merged_df = merge_with_existing(new_records, str(existing_parquet))

    # Save merged data
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    merged_df.to_parquet(output_parquet, index=False)

    print(f"\n✓ Successfully saved merged data to: {output_parquet}")
    print(f"  Total records: {len(merged_df)}")
    print(f"  Columns: {list(merged_df.columns)}")
    print("="*70)

if __name__ == "__main__":
    main()
