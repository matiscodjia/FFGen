#!/usr/bin/env python3
"""
extract_code.py - Extract C code snippets from source files

Simple script to extract code from C files and save them in JSONL format
with author_id, code_id, site, and code_snippet fields.
"""

import os
import json
import uuid
from pathlib import Path
from typing import List, Dict
import yaml


def extract_code_from_file(file_path: str) -> str:
    """Extract code content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def extract_metadata_from_path(file_path: Path, codes_dir: Path) -> Dict[str, str]:
    """
    Extract metadata from file path structure.

    Expected structure: codes_dir/site/day/author_id/file.c
    Example: codes/LYN/cpoolday02/john.doe/my_function.c

    Returns dict with: author_id, site, day
    """
    try:
        # Get relative path from codes_dir
        rel_path = file_path.relative_to(codes_dir)
        parts = rel_path.parts

        if len(parts) >= 3:
            site = parts[0]  # e.g., "LYN"
            day = parts[1]   # e.g., "cpoolday02"
            author_id = parts[2]  # e.g., "john.doe"

            return {
                "author_id": author_id,
                "site": site,
                "day": day
            }
    except:
        pass

    # Default values if extraction fails
    return {
        "author_id": "unknown",
        "site": "unknown",
        "day": "unknown"
    }


def extract_codes_from_directory(codes_dir: str, max_lines: int = 50) -> List[Dict]:
    """
    Extract code snippets from all .c files in directory.

    Args:
        codes_dir: Directory containing C source files
        max_lines: Maximum number of lines per snippet

    Returns:
        List of dictionaries with code snippets and metadata
    """
    snippets = []
    codes_path = Path(codes_dir)

    if not codes_path.exists():
        print(f"Error: Directory not found: {codes_dir}")
        return snippets

    # Find all .c files recursively
    c_files = list(codes_path.rglob("*.c"))

    print(f"Found {len(c_files)} C files")

    for file_path in c_files:
        code_content = extract_code_from_file(str(file_path))

        if not code_content or len(code_content.strip()) < 10:
            continue

        # Split into lines and limit length
        lines = code_content.split('\n')
        if len(lines) > max_lines:
            # Take first max_lines
            code_content = '\n'.join(lines[:max_lines])

        # Extract metadata from path
        metadata = extract_metadata_from_path(file_path, codes_path)

        snippet = {
            "code_id": str(uuid.uuid4()),
            "author_id": metadata["author_id"],
            "site": metadata["site"],
            "code_snippet": code_content,
        }

        snippets.append(snippet)

    print(f"Extracted {len(snippets)} code snippets")
    return snippets


def save_to_jsonl(snippets: List[Dict], output_file: str):
    """Save snippets to JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for snippet in snippets:
            f.write(json.dumps(snippet, ensure_ascii=False) + '\n')

    print(f"Saved {len(snippets)} snippets to {output_file}")


def main():
    """Main extraction pipeline."""
    # Load configuration
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    codes_dir = config['data']['codes_dir']
    output_file = config['data']['dataset_file']

    print("="*70)
    print("CODE EXTRACTION - FFGen")
    print("="*70)
    print(f"Source directory: {codes_dir}")
    print(f"Output file: {output_file}")
    print()

    # Extract codes
    snippets = extract_codes_from_directory(codes_dir)

    if not snippets:
        print("No code snippets extracted. Exiting.")
        return

    # Save to JSONL
    save_to_jsonl(snippets, output_file)

    print("\n✓ Extraction completed successfully")


if __name__ == "__main__":
    main()
