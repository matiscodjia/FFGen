"""
Data Preprocessing Utilities

Provides utilities for cleaning, validating, and transforming data
before feeding it into the feedback generation pipeline.
"""

import pandas as pd
import re
from typing import List, Dict, Any
import logging


def clean_code_snippet(code: str) -> str:
    """
    Clean and normalize code snippets

    Args:
        code: Raw code string

    Returns:
        Cleaned code string
    """
    if not isinstance(code, str):
        return ""

    # Remove excessive whitespace
    code = re.sub(r'\s+', ' ', code)

    # Strip leading/trailing whitespace
    code = code.strip()

    return code


def validate_record(record: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate that a record contains all required fields

    Args:
        record: Data record to validate
        required_fields: List of required field names

    Returns:
        True if valid, False otherwise
    """
    for field in required_fields:
        if field not in record or not record[field]:
            logging.warning(f"Record missing required field: {field}")
            return False

    return True


def filter_by_length(
    df: pd.DataFrame,
    column: str,
    min_length: int = 10,
    max_length: int = 10000
) -> pd.DataFrame:
    """
    Filter dataframe rows by text length in specified column

    Args:
        df: Input dataframe
        column: Column name to check
        min_length: Minimum character length (inclusive)
        max_length: Maximum character length (inclusive)

    Returns:
        Filtered dataframe
    """
    initial_count = len(df)

    df = df[df[column].str.len() >= min_length]
    df = df[df[column].str.len() <= max_length]

    filtered_count = initial_count - len(df)

    if filtered_count > 0:
        logging.info(f"Filtered out {filtered_count} records based on length constraints")

    return df


def deduplicate_dataframe(
    df: pd.DataFrame,
    column: str,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Remove duplicate rows based on a column

    Args:
        df: Input dataframe
        column: Column to check for duplicates
        keep: Which duplicate to keep ('first', 'last', or False to drop all)

    Returns:
        Deduplicated dataframe
    """
    initial_count = len(df)

    df = df.drop_duplicates(subset=[column], keep=keep)

    removed_count = initial_count - len(df)

    if removed_count > 0:
        logging.info(f"Removed {removed_count} duplicate records")

    return df


def preprocess_dataset(
    input_path: str,
    output_path: str,
    min_code_length: int = 10,
    max_code_length: int = 5000
) -> str:
    """
    Preprocess a dataset with cleaning and validation

    Args:
        input_path: Path to input parquet file
        output_path: Path to save processed parquet file
        min_code_length: Minimum code snippet length
        max_code_length: Maximum code snippet length

    Returns:
        Path to processed dataset
    """
    logging.info(f"Loading dataset from {input_path}")
    df = pd.read_parquet(input_path)

    initial_count = len(df)
    logging.info(f"Initial dataset size: {initial_count} records")

    if 'code_snippet' in df.columns:
        df['code_snippet'] = df['code_snippet'].apply(clean_code_snippet)

    df = filter_by_length(df, 'code_snippet', min_code_length, max_code_length)

    df = deduplicate_dataframe(df, 'code_snippet')

    # Remove empty records
    df = df[df['code_snippet'].str.len() > 0]

    final_count = len(df)
    logging.info(f"Final dataset size: {final_count} records ({final_count/initial_count*100:.1f}% retained)")

    # Save processed dataset
    df.to_parquet(output_path, index=False)
    logging.info(f"Processed dataset saved to {output_path}")

    return output_path
