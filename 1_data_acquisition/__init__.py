"""
Stage 1: Data Acquisition and Mining
Extracts and validates code from source files
"""

from .ingest_code import run_data_acquisition
from .validate_code import compile_c_snippet, process_jsonl_file

__all__ = ['run_data_acquisition', 'compile_c_snippet', 'process_jsonl_file']
