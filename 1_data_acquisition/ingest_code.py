import os
import pandas as pd
import uuid
import re
import logging
from tqdm import tqdm
from typing import Dict, Any, Optional

# C parser will be loaded on-demand if AST computation is enabled
language = None
parser = None

def init_c_parser():
    """Initialize C parser on-demand"""
    global language, parser
    if parser is None:
        try:
            from tree_sitter_languages import get_language, get_parser
            language = get_language('c')
            parser = get_parser('c')
            print("Parseur C initialisé avec succès.")
        except Exception as e:
            print(f"Erreur lors de l'initialisation du parseur C : {e}")
            raise

def linearize_ast_structurel(node, source_code_bytes) -> str:
    node_type = str(node.type)

    if node_type == "primitive_type": return f"type:{node.text.decode('utf8')}"
    if node_type == "string_literal": return f"str:{node.text.decode('utf8')}"
    if node_type == "identifier": return f"id:{node.text.decode('utf8')}"
    if node_type == "number_literal": return f"num:{node.text.decode('utf8')}"

    if not node.is_named or node_type == "comment":
        return ""

    children_strings = [linearize_ast_structurel(child, source_code_bytes) for child in node.children]
    children_strings = [s for s in children_strings if s]
    
    if not children_strings:
        return ""
        
    linearized_children = " ".join(children_strings)

    if node_type in ['translation_unit', 'expression_statement', 'compound_statement', 'parameter_list', 'argument_list']:
        return linearized_children
           
    return f"({node_type} {linearized_children})"

def get_ast_from_code(code_string: str) -> str:
    """Generate AST representation from C code string"""
    if parser is None:
        return "(error parser_not_initialized)"
    if not isinstance(code_string, str):
        return "(error type:input_not_string)"
    try:
        code_bytes = bytes(code_string, "utf8")
        tree = parser.parse(code_bytes)
        ast_string = linearize_ast_structurel(tree.root_node, code_bytes)
        return " ".join(ast_string.split())
    except Exception as e:
        return f"(error parsing_failed:{str(e)})"


def extract_info_from_path(full_path, root_dir):
    relative_path = os.path.relpath(full_path, root_dir)
    student_dir = os.path.dirname(relative_path).split(os.sep)[0]
    
    try:
        parts = student_dir.split('-')
        author_id_raw = student_dir.split('cpoolday')[1].split('-')[1]
        cpoolday = next((p for p in parts if p.startswith('cpoolday')), 'unknown')
        site = parts[3] 
        return author_id_raw, cpoolday, site
    except (IndexError, StopIteration):
        logging.warning(f"Impossible d'analyser le chemin : {student_dir}")
        return "ERROR", "ERROR", "ERROR"

# --- Main Function for Data Acquisition Stage ---
def run_data_acquisition(config: Dict[str, Any]):
    """
    Stage 1: Data Mining and Acquisition

    Extracts C code files from source directory. Optionally generates AST representations
    if compute_ast is enabled in config.

    Args:
        config: Configuration dictionary containing paths and settings

    Returns:
        Path to the generated parquet file with extracted code data
    """
    output_parquet = config['paths']['processed_collection']
    root_dir = config['paths']['source_code_dir']
    compute_ast = config['paths'].get('compute_ast', False)

    # Initialize C parser only if AST computation is enabled
    if compute_ast:
        print("[Stage 1] AST computation enabled. Initializing C parser...")
        init_c_parser()
    else:
        print("[Stage 1] AST computation disabled.")

    if os.path.exists(output_parquet):
        print(f"[Stage 1] Data already acquired: '{output_parquet}' exists. Skipping.")
        return output_parquet

    print(f"[Stage 1] Starting data acquisition from '{root_dir}'...")
    
    data_records = []
    
    # 1. Collecter tous les fichiers .c
    c_files_paths = []
    for dirpath, _, files in os.walk(top=root_dir):
        for file in files:
            if file.endswith('.c'):
                c_files_paths.append(os.path.join(dirpath, file))
                
    if not c_files_paths:
        print(f"Aucun fichier .c trouvé dans '{root_dir}'. Arrêt.")
        raise FileNotFoundError(f"Pas de codes sources dans {root_dir}")

    # 2. Process files
    print(f"Processing {len(c_files_paths)} .c files...")
    for full_path in tqdm(c_files_paths, desc="[Stage 1] Mining code files"):
        author_id_raw, cpoolday, site = extract_info_from_path(full_path, root_dir)
        exercise_name = os.path.basename(full_path)
        
        if author_id_raw == "ERROR":
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
        except Exception:
            logging.warning(f"Erreur de lecture (encodage?) sur {full_path}")
            continue
            
        # 3. Séparer Header / Code (logique de ffgen.ipynb)
        header_match = re.search(r'^\s*/\*(.*?)\*/(.*)$', full_content, re.DOTALL)
        
        if header_match:
            header = header_match.group(1).strip()
            code_snippet = header_match.group(2).strip()
        else:
            header = ""
            code_snippet = full_content.strip()
            
        if not code_snippet:
            continue

        # 4. Optionally generate AST if enabled in config
        record = {
            'code_id': str(uuid.uuid4()),
            'author_id': str(uuid.uuid5(uuid.NAMESPACE_DNS, author_id_raw)),
            'code_snippet': code_snippet,
        }

        if compute_ast:
            ast_representation = get_ast_from_code(code_snippet)
            record['code_ast_structural'] = ast_representation

        data_records.append(record)
    if not data_records:
        print("[Stage 1] ERROR: No records created. Check your source files.")
        return None

    print(f"[Stage 1] Collected {len(data_records)} records. Saving to {output_parquet}...")
    df = pd.DataFrame(data_records)
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"[Stage 1] ✓ Data acquisition completed. Saved to: {output_parquet}")
    return output_parquet