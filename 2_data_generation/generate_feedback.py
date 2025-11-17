import pandas as pd
import logging
import os
import math
import sys
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI
from typing import Dict, Any, List
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_progress,
    save_result_batch
)

try:
    from utils.inference_service import InferenceServer
    INFERENCE_SERVICE_AVAILABLE = True
except ImportError:
    INFERENCE_SERVICE_AVAILABLE = False
    print("[Generation] Warning: inference_service not available, using AsyncOpenAI only")

# Global client and config (initialized in run_feedback_generation)
async_client = None
inference_server = None
current_config = None
use_inference_server = False

def initialize_llm_client(config_path: str = "./configs/config.yml"):
    """
    Initialize LLM client with fallback support from config file.

    Priority:
    1. Try InferenceServer from config (OpenAI-compatible server)
    2. Fall back to AsyncOpenAI direct connection

    Args:
        config_path: Path to config YAML file
    """
    global async_client, inference_server, use_inference_server

    if INFERENCE_SERVICE_AVAILABLE:
        try:
            # Try InferenceServer from config (has built-in fallback)
            inference_server = InferenceServer.from_config(
                config_path=config_path,
                service_type="llm"
            )
            use_inference_server = True
            print(f"[Generation] ✓ Using InferenceServer from config")
            return
        except Exception as e:
            print(f"[Generation] InferenceServer unavailable: {e}")

    # Fallback to AsyncOpenAI with config values
    if async_client is None:
        # Try to read URL from config, fallback to default
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            server_url = config.get('inference', {}).get('llm', {}).get('url', 'http://localhost:8001/v1')
        except:
            server_url = 'http://localhost:8001/v1'

        async_client = AsyncOpenAI(
            base_url=server_url,
            api_key="none"
        )
        use_inference_server = False
        print(f"[Generation] Using AsyncOpenAI at {server_url}")

async def call_llm(messages):
    """
    Call LLM with given messages using available backend.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        Generated text response
    """
    if use_inference_server and inference_server:
        # Use InferenceServer
        return await inference_server.chat(messages)
    else:
        # Use AsyncOpenAI
        resp = await async_client.chat.completions.create(
            model=current_config['generation']['llm_model'],
            messages=messages
        )
        return resp.choices[0].message.content.strip()

def run_agent_chain_on_batch(
    batch_items: List[Dict[str, Any]], 
    agents_config: List[Dict[str, Any]], 
    agent_prompts: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Exécute la chaîne d'agents complète sur un SEUL batch d'items.
    """
    current_batch_data = batch_items
    
    for agent_conf in agents_config:
        agent_name = agent_conf['agent']
        prompt_template = agent_prompts[agent_name]
        input_col = agent_conf['input_col']
        output_col = agent_conf['output_col']
        
        prompts_to_process = []
        
        for item in current_batch_data:
            if agent_name == "adversary":
                code_col = "code_snippet" # On suppose que l'ancre est toujours le code snippet
                pos_col = input_col # Le 'input_col' de l'adversaire est le 'positive'
                
                user_content = prompt_template.format(
                    code=item.get(code_col, ""), 
                    positive=item.get(pos_col, "")
                )
                chat_prompt = [{"role": "user", "content": user_content}]
            else:
                # Cas standard (tutor, editor)
                input_text = item.get(input_col, "")
                chat_prompt = [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": input_text}
                ]
            prompts_to_process.append(chat_prompt)


        async def run_batch():
            tasks = [call_llm(p) for p in prompts_to_process]
            return await asyncio.gather(*tasks)
        try:
            outputs = asyncio.run(run_batch())
   
        except Exception as e:
            print(f"Échec du pipeline de génération (batch) pour l'agent {agent_name}")
            print(e)

            outputs = [{"generated_text": "ERROR: LLM_BATCH_FAILED"}] * len(prompts_to_process)

        updated_batch = []
        for item, generated in zip(current_batch_data, outputs):
            item[output_col] = generated
            updated_batch.append(item)
        
        current_batch_data = updated_batch
    return current_batch_data

def run_feedback_generation(config: Dict[str, Any]) -> str:
    """
    Stage 2: Data Processing - LLM-based Feedback Generation

    Generates synthetic feedback using multiple LLM agents (tutor, editor, adversary, conceptual).
    Processes data in batches with resumable progress.

    Args:
        config: Configuration dictionary containing generation settings

    Returns:
        Path to the generated JSONL dataset file
    """
    # Initialize global config and LLM client
    global current_config
    current_config = config
    initialize_llm_client()

    gen_config = config['generation']
    source_parquet = config['paths']['processed_collection']
    final_dataset_path = gen_config['final_dataset_file']
    batch_size = gen_config['batch_size']

    print(f"[Stage 2] Starting feedback generation (Batch Size = {batch_size})...")
    # 1. Load source data
    if not os.path.exists(source_parquet):
        print(f"[Stage 2] ERROR: Source file not found: {source_parquet}. Run Stage 1 first.")
        raise FileNotFoundError(source_parquet)
    
    df_source = pd.read_parquet(source_parquet)
    all_source_items = df_source.to_dict('records')
    total_items = len(all_source_items)
    
    if not all_source_items:
        print("Le fichier source est vide.")
        raise ValueError("Données sources vides.")

    progress = load_progress(final_dataset_path, id_column='code_id')
    
    items_to_process = [
        item for item in all_source_items 
        if item['code_id'] not in progress
    ]
    num_to_process = len(items_to_process)
    
    if num_to_process == 0:
        print("[Stage 2] All items already processed. Skipping generation.")
        return final_dataset_path

    print(f"[Stage 2] Found {len(progress)} processed items. {num_to_process} remaining to generate.")

        
    agent_prompts = {}
    for agent_conf in gen_config['agents']:
        prompt_path = agent_conf['prompt_file']
        agent_name = agent_conf['agent']
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                agent_prompts[agent_name] = f.read()
            print(f"Prompt chargé pour l'agent : {agent_name}")
        except FileNotFoundError:
            print(f"Prompt introuvable : {prompt_path}")
            raise

    # 5. Boucle principale par BATCHS
    effective_agents_config = []
    print("Construction de la chaîne d'agents effective :")
    for agent_conf in gen_config['agents']:
        agent_name = agent_conf['agent']
        print(agent_name)
        num_runs = agent_conf.get('runs', 1) 
        
        if num_runs > 1:
            print(f" -> Agent '{agent_name}' sera exécuté {num_runs} fois (raffinement itératif).")
        else:
            print(f" -> Agent '{agent_name}' sera exécuté 1 fois.")

        original_input_col = agent_conf['input_col']
        output_col = agent_conf['output_col']

        for i in range(num_runs):
            run_conf = agent_conf.copy()
            
            if i == 0:
                run_conf['input_col'] = original_input_col
            else:
                run_conf['input_col'] = output_col 
            
            run_conf['output_col'] = output_col
            
            effective_agents_config.append(run_conf)
    num_batches = math.ceil(num_to_process / batch_size)
    print("[Stage 2] Generating feedback batches...")
    for i in tqdm(range(0, num_to_process, batch_size), total=num_batches, desc="[Stage 2] Generating Feedbacks"):
        batch_items = items_to_process[i : i + batch_size]
        
        try:
            processed_batch = run_agent_chain_on_batch(
                batch_items,
                gen_config['agents'],
                agent_prompts
            )
            
            save_result_batch(final_dataset_path, processed_batch)
            
        except Exception:
            print(f"Échec critique sur un batch (index {i}). Batch ignoré.")
            continue
    
    print(f"[Stage 2] ✓ Feedback generation completed. Dataset: {final_dataset_path}")
    return final_dataset_path
