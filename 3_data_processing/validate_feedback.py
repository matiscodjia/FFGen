"""
Feedback Validation using Judge LLM

Uses a larger/better model to evaluate and potentially correct feedback quality.
Produces quantifiable scores and actionable improvements.
"""

import pandas as pd
import logging
import os
import json
import math
import sys
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI
from typing import Dict, Any, List
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_progress, save_result_batch

# Global client and config
async_client = None
current_config = None

def initialize_judge_client(base_url: str = "http://localhost:1234/v1"):
    """Initialize the OpenAI async client for judge model"""
    global async_client
    if async_client is None:
        async_client = AsyncOpenAI(
            base_url=base_url,
            api_key="none"
        )

async def call_judge_llm(messages, model: str):
    """Call judge LLM with given messages"""
    resp = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # Zero temperature for maximum consistency
        max_tokens=500,   # Limit response length to force concise JSON
    )
    return resp.choices[0].message.content.strip()


JUDGE_SYSTEM_PROMPT = """You are an expert judge for code feedback quality.

IMPORTANT CONTEXT:
- The feedback is meant to be GENERAL and CONCEPTUAL (not specific to one student)
- The goal is teaching REUSABLE programming principles
- Generic feedback is GOOD if it's factually correct and captures the right concept

Evaluate on these criteria:

1. FACTUAL CORRECTNESS: Does the feedback correctly describe a real issue or concept for this type of code?
2. CONCEPTUAL CLARITY: Is the general principle clearly expressed?
3. APPLICABILITY: Would this feedback apply to similar code patterns?

**CRITICAL**: Being generic/conceptual is NOT a problem. Only flag issues if the feedback is:
- Factually wrong about the code behavior
- Contradicts what the code actually does
- Teaches an incorrect programming principle

Respond with ONLY this JSON (no other text, no code):

{
  "quality_score": 85,
  "is_factually_correct": true,
  "is_conceptually_clear": true,
  "is_applicable": true,
  "issues": [],
  "needs_correction": false,
  "corrected_feedback": ""
}

Scoring:
- 90-100: Excellent - Correct principle, clearly expressed
- 70-89:  Good - Mostly correct, minor clarity issues
- 50-69:  Mediocre - Correct but poorly expressed
- 30-49:  Poor - Contains some factual errors
- 0-29:   Bad - Factually wrong or misleading

needs_correction: true only if quality_score < 70
corrected_feedback: Provide ONLY if needs_correction is true

**OUTPUT ONLY JSON. NO CODE. NO EXPLANATIONS.**"""


def create_judge_prompt(code: str, feedback: str, feedback_type: str) -> List[Dict[str, str]]:
    """Create a prompt for the judge to evaluate feedback quality"""

    user_content = f"""Evaluate the quality of this {feedback_type} feedback:

CODE:
```c
{code}
```

FEEDBACK TO EVALUATE:
{feedback}

Respond with a JSON evaluation following the schema in the system prompt."""

    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


async def validate_single_item(item: Dict[str, Any], judge_model: str, feedback_types: List[str]) -> Dict[str, Any]:
    """Validate feedback for a single item"""

    code = item.get('code_snippet', '')
    validations = {}

    for feedback_type in feedback_types:
        feedback = item.get(feedback_type, '')

        if not feedback:
            continue

        prompt = create_judge_prompt(code, feedback, feedback_type)

        try:
            response = await call_judge_llm(prompt, judge_model)

            # Parse JSON response
            # Sometimes LLMs wrap JSON in ```json ... ```, handle that
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()

            validation_result = json.loads(response)
            validations[f"{feedback_type}_validation"] = validation_result

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON for {feedback_type}: {e}")
            logging.error(f"Response was: {response}")
            validations[f"{feedback_type}_validation"] = {
                "quality_score": 0,
                "is_correct": False,
                "is_specific": False,
                "is_actionable": False,
                "issues": ["JSON parsing failed"],
                "needs_correction": True,
                "corrected_feedback": ""
            }
        except Exception as e:
            logging.error(f"Error validating {feedback_type}: {e}")
            validations[f"{feedback_type}_validation"] = {
                "quality_score": 0,
                "is_correct": False,
                "is_specific": False,
                "is_actionable": False,
                "issues": [f"Validation error: {str(e)}"],
                "needs_correction": True,
                "corrected_feedback": ""
            }

    # Merge validations into item
    item.update(validations)
    return item


async def validate_batch(
    batch_items: List[Dict[str, Any]],
    judge_model: str,
    feedback_types: List[str]
) -> List[Dict[str, Any]]:
    """Validate a batch of items concurrently"""

    tasks = [validate_single_item(item, judge_model, feedback_types) for item in batch_items]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    validated_batch = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logging.error(f"Batch validation error for item {i}: {result}")
            # Keep original item without validation
            validated_batch.append(batch_items[i])
        else:
            validated_batch.append(result)

    return validated_batch


def run_feedback_validation(
    input_dataset_path: str,
    output_dataset_path: str,
    judge_model: str = "gpt-4",  # Or any other model
    feedback_types: List[str] = None,
    batch_size: int = 4,  # Smaller batch for larger models
    base_url: str = "http://localhost:1234/v1"
) -> str:
    """
    Validate feedback quality using a judge LLM

    Args:
        input_dataset_path: Path to JSONL dataset with feedback to validate
        output_dataset_path: Path to save validated dataset
        judge_model: Name of the judge model (e.g., "gpt-4", "llama-3.1-70b-instruct")
        feedback_types: List of feedback fields to validate (default: all)
        batch_size: Number of items to process in parallel
        base_url: API base URL for the judge model

    Returns:
        Path to validated dataset
    """

    print(f"[Feedback Validation] Starting validation with judge model: {judge_model}")
    print(f"[Feedback Validation] API Base URL: {base_url}")

    # Initialize client
    initialize_judge_client(base_url)

    # Default feedback types to validate
    if feedback_types is None:
        feedback_types = [
            'generated_feedback',
            'refined_feedback',
            'conceptual_feedback',
            'negative_feedback'
        ]

    print(f"[Feedback Validation] Validating feedback types: {', '.join(feedback_types)}")

    # Load input dataset
    if not os.path.exists(input_dataset_path):
        raise FileNotFoundError(f"Input dataset not found: {input_dataset_path}")

    with open(input_dataset_path, 'r', encoding='utf-8') as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    total_items = len(all_items)
    print(f"[Feedback Validation] Loaded {total_items} items")

    # Check for existing progress
    progress = load_progress(output_dataset_path, id_column='code_id')

    items_to_process = [
        item for item in all_items
        if item['code_id'] not in progress
    ]
    num_to_process = len(items_to_process)

    if num_to_process == 0:
        print("[Feedback Validation] All items already validated. Skipping.")
        return output_dataset_path

    print(f"[Feedback Validation] Found {len(progress)} validated items. {num_to_process} remaining.")

    # Process in batches
    num_batches = math.ceil(num_to_process / batch_size)
    print(f"[Feedback Validation] Processing {num_batches} batches...")

    for i in tqdm(range(0, num_to_process, batch_size), total=num_batches, desc="Validating"):
        batch_items = items_to_process[i : i + batch_size]

        try:
            validated_batch = asyncio.run(
                validate_batch(batch_items, judge_model, feedback_types)
            )
            save_result_batch(output_dataset_path, validated_batch)

        except Exception as e:
            logging.error(f"Critical error on batch {i}: {e}")
            continue

    print(f"[Feedback Validation] ✓ Validation completed. Output: {output_dataset_path}")

    # Generate summary statistics
    print_validation_summary(output_dataset_path, feedback_types)

    return output_dataset_path


def print_validation_summary(dataset_path: str, feedback_types: List[str]):
    """Print summary statistics of validation results"""

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    for feedback_type in feedback_types:
        validation_key = f"{feedback_type}_validation"

        scores = []
        factually_correct_count = 0
        conceptually_clear_count = 0
        applicable_count = 0
        needs_correction_count = 0

        for item in data:
            val = item.get(validation_key, {})
            if val:
                scores.append(val.get('quality_score', 0))
                if val.get('is_factually_correct', False):
                    factually_correct_count += 1
                if val.get('is_conceptually_clear', False):
                    conceptually_clear_count += 1
                if val.get('is_applicable', False):
                    applicable_count += 1
                if val.get('needs_correction', False):
                    needs_correction_count += 1

        if scores:
            print(f"\n{feedback_type.upper()}:")
            print(f"  Mean Quality Score:      {sum(scores)/len(scores):.1f}/100")
            print(f"  Factually Correct:       {factually_correct_count}/{len(scores)} ({factually_correct_count/len(scores)*100:.1f}%)")
            print(f"  Conceptually Clear:      {conceptually_clear_count}/{len(scores)} ({conceptually_clear_count/len(scores)*100:.1f}%)")
            print(f"  Applicable (Generic):    {applicable_count}/{len(scores)} ({applicable_count/len(scores)*100:.1f}%)")
            print(f"  Needs Correction:        {needs_correction_count}/{len(scores)} ({needs_correction_count/len(scores)*100:.1f}%)")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate feedback quality using judge LLM")
    parser.add_argument("input", help="Input JSONL file with feedback")
    parser.add_argument("output", help="Output JSONL file with validations")
    parser.add_argument("--judge-model", default="gpt-4", help="Judge model name")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="API base URL")
    parser.add_argument("--feedback-types", nargs="+", help="Feedback types to validate")

    args = parser.parse_args()

    run_feedback_validation(
        input_dataset_path=args.input,
        output_dataset_path=args.output,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        base_url=args.base_url,
        feedback_types=args.feedback_types
    )
