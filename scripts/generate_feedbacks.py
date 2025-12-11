#!/usr/bin/env python3
"""
generate_feedbacks.py - Generate feedback using Mistral Large API

This script takes code snippets and generates high-quality feedback using
Mistral Large API with a sophisticated prompt template.
"""

import os
import json
import asyncio
import yaml
from pathlib import Path
from typing import List, Dict
from tqdm.asyncio import tqdm as async_tqdm
from mistralai import Mistral


class FeedbackGenerator:
   
    def __init__(self, api_key: str, prompt_template: str, model: str = "mistral-large-2411",
                 max_tokens: int = 500, temperature: float = 0.7):
        """
        Initialize the feedback generator.

        Args:
            api_key: Mistral API key
            prompt_template: Prompt template with {code} placeholder
            model: Model name to use
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
        """
        self.client = Mistral(api_key=api_key)
        self.prompt_template = prompt_template
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate_single_feedback(self, code: str, semaphore: asyncio.Semaphore) -> str:
        """
        Generate feedback for a single code snippet.

        Args:
            code: Code snippet to analyze
            semaphore: Semaphore for rate limiting

        Returns:
            Generated feedback text
        """
        async with semaphore:
            try:
                # Format prompt with code
                prompt = self.prompt_template
                response = await asyncio.to_thread(
                    self.client.chat.complete,
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}
                        ,{"role": "user", "content": code}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )

                # Extract generated text
                feedback = response.choices[0].message.content.strip()
                return feedback

            except Exception as e:
                print(f"\nError generating feedback: {e}")
                return f"ERROR: {str(e)}"

    async def generate_batch(self, items: List[Dict], max_concurrent: int = 5) -> List[Dict]:
        """
        Generate feedbacks for a batch of items.

        Args:
            items: List of dictionaries with 'code_snippet' field
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of items with added 'generated_feedback' field
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create tasks for all items
        tasks = [
            self.generate_single_feedback(item['code_snippet'], semaphore)
            for item in items
        ]

        # Run all tasks with progress bar
        feedbacks = await async_tqdm.gather(*tasks, desc="Generating")

        # Add feedbacks to items
        for item, feedback in zip(items, feedbacks):
            item['generated_feedback'] = feedback

        return items


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict], file_path: str):
    """Save data to JSONL file."""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def filter_already_processed(items: List[Dict]) -> List[Dict]:
    """Filter out items that already have generated_feedback."""
    return [item for item in items if 'generated_feedback' not in item]


async def main():
    """Main generation pipeline."""
    # Load configuration
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Get API key from environment
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        print("Please set it with: export MISTRAL_API_KEY='your-key-here'")
        return

    # Load prompt template
    prompt_file = config['prompt_file']
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Load generation config
    gen_config = config['generation']
    dataset_file = config['data']['dataset_file']

    print("="*70)
    print("FEEDBACK GENERATION - FFGen")
    print("="*70)
    print(f"Dataset: {dataset_file}")
    print(f"Prompt: {prompt_file}")
    print(f"Batch size: {gen_config['batch_size']}")
    print(f"Max concurrent: {gen_config['max_concurrent']}")
    print()

    # Load dataset
    if not Path(dataset_file).exists():
        print(f"Error: Dataset file not found: {dataset_file}")
        print("Run extract_code.py first to generate the dataset.")
        return

    items = load_jsonl(dataset_file)
    print(f"Loaded {len(items)} items from dataset")

    # Filter already processed
    items_to_process = filter_already_processed(items)
    if not items_to_process:
        print("All items already have feedback. Nothing to generate.")
        return

    print(f"Found {len(items_to_process)} items without feedback")

    # Initialize generator
    generator = FeedbackGenerator(
        api_key=api_key,
        prompt_template=prompt_template,
        max_tokens=gen_config['max_tokens'],
        temperature=gen_config['temperature']
    )

    # Process in batches
    batch_size = gen_config['batch_size']
    max_concurrent = gen_config['max_concurrent']

    all_processed = []
    num_batches = (len(items_to_process) + batch_size - 1) // batch_size

    for i in range(0, len(items_to_process), batch_size):
        batch = items_to_process[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"\nBatch {batch_num}/{num_batches} ({len(batch)} items)")

        # Generate feedbacks for batch
        processed_batch = await generator.generate_batch(batch, max_concurrent)
        all_processed.extend(processed_batch)

        # Save progress after each batch
        # Merge with already processed items
        all_items_updated = items.copy()
        processed_ids = {item['code_id'] for item in all_processed}

        # Replace items with processed versions
        for j, item in enumerate(all_items_updated):
            if item['code_id'] in processed_ids:
                # Find the processed version
                processed_item = next(p for p in all_processed if p['code_id'] == item['code_id'])
                all_items_updated[j] = processed_item

        save_jsonl(all_items_updated, dataset_file)
        print(f"Progress saved: {len(all_processed)}/{len(items_to_process)} completed")

    print("\n" + "="*70)
    print(f"✓ Generation completed successfully")
    print(f"Total feedbacks generated: {len(all_processed)}")
    print(f"Dataset saved to: {dataset_file}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
