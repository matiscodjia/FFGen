#!/usr/bin/env python3
"""
generate_paraphrases.py - Generate paraphrases for training data augmentation

This script takes feedbacks and generates multiple paraphrases to:
1. Augment positive examples (same meaning, different wording)
2. Create semi-hard negatives (similar but incorrect)

Output format optimized for contrastive learning:
{
  "code_id": "...",
  "code_snippet": "...",
  "positive_feedbacks": ["original", "paraphrase1", "paraphrase2"],
  "negative_feedbacks": ["neg1", "neg_paraphrase1", "neg_paraphrase2"]
}
"""

import os
import json
import asyncio
import yaml
from pathlib import Path
from typing import List, Dict
from tqdm.asyncio import tqdm as async_tqdm
from mistralai import Mistral


class ParaphraseGenerator:
    """Generate paraphrases using Mistral Large API."""

    def __init__(self, api_key: str, prompt_template: str,
                 num_variations: int = 3, temperature: float = 0.8):
        """
        Initialize the paraphrase generator.

        Args:
            api_key: Mistral API key
            prompt_template: Prompt template with {feedback} placeholder
            num_variations: Number of paraphrases to generate
            temperature: Sampling temperature (higher = more variation)
        """
        self.client = Mistral(api_key=api_key)
        self.prompt_template = prompt_template
        self.num_variations = num_variations
        self.temperature = temperature
        self.model = "mistral-large-latest"

    async def generate_paraphrase(self, feedback: str, semaphore: asyncio.Semaphore) -> str:
        """
        Generate a single paraphrase.

        Args:
            feedback: Original feedback text
            semaphore: Semaphore for rate limiting

        Returns:
            Paraphrased feedback text
        """
        async with semaphore:
            try:
                # Format prompt
                prompt = self.prompt_template.format(feedback=feedback)

                # Call Mistral API
                response = await asyncio.to_thread(
                    self.client.chat.complete,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=self.temperature
                )

                # Extract paraphrase
                paraphrase = response.choices[0].message.content.strip()
                return paraphrase

            except Exception as e:
                print(f"\nError generating paraphrase: {e}")
                return feedback  # Return original on error

    async def generate_variations(self, feedback: str, semaphore: asyncio.Semaphore) -> List[str]:
        """
        Generate multiple paraphrases for a feedback.

        Args:
            feedback: Original feedback text
            semaphore: Semaphore for rate limiting

        Returns:
            List of paraphrases (including original as first item)
        """
        # Always include original
        variations = [feedback]

        # Generate paraphrases
        tasks = [
            self.generate_paraphrase(feedback, semaphore)
            for _ in range(self.num_variations - 1)
        ]

        paraphrases = await asyncio.gather(*tasks)

        # Filter out duplicates and errors
        for p in paraphrases:
            if p and p != feedback and p not in variations:
                variations.append(p)

        return variations

    async def process_batch(self, items: List[Dict], max_concurrent: int = 5) -> List[Dict]:
        """
        Process a batch of items and generate paraphrases.

        Args:
            items: List of items with 'generated_feedback' field
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of items with 'positive_feedbacks' field added
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create tasks for all items
        tasks = [
            self.generate_variations(item.get('generated_feedback', ''), semaphore)
            for item in items
        ]

        # Run all tasks with progress bar
        all_variations = await async_tqdm.gather(*tasks, desc="Generating paraphrases")

        # Add variations to items
        for item, variations in zip(items, all_variations):
            item['positive_feedbacks'] = variations

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


def filter_items_without_paraphrases(items: List[Dict]) -> List[Dict]:
    """Filter items that don't have paraphrases yet."""
    return [item for item in items if 'positive_feedbacks' not in item]


async def main():
    """Main paraphrase generation pipeline."""
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
    prompt_file = config['paraphrase_prompt_file']
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    # Load paraphrase config
    para_config = config['paraphrase']
    dataset_file = config['data']['dataset_file']

    print("="*70)
    print("PARAPHRASE GENERATION - FFGen")
    print("="*70)
    print(f"Dataset: {dataset_file}")
    print(f"Variations per feedback: {para_config['num_variations']}")
    print(f"Batch size: {para_config['batch_size']}")
    print()

    # Load dataset
    if not Path(dataset_file).exists():
        print(f"Error: Dataset file not found: {dataset_file}")
        print("Run generate_feedbacks.py first.")
        return

    items = load_jsonl(dataset_file)
    print(f"Loaded {len(items)} items from dataset")

    # Filter items without paraphrases
    items_to_process = filter_items_without_paraphrases(items)
    if not items_to_process:
        print("All items already have paraphrases. Nothing to generate.")
        return

    print(f"Found {len(items_to_process)} items without paraphrases")

    # Filter items without feedbacks
    items_with_feedback = [
        item for item in items_to_process
        if item.get('generated_feedback') and len(item.get('generated_feedback', '')) > 10
    ]

    print(f"Processing {len(items_with_feedback)} items with valid feedbacks")

    # Initialize generator
    generator = ParaphraseGenerator(
        api_key=api_key,
        prompt_template=prompt_template,
        num_variations=para_config['num_variations'],
        temperature=para_config['temperature']
    )

    # Process in batches
    batch_size = para_config['batch_size']
    max_concurrent = para_config['max_concurrent']

    all_processed = []
    num_batches = (len(items_with_feedback) + batch_size - 1) // batch_size

    for i in range(0, len(items_with_feedback), batch_size):
        batch = items_with_feedback[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"\nBatch {batch_num}/{num_batches} ({len(batch)} items)")

        # Generate paraphrases for batch
        processed_batch = await generator.process_batch(batch, max_concurrent)
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
        print(f"Progress saved: {len(all_processed)}/{len(items_with_feedback)} completed")

    print("\n" + "="*70)
    print(f"✓ Paraphrase generation completed")
    print(f"Total items processed: {len(all_processed)}")
    print(f"Dataset saved to: {dataset_file}")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
