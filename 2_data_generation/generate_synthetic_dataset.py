#!/usr/bin/env python3
"""
Generate a synthetic dataset with obvious patterns for sanity checking training.
Each example has clear anchor-positive-negative distinction.
"""

import json
import uuid
from typing import List, Dict

def generate_synthetic_examples(num_examples: int = 500) -> List[Dict]:
    """Generate synthetic triplets with obvious patterns."""

    examples = []

    # Pattern templates with very distinct positive vs negative feedback
    patterns = [
        {
            "code": "int add(int a, int b)\n{\n    return a + b\n}",
            "positive": "Missing semicolon after return statement. Add ';' at the end of the return line.",
            "negative": "Array index out of bounds. Check your array access and ensure indices are within valid range.",
        },
        {
            "code": "int x;\nprintf(\"%d\", x);",
            "positive": "Variable 'x' is used without being initialized. Initialize the variable before use.",
            "negative": "Memory leak detected. Make sure to free all dynamically allocated memory.",
        },
        {
            "code": "void func()\n{\n    int arr[5];\n    arr[10] = 42;\n}",
            "positive": "Array index out of bounds. Index 10 exceeds array size of 5.",
            "negative": "Function has wrong return type. Change return type from void to int.",
        },
        {
            "code": "char *str = \"hello\";\nstr[0] = 'H';",
            "positive": "Attempting to modify string literal. String literals are read-only in C.",
            "negative": "Infinite loop detected. Add a proper loop termination condition.",
        },
        {
            "code": "int *ptr = malloc(100);\nptr = NULL;",
            "positive": "Memory leak: allocated memory is lost without being freed before pointer reassignment.",
            "negative": "Missing semicolon at end of statement. Add ';' to terminate the line.",
        },
        {
            "code": "int divide(int a, int b)\n{\n    return a / b;\n}",
            "positive": "Potential division by zero error. Add check for b != 0 before division.",
            "negative": "Variable declared but never used. Remove unused variable declaration.",
        },
        {
            "code": "while(1)\n{\n    printf(\"hello\");\n}",
            "positive": "Infinite loop without break condition. Add a way to exit the loop.",
            "negative": "Uninitialized pointer dereference. Initialize pointer before accessing its value.",
        },
        {
            "code": "int foo()\n{\n    // no return\n}",
            "positive": "Function declared to return int but has no return statement.",
            "negative": "Buffer overflow risk. Validate input size before copying to buffer.",
        },
        {
            "code": "int *p;\n*p = 5;",
            "positive": "Dereferencing uninitialized pointer. Initialize pointer before use.",
            "negative": "Syntax error: missing parentheses in function call.",
        },
        {
            "code": "char buffer[10];\nstrcpy(buffer, \"This is a very long string\");",
            "positive": "Buffer overflow: source string is larger than destination buffer size.",
            "negative": "Wrong variable type. Change from char to int for numeric operations.",
        }
    ]

    # Generate examples by repeating patterns
    for i in range(num_examples):
        pattern = patterns[i % len(patterns)]

        example = {
            "code_id": str(uuid.uuid4()),
            "author_id": str(uuid.uuid4()),
            "code_snippet": pattern["code"],
            "generated_feedback": pattern["positive"],
            "refined_feedback": pattern["positive"],
            "negative_feedback": pattern["negative"],
            "conceptual_feedback": pattern["positive"],
            "hard_negative_feedback": pattern["negative"],  # FIXED: was pattern["positive"]
            "hard_negative_similarity": 0.3,  # Low similarity for easy distinction
            "hard_negative_source_id": str(uuid.uuid4()),
            "llm_negative_feedback": pattern["negative"]
        }

        examples.append(example)

    return examples


def main():
    print("🔧 Generating synthetic dataset...")

    # Generate 500 examples
    examples = generate_synthetic_examples(500)

    # Save to JSONL
    output_path = "data/synthetic_simple.jsonl"
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f" Generated {len(examples)} synthetic examples")
    print(f" Saved to: {output_path}")

    # Show statistics
    print(f"\nDataset composition:")
    print(f"   - 10 distinct error patterns")
    print(f"   - Each pattern repeated ~50 times")
    print(f"   - Very clear positive/negative distinction")
    print(f"   - Low similarity between pos/neg (0.3)")

    # Show example
    print(f"\nExample triplet:")
    ex = examples[0]
    print(f"   Anchor (code):")
    print(f"      {ex['code_snippet'][:60]}...")
    print(f"   Positive (should be close):")
    print(f"      {ex['hard_negative_feedback'][:60]}...")
    print(f"   Negative (should be far):")
    print(f"      {ex['llm_negative_feedback'][:60]}...")


if __name__ == "__main__":
    main()
