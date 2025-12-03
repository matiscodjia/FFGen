#!/usr/bin/env python3
"""
test_setup.py - Verify FFGen setup

Run this to check if everything is configured correctly.
"""

import os
import sys
from pathlib import Path


def check_file(path, name):
    """Check if file exists."""
    if Path(path).exists():
        print(f"[OK] {name}: {path}")
        return True
    else:
        print(f"[MISSING] {name}: {path}")
        return False


def check_env_var(var_name):
    """Check if environment variable is set."""
    value = os.getenv(var_name)
    if value:
        print(f"[OK] {var_name}: {'*' * 8}{value[-4:]}")
        return True
    else:
        print(f"[NOT SET] {var_name}")
        return False


def check_import(module_name):
    """Check if module can be imported."""
    try:
        __import__(module_name)
        print(f"[OK] {module_name} installed")
        return True
    except ImportError:
        print(f"[MISSING] {module_name} not installed")
        return False


def main():
    print("="*70)
    print("FFGen Setup Check")
    print("="*70)

    all_ok = True

    # Check files
    print("\nFiles:")
    all_ok &= check_file("config.yml", "Config")
    all_ok &= check_file("prompt.txt", "Prompt")
    all_ok &= check_file("extract_code.py", "Extract script")
    all_ok &= check_file("generate_feedbacks.py", "Generate script")
    all_ok &= check_file("viewer_app/app.py", "Viewer app")

    # Check data directory
    print("\nData:")
    all_ok &= check_file("data", "Data directory")
    if Path("data/dataset.jsonl").exists():
        with open("data/dataset.jsonl", 'r') as f:
            num_lines = sum(1 for _ in f)
        print(f"[OK] Dataset: {num_lines} items")
    else:
        print(f"[NOT FOUND] Dataset (run extract_code.py)")

    # Check environment
    print("\nEnvironment:")
    api_key_set = check_env_var("MISTRAL_API_KEY")

    # Check dependencies
    print("\nDependencies:")
    deps_ok = True
    deps_ok &= check_import("mistralai")
    deps_ok &= check_import("yaml")
    deps_ok &= check_import("tqdm")
    deps_ok &= check_import("streamlit")

    # Summary
    print("\n" + "="*70)
    if all_ok and deps_ok and api_key_set:
        print("Setup complete! Ready to use.")
        print("\nNext steps:")
        print("  1. python extract_code.py")
        print("  2. python generate_feedbacks.py")
        print("  3. streamlit run apps/viewer.py")
    else:
        print("Setup incomplete. Fix issues above.")
        if not deps_ok:
            print("\nInstall dependencies:")
            print("  uv pip install mistralai pyyaml tqdm streamlit")
        if not api_key_set:
            print("\nSet API key:")
            print("  export MISTRAL_API_KEY='your-key-here'")
    print("="*70)


if __name__ == "__main__":
    main()
