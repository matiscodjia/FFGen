"""
FFGen - Focused Feedback Generation

Minimal setup.py for development installation.
"""

from setuptools import setup, find_packages

setup(
    name="ffgen",
    version="1.0.0",
    description="Training pipeline for code feedback embeddings with triplet loss",
    author="Matis Codjia",
    packages=find_packages(where=".", exclude=["tests*", "notebooks*", ".venv*"]),
    python_requires=">=3.11",
)
