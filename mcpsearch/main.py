#!/usr/bin/env python3
"""
Main entry point: builds index if missing and launches the async agent.
"""

import asyncio
import os

from agent.agent import run_agent  # async function
from retrieval.index_builder import build_index

INDEX_FILE = "data/index/embeddings.npy"


def ensure_index():
    """Check if vector index exists, build if missing."""
    if not os.path.exists(INDEX_FILE):
        print("Vector index not found. Building index from PDFs...")
        build_index()
        print("Index build complete.")
    else:
        print("Vector index found.")


def main():
    print("\nStarting Book Research Agent\n")
    ensure_index()
    asyncio.run(run_agent())  # run the async agent


if __name__ == "__main__":
    main()
