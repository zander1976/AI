"""
Index builder.

This module scans the PDF directory,
extracts text, chunks it, creates embeddings,
and stores the search index.
"""

import json
import os

import numpy as np
import pdfplumber

from retrieval.chunking import chunk_text
from retrieval.embeddings import embed_texts

PDF_DIR = "data/pdfs"
INDEX_DIR = "data/index"


def extract_text_from_pdf(path):
    """
    Extract text from a PDF file.
    """

    text = ""

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def build_index():
    """
    Build the vector index from all PDFs.
    """

    os.makedirs(INDEX_DIR, exist_ok=True)

    all_chunks = []
    metadata = []

    print("Scanning PDFs...")

    for file in os.listdir(PDF_DIR):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(PDF_DIR, file)

        print("Processing", file)

        text = extract_text_from_pdf(path)

        chunks = chunk_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)

            metadata.append({"book": file})

    print("Generating embeddings...")

    embeddings = embed_texts(all_chunks)

    print("Saving index...")

    np.save(f"{INDEX_DIR}/embeddings.npy", embeddings)

    with open(f"{INDEX_DIR}/chunks.json", "w") as f:
        json.dump(all_chunks, f)

    with open(f"{INDEX_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Index complete.")
