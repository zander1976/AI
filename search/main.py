#!/usr/bin/env python3
"""
Procedural RAG Pipeline with MLX LLaMA + PDF Document Retrieval

Features:
- Procedural structure with functions
- Embedding caching to avoid recomputation
- Multi-turn query support
- Skips PDF loading if embeddings already exist
"""

import os
import time

import numpy as np
import pdfplumber
from mlx_lm import generate, load
from rich import print
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------
PDF_FILES = ["book.pdf"]  # can add more PDFs here
CHUNK_SIZE_WORDS = 300
TOP_K = 3  # number of chunks to retrieve per query
EMBEDDING_CACHE_FILE = "embeddings_cache.npz"


# -----------------------------
# Functions
# -----------------------------


def load_llm(model_name="mlx-community/Llama-3.2-1B-Instruct-4bit"):
    """Load MLX LLaMA model."""
    print("\n[blue]Step 1: Loading MLX LLaMA model...[/blue]")
    start_time = time.time()
    model, tokenizer = load(model_name)
    print(f"[green]LLM loaded (took {time.time() - start_time:.2f}s)[/green]")
    return model, tokenizer


def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load SentenceTransformer embedding model."""
    print("\n[blue]Step 2: Loading embedding model...[/blue]")
    start_time = time.time()
    model = SentenceTransformer(model_name)
    print(
        f"[green]Embedding model ready (took {time.time() - start_time:.2f}s)[/green]"
    )
    return model


def extract_text_from_pdfs(pdf_files):
    """Extract text from one or more PDFs."""
    all_text = ""
    print("\n[blue]Step 3: Extracting text from PDFs...[/blue]")
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        with pdfplumber.open(pdf_file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
                if i % 10 == 0:
                    print(f"  Extracted page {i + 1}/{len(pdf.pages)}")
    print(f"[green]Finished extracting text ({len(all_text.split())} words)[/green]")
    return all_text


def chunk_text(full_text, chunk_size_words=CHUNK_SIZE_WORDS):
    """Split text into chunks of roughly chunk_size_words."""
    print("\n[blue]Step 4: Chunking text...[/blue]")
    lines = [line.strip() for line in full_text.split("\n") if line.strip()]

    chunks = []
    chunk = []
    words_in_chunk = 0
    for line in lines:
        line_words = len(line.split())
        chunk.append(line)
        words_in_chunk += line_words
        if words_in_chunk >= chunk_size_words:
            chunks.append(" ".join(chunk))
            chunk = []
            words_in_chunk = 0
    if chunk:
        chunks.append(" ".join(chunk))

    print(f"[green]Created {len(chunks)} chunks[/green]")
    print(f"Example chunk (first 300 chars):\n{chunks[0][:300]}")
    return chunks


def load_or_create_chunks_and_embeddings(
    pdf_files, embed_model, cache_file=EMBEDDING_CACHE_FILE
):
    """
    Load chunks + embeddings from cache if available,
    otherwise process PDFs, chunk text, embed, and save.
    """
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        chunks = data["chunks"].tolist()
        chunk_embeddings = data["embeddings"]
        print(f"[green]Loaded cached embeddings from {cache_file}[/green]")
        return chunks, chunk_embeddings

    # Cache does not exist, process PDFs
    full_text = extract_text_from_pdfs(pdf_files)
    chunks = chunk_text(full_text)

    print("\n[blue]Step 5: Generating embeddings for chunks...[/blue]")
    start_time = time.time()
    chunk_embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    chunk_embeddings = chunk_embeddings / np.linalg.norm(
        chunk_embeddings, axis=1, keepdims=True
    )
    np.savez_compressed(cache_file, chunks=chunks, embeddings=chunk_embeddings)
    print(
        f"[green]Chunk embeddings done and cached (took {time.time() - start_time:.2f}s)[/green]"
    )

    return chunks, chunk_embeddings


def ask_question(
    query,
    llm_model,
    llm_tokenizer,
    embed_model,
    chunks,
    chunk_embeddings,
    top_k=TOP_K,
    prev_context="",
):
    """Ask a question using RAG."""
    print(f"\n[cyan]Processing query:[/cyan] {query}")
    start_time = time.time()

    # Embed query
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb)

    # Cosine similarity search
    sims = chunk_embeddings @ q_emb.T
    top_idx = np.argsort(sims.flatten())[-top_k:][::-1]

    # Build context
    context = "\n\n".join([chunks[i] for i in top_idx])
    if prev_context:
        context = prev_context + "\nPrevious answer:\n" + context

    # LLM prompt
    prompt = (
        f"You are a helpful AI assistant.\nUse ONLY the context below to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
    )

    # Generate answer
    answer = generate(llm_model, llm_tokenizer, prompt, max_tokens=256)
    print(f"[green]Query processed (took {time.time() - start_time:.2f}s)[/green]")
    return answer


# -----------------------------
# Main procedure
# -----------------------------
def main():
    print("\n[bold green]Starting Procedural RAG pipeline...[/bold green]")

    llm_model, llm_tokenizer = load_llm()
    embed_model = load_embedding_model()

    # Load chunks and embeddings, skip PDFs if cached
    chunks, chunk_embeddings = load_or_create_chunks_and_embeddings(
        PDF_FILES, embed_model
    )

    # Example usage
    first_answer = ask_question(
        "Summarize the main points of this AI book.",
        llm_model,
        llm_tokenizer,
        embed_model,
        chunks,
        chunk_embeddings,
    )
    print("\n[bold magenta]First answer:[/bold magenta]\n", first_answer)

    follow_up = ask_question(
        "Explain the most important concept in more detail.",
        llm_model,
        llm_tokenizer,
        embed_model,
        chunks,
        chunk_embeddings,
        prev_context=first_answer,
    )
    print("\n[bold magenta]Follow-up answer:[/bold magenta]\n", follow_up)


if __name__ == "__main__":
    main()
