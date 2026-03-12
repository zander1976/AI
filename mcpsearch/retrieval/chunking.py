"""
Chunking utilities.

This module splits large text documents into
smaller chunks suitable for embeddings.
"""

CHUNK_SIZE = 300


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """
    Split text into chunks of roughly N words.

    Example:
    2000 word document → ~7 chunks
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

    return chunks
