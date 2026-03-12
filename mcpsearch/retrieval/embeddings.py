"""
Embedding utilities.

Handles loading the embedding model and
creating embeddings for text or queries.
"""

from sentence_transformers import SentenceTransformer

_model = None


def get_embedding_model():
    """
    Load the embedding model lazily.
    """

    global _model

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


def embed_texts(text_list):
    """
    Generate embeddings for a list of texts.
    """

    model = get_embedding_model()

    embeddings = model.encode(text_list, convert_to_numpy=True)

    return embeddings


def embed_query(query):
    """
    Generate embedding for a single query.
    """

    model = get_embedding_model()

    embedding = model.encode([query], convert_to_numpy=True)

    return embedding
