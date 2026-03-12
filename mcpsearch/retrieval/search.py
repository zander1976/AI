"""
Vector search module.

Loads the embedding index and performs
similarity search for queries.
"""

import json

import numpy as np

from retrieval.embeddings import embed_query

INDEX_DIR = "data/index"


class VectorSearch:
    """
    Simple vector search engine using numpy.
    """

    def __init__(self):

        print("Loading vector index...")

        self.embeddings = np.load(f"{INDEX_DIR}/embeddings.npy")

        with open(f"{INDEX_DIR}/chunks.json") as f:
            self.chunks = json.load(f)

        with open(f"{INDEX_DIR}/metadata.json") as f:
            self.metadata = json.load(f)

    def search(self, query, k=5):
        """
        Return top-k most relevant chunks.
        """

        q_emb = embed_query(query)

        sims = self.embeddings @ q_emb.T

        idx = np.argsort(sims.flatten())[-k:][::-1]

        results = []

        for i in idx:
            results.append(
                {
                    "book": self.metadata[i]["book"],
                    "text": self.chunks[i],
                }
            )

        return results
