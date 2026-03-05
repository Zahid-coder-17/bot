"""Embedding generation using Ollama local API."""

from typing import List
import requests

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIM


class EmbeddingGenerator:
    """Generate embeddings using Ollama's local embedding endpoint."""

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL
        self.model = EMBEDDING_MODEL
        self._embed_url = f"{self.base_url}/api/embeddings"

    def _call_ollama(self, text: str) -> List[float]:
        try:
            response = requests.post(
                self._embed_url,
                json={"model": self.model, "prompt": text},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_embedding(self, text: str) -> List[float]:
        return self._call_ollama(text)

    def generate_query_embedding(self, query: str) -> List[float]:
        return self._call_ollama(query)

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.generate_embedding(t) for t in texts]


if __name__ == "__main__":
    gen = EmbeddingGenerator()
    emb = gen.generate_query_embedding("How is course progress tracked?")
    if emb:
        print(f"Embedding dim: {len(emb)}, first 5: {emb[:5]}")
    else:
        print("Failed. Is Ollama running?")
