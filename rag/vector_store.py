"""Pure-Python vector store using numpy for cosine similarity.

Replaces ChromaDB to avoid C++ / pydantic-settings compatibility issues
on Python 3.14+. Data is persisted as a JSON file alongside the embeddings
in a numpy .npy file for fast loading.
"""

import json
import os
from typing import List, Dict, Any, Optional

import numpy as np

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K_RESULTS, EMBEDDING_DIM
from .embeddings import EmbeddingGenerator


class VectorStore:
    """File-backed vector store with cosine-similarity search."""

    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self._meta_path = os.path.join(persist_dir, f"{COLLECTION_NAME}_meta.json")
        self._vecs_path = os.path.join(persist_dir, f"{COLLECTION_NAME}_vecs.npy")

        self.embedding_generator = EmbeddingGenerator()
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self):
        """Load metadata and vectors from disk."""
        if os.path.exists(self._meta_path) and os.path.exists(self._vecs_path):
            with open(self._meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ids: List[str] = data["ids"]
            self._documents: List[str] = data["documents"]
            self._metadatas: List[Dict] = data["metadatas"]
            self._vectors: np.ndarray = np.load(self._vecs_path)
        else:
            self._ids = []
            self._documents = []
            self._metadatas = []
            self._vectors = np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    def _save(self):
        """Persist metadata and vectors to disk."""
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"ids": self._ids, "documents": self._documents, "metadatas": self._metadatas},
                f,
                ensure_ascii=False,
                indent=2,
            )
        np.save(self._vecs_path, self._vectors)

    # ------------------------------------------------------------------
    # Public API (mirrors the old ChromaDB-based API)
    # ------------------------------------------------------------------

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed and store document chunks.

        Args:
            chunks: List of dicts with keys: id, content, metadata.
        """
        if not chunks:
            print("No chunks to add.")
            return

        new_vecs = []
        print(f"Generating embeddings for {len(chunks)} chunks...")

        for i, chunk in enumerate(chunks):
            embedding = self.embedding_generator.generate_embedding(chunk["content"])

            if embedding:
                vec = np.array(embedding, dtype=np.float32)
            else:
                print(f"Warning: zero-vector fallback for chunk {chunk['id']}")
                vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

            # Normalise to unit length for cosine similarity via dot product
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            self._ids.append(chunk["id"])
            self._documents.append(chunk["content"])
            self._metadatas.append(chunk["metadata"])
            new_vecs.append(vec)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks")

        new_matrix = np.stack(new_vecs, axis=0)  # (N, D)
        self._vectors = (
            np.vstack([self._vectors, new_matrix])
            if self._vectors.shape[0] > 0
            else new_matrix
        )
        self._save()
        print(f"Added {len(chunks)} chunks. Total: {len(self._ids)}")

    def query(
        self,
        query_text: str,
        k: int = TOP_K_RESULTS,
        filter_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-k most similar chunks for query_text.

        Args:
            query_text: The user's query.
            k: How many results to return.
            filter_topic: If set, only consider chunks whose metadata['topic'] matches.

        Returns:
            List of dicts with keys: content, metadata, distance (1 - cosine_sim).
        """
        if len(self._ids) == 0:
            return []

        q_emb = self.embedding_generator.generate_query_embedding(query_text)
        if not q_emb:
            print("Failed to generate query embedding.")
            return []

        q_vec = np.array(q_emb, dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm

        # Build candidate index list (apply topic filter if requested)
        if filter_topic:
            candidate_idx = [
                i for i, m in enumerate(self._metadatas) if m.get("topic") == filter_topic
            ]
        else:
            candidate_idx = list(range(len(self._ids)))

        if not candidate_idx:
            return []

        sub_vecs = self._vectors[candidate_idx]           # (M, D)
        sims = sub_vecs @ q_vec                            # cosine similarity (unit vecs)

        top_k = min(k, len(candidate_idx))
        top_local_idx = np.argsort(sims)[::-1][:top_k]   # descending similarity

        results = []
        for local_i in top_local_idx:
            global_i = candidate_idx[local_i]
            results.append(
                {
                    "content": self._documents[global_i],
                    "metadata": self._metadatas[global_i],
                    "distance": float(1.0 - sims[local_i]),  # cosine distance
                }
            )
        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic stats about the vector store."""
        return {
            "collection_name": COLLECTION_NAME,
            "document_count": len(self._ids),
            "persist_directory": self.persist_dir,
        }

    def clear_collection(self) -> None:
        """Remove all stored vectors and metadata."""
        self._ids = []
        self._documents = []
        self._metadatas = []
        self._vectors = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        self._save()
        print("Collection cleared.")


if __name__ == "__main__":
    store = VectorStore()
    print("Stats:", store.get_collection_stats())

    results = store.query("How do I track my progress?", k=3)
    print(f"\nFound {len(results)} results")
    for i, r in enumerate(results):
        print(f"\n{i+1}. Topic: {r['metadata'].get('topic', 'N/A')}")
        print(f"   Distance: {r['distance']:.4f}")
        print(f"   Content: {r['content'][:150]}...")
