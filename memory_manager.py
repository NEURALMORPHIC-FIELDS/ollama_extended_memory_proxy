import time
import pickle
import logging
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss

from config import ProxyConfig

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation memory using FAISS directly with raw embeddings.

    Bypasses FCPE encoding (which collapses single-vector inputs to identical
    vectors) and stores L2-normalized embeddings as-is.  Inner product on
    unit vectors = cosine similarity, giving real semantic ranking.
    """

    def __init__(self, config: ProxyConfig):
        self._config = config
        self._storage_path = Path(config.memory_storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._dim = config.embedding_dim
        self._lock = threading.Lock()

        # Paths for persistence
        self._faiss_path = self._storage_path / "faiss_index.bin"
        self._meta_path = self._storage_path / "metadata.pkl"

        # Metadata store: id -> dict
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._id_map: List[int] = []  # position -> context_id
        self._next_id = 0

        # Load existing or create new FAISS index
        if self._faiss_path.exists() and self._meta_path.exists():
            self._load()
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            logger.info(f"Created new FAISS IndexFlatIP(dim={self._dim})")

    def _load(self):
        """Load FAISS index and metadata from disk."""
        try:
            self._index = faiss.read_index(str(self._faiss_path))
            with open(self._meta_path, "rb") as f:
                saved = pickle.load(f)
            self._metadata = saved["metadata"]
            self._id_map = saved["id_map"]
            self._next_id = saved["next_id"]
            logger.info(
                f"Loaded {self._index.ntotal} vectors from {self._faiss_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load saved state: {e}. Creating new index.")
            self._index = faiss.IndexFlatIP(self._dim)
            self._metadata = {}
            self._id_map = []
            self._next_id = 0

    def store_message(
        self,
        embedding: np.ndarray,
        text: str,
        role: str,
        model: str = "",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a conversation message in memory."""
        # Ensure 2D, float32, L2-normalized
        vec = embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        metadata = {
            "text": text,
            "role": role,
            "model": model,
            "timestamp": time.time(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        with self._lock:
            ctx_id = self._next_id
            self._next_id += 1
            self._index.add(vec)
            self._id_map.append(ctx_id)
            self._metadata[ctx_id] = metadata

        return ctx_id

    def search_relevant(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search for relevant past conversations above the similarity threshold."""
        if self._index.ntotal == 0:
            return []

        # Ensure 2D, float32, L2-normalized
        vec = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        k = min(top_k, self._index.ntotal)

        with self._lock:
            scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            ctx_id = self._id_map[idx]
            meta = self._metadata.get(ctx_id, {})
            if score >= similarity_threshold:
                results.append({
                    "ctx_id": ctx_id,
                    "similarity": float(score),
                    "metadata": meta,
                })

        return results

    @property
    def count(self) -> int:
        return self._index.ntotal

    def save(self):
        """Persist FAISS index and metadata to disk."""
        with self._lock:
            faiss.write_index(self._index, str(self._faiss_path))
            with open(self._meta_path, "wb") as f:
                pickle.dump(
                    {
                        "metadata": self._metadata,
                        "id_map": self._id_map,
                        "next_id": self._next_id,
                    },
                    f,
                )
        logger.info(f"Saved {self._index.ntotal} vectors to {self._faiss_path}")

    def stats(self) -> Dict[str, Any]:
        return {
            "num_contexts": self._index.ntotal,
            "dim": self._dim,
            "faiss_index_type": "IndexFlatIP",
            "storage_path": str(self._storage_path),
        }
