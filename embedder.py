import numpy as np
from sentence_transformers import SentenceTransformer

from config import ProxyConfig


class Embedder:
    """Singleton wrapper for sentence-transformers embedding model.

    Uses all-MiniLM-L6-v2 on CPU to produce 384-dim L2-normalized vectors.
    Runs on CPU to keep GPU VRAM free for the LLM.
    """

    def __init__(self, config: ProxyConfig):
        self._model = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
        )
        self._dim = config.embedding_dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a 384-dim float32 vector."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.astype(np.float32)

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed multiple texts. Returns shape (N, 384)."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim
