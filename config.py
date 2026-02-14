import os
from dataclasses import dataclass


@dataclass
class ProxyConfig:
    # Network - Transparent proxy mode:
    # Proxy listens on 11434 (default Ollama port) so all clients work automatically.
    # Ollama must be moved to 11436 via OLLAMA_HOST=127.0.0.1:11436
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 11434
    ollama_base_url: str = "http://127.0.0.1:11436"

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_device: str = "cpu"

    # Memory - absolute path so it works regardless of working directory
    memory_storage_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ollama_memory_data"
    )

    # Search
    search_top_k: int = 15
    similarity_threshold: float = 0.3

    # Context Injection
    max_context_items: int = 10
    max_context_chars: int = 3000

    @classmethod
    def from_env(cls) -> "ProxyConfig":
        return cls(
            proxy_host=os.getenv("PROXY_HOST", cls.proxy_host),
            proxy_port=int(os.getenv("PROXY_PORT", str(cls.proxy_port))),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            memory_storage_path=os.getenv(
                "MEMORY_STORAGE_PATH", cls.memory_storage_path
            ),
            similarity_threshold=float(
                os.getenv("SIMILARITY_THRESHOLD", str(cls.similarity_threshold))
            ),
            search_top_k=int(os.getenv("SEARCH_TOP_K", str(cls.search_top_k))),
        )
