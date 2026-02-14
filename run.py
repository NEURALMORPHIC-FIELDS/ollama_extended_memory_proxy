import sys
import os
import logging

# Ensure the proxy directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from config import ProxyConfig


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ProxyConfig.from_env()

    print("=" * 60)
    print("  Ollama Memory Proxy (TRANSPARENT MODE)")
    print("=" * 60)
    print(f"  Proxy:      http://{config.proxy_host}:{config.proxy_port}")
    print(f"  Ollama:     {config.ollama_base_url}")
    print(f"  Embedder:   {config.embedding_model} ({config.embedding_device})")
    print(f"  Memory:     {config.memory_storage_path}")
    print(f"  Threshold:  {config.similarity_threshold}")
    print(f"  Top-K:      {config.search_top_k}")
    print("=" * 60)
    print("  SETUP (one-time):")
    print("    1. Set system env: OLLAMA_HOST=127.0.0.1:11436")
    print("    2. Restart Ollama (it will listen on 11436)")
    print("    3. Start this proxy (listens on 11434)")
    print("  All clients work automatically - no config changes!")
    print("=" * 60)

    uvicorn.run(
        "proxy:app",
        host=config.proxy_host,
        port=config.proxy_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
