# Ollama Memory Proxy

A transparent proxy that gives **any Ollama model persistent conversation memory** across sessions. Your LLM remembers what you told it yesterday, last week, or months ago.

```
Without memory:  "What is my name?"  →  "I don't know your name."
With memory:     "What is my name?"  →  "Your name is Lucian."
```

## How It Works

```
Client (port 11435)          Ollama (port 11434)
       |                            ^
       v                            |
   Memory Proxy                     |
   1. Embed user query   (~1ms)     |
   2. Search FAISS index (~0.1ms)   |
   3. Inject relevant context       |
   4. Forward to Ollama  ───────────┘
   5. Stream response back
   6. Store conversation (background)
```

The proxy intercepts `/api/chat`, searches past conversations for relevant context using semantic similarity (FAISS + sentence-transformers), injects it into the system message, and forwards to Ollama. All other endpoints (`/api/tags`, `/api/pull`, etc.) pass through unchanged.

**Zero overhead.** Embedding + FAISS search takes ~1ms. The LLM never knows the difference.

## Benchmark Results

Tested with `qwen3:0.6b` — 14 tests across 5 categories:

| Metric | Without Memory | With Memory | Improvement |
|--------|---------------|-------------|-------------|
| **Average Score** | 11.9% | **78.6%** | **+66.7%** |
| Tests Passed | 0/14 | **10/14** | +10 |
| Latency Overhead | — | — | **~0ms** |

| Category | Without | With | Delta |
|----------|---------|------|-------|
| Personal Fact Recall | 10% | **100%** | +90% |
| Multi-Turn Continuity | 11% | **50%** | +39% |
| Temporal Context | 25% | **75%** | +50% |
| Cross-Topic Association | 17% | **50%** | +33% |
| Precision Recall | 0% | **100%** | +100% |

Run `python benchmark.py` to reproduce these results on your hardware. Full report in [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running on port 11434 ([ollama.com](https://ollama.com))
- At least one Ollama model pulled (e.g., `ollama pull qwen3:0.6b`)

### Install

```bash
git clone https://github.com/yourusername/ollama-memory-proxy.git
cd ollama-memory-proxy
pip install -r requirements.txt
```

Or on Windows:
```cmd
git clone https://github.com/yourusername/ollama-memory-proxy.git
cd ollama-memory-proxy
install.bat
```

On Linux/macOS:
```bash
git clone https://github.com/yourusername/ollama-memory-proxy.git
cd ollama-memory-proxy
chmod +x install.sh
./install.sh
```

The first run downloads the embedding model (~80MB). Subsequent starts are instant.

### Run

```bash
python run.py
```

Output:
```
============================================================
  Ollama Memory Proxy
============================================================
  Proxy:      http://0.0.0.0:11435
  Ollama:     http://127.0.0.1:11434
  Embedder:   all-MiniLM-L6-v2 (cpu)
  Memory:     ./ollama_memory_data
  Threshold:  0.3
  Top-K:      5
============================================================
```

### Use

Point any Ollama client to **port 11435** instead of 11434:

**curl:**
```bash
# Tell the LLM something
curl http://localhost:11435/api/chat -d '{
  "model": "qwen3:0.6b",
  "messages": [{"role": "user", "content": "My name is Alice and I work at Google."}],
  "stream": false
}'

# Later (new conversation), ask about it
curl http://localhost:11435/api/chat -d '{
  "model": "qwen3:0.6b",
  "messages": [{"role": "user", "content": "What is my name and where do I work?"}],
  "stream": false
}'
# → "Your name is Alice and you work at Google."
```

**Ollama CLI:**
```bash
# Linux/macOS
export OLLAMA_HOST=http://localhost:11435

# Windows
set OLLAMA_HOST=http://localhost:11435

ollama run qwen3:0.6b
```

**Open WebUI / any Ollama-compatible client:**
Change the Ollama URL in settings to `http://localhost:11435`.

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_HOST` | `0.0.0.0` | Proxy listen address |
| `PROXY_PORT` | `11435` | Proxy listen port |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to inject context |
| `SEARCH_TOP_K` | `5` | Max memory results per query |
| `MEMORY_STORAGE_PATH` | `./ollama_memory_data` | Where to persist memory on disk |

Copy `.env.example` to `.env` to customize.

## Architecture

```
ollama-memory-proxy/
├── run.py                 # Entry point (uvicorn launcher)
├── proxy.py               # FastAPI app — request interception & forwarding
├── embedder.py            # Sentence-transformers wrapper (CPU, 384-dim)
├── memory_manager.py      # FAISS IndexFlatIP + metadata persistence
├── context_injection.py   # Formats & injects memory into messages
├── config.py              # Configuration with env var overrides
├── benchmark.py           # Comprehensive benchmark suite
└── BENCHMARK_REPORT.md    # Latest benchmark results
```

### Key Design Decisions

1. **sentence-transformers on CPU** — doesn't consume GPU VRAM. The `all-MiniLM-L6-v2` model produces 384-dim L2-normalized vectors in ~1ms per query.

2. **FAISS IndexFlatIP** — inner product on L2-normalized vectors = cosine similarity. Exact search, no approximation errors. Scales to 100k+ memories easily.

3. **Background storage** via `asyncio.create_task` — conversations are stored after the response streams back. Zero added latency for the user.

4. **Model-agnostic** — memory is shared across all models. Tell something to `qwen3:0.6b`, ask about it with `deepseek-r1:8b`.

5. **Persistent** — memory survives restarts. FAISS index + metadata are saved to disk on shutdown and loaded on startup.

## Autostart with Windows

To start the proxy automatically when Windows boots:

```cmd
copy start_proxy.vbs "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\"
```

The proxy will run silently in the background alongside Ollama.

## Troubleshooting

**Proxy won't start (port in use):**
```bash
# Find and kill the process using port 11435
netstat -ano | findstr 11435
taskkill /PID <pid> /F
```

**First request times out:**
Ollama needs to load the model into GPU VRAM on first use. This can take 30-300+ seconds depending on model size. Subsequent requests are fast.

**Memory not working:**
Check that the proxy is actually running on 11435 and you're connecting to the proxy port, not directly to Ollama on 11434.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Vasile Lucian Borbeleac
