import json
import asyncio
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from config import ProxyConfig
from embedder import Embedder
from memory_manager import MemoryManager
from context_injection import build_memory_block, inject_context_into_messages, inject_context_into_system

logger = logging.getLogger(__name__)

config: ProxyConfig = None
embedder: Embedder = None
memory: MemoryManager = None
http_client: httpx.AsyncClient = None

app = FastAPI(title="Ollama Memory Proxy")


@app.on_event("startup")
async def startup():
    global config, embedder, memory, http_client
    config = ProxyConfig.from_env()

    logger.info("Loading embedding model (first time may download ~80MB)...")
    embedder = Embedder(config)
    # Warmup to ensure model is fully loaded
    embedder.embed("warmup")
    logger.info(f"Embedding model '{config.embedding_model}' loaded on {config.embedding_device}")

    memory = MemoryManager(config)
    logger.info(f"Memory initialized: {memory.count} stored contexts")

    http_client = httpx.AsyncClient(
        base_url=config.ollama_base_url,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )
    logger.info(f"Proxy ready: :{config.proxy_port} -> {config.ollama_base_url}")


@app.on_event("shutdown")
async def shutdown():
    memory.save()
    await http_client.aclose()
    logger.info("Proxy shut down. Memory saved.")


# ==================================================================
# /api/chat -- Intercepted endpoint with memory augmentation
# ==================================================================

@app.post("/api/chat")
async def chat(request: Request):
    # Robust JSON parsing: handle encoding issues from various clients
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"JSON parse failed ({e}), trying with utf-8 replace...")
        try:
            body = json.loads(raw_body.decode("utf-8", errors="replace"))
        except Exception:
            # Last resort: forward raw to Ollama without memory
            logger.error("Cannot parse request body, forwarding raw to Ollama")
            return await _stream_passthrough("POST", "/api/chat", {}, raw_body)

    messages = body.get("messages", [])
    stream = body.get("stream", True)
    model_name = body.get("model", "unknown")

    # Step 1: Extract latest user message
    user_text = _extract_last_user_message(messages)

    # Step 2: Generate embedding and search memory
    results = []
    if user_text and memory.count > 0:
        try:
            query_embedding = await asyncio.to_thread(embedder.embed, user_text)
            results = await asyncio.to_thread(
                memory.search_relevant,
                query_embedding,
                config.search_top_k,
                config.similarity_threshold,
            )
            if results:
                logger.info(
                    f"Memory: {len(results)} results found "
                    f"(best sim: {results[0]['similarity']:.3f})"
                )
        except Exception as e:
            logger.error(f"Memory search failed: {e}")

    # Step 3: Inject memory context (always when memory has data)
    context_block = build_memory_block(results, config, memory.count)
    if context_block:
        body["messages"] = inject_context_into_messages(messages, context_block)

    # Step 4: Forward to Ollama
    if stream:
        return await _stream_chat(body, user_text, model_name)
    else:
        return await _non_stream_chat(body, user_text, model_name)


async def _stream_chat(body: dict, user_text: str, model_name: str):
    """Handle streaming /api/chat with NDJSON passthrough."""
    collected_response = []

    async def generate():
        async with http_client.stream("POST", "/api/chat", json=body) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                yield line + "\n"

                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        collected_response.append(content)
                except json.JSONDecodeError:
                    pass

        # Store in memory after stream completes
        assistant_text = "".join(collected_response)
        asyncio.create_task(
            _store_conversation(user_text, assistant_text, model_name)
        )

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


async def _non_stream_chat(body: dict, user_text: str, model_name: str):
    """Handle non-streaming /api/chat."""
    response = await http_client.post("/api/chat", json=body)
    data = response.json()

    assistant_text = data.get("message", {}).get("content", "")
    asyncio.create_task(
        _store_conversation(user_text, assistant_text, model_name)
    )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type="application/json",
    )


# ==================================================================
# /api/generate -- Intercepted endpoint (used by `ollama run` CLI)
# ==================================================================

@app.post("/api/generate")
async def generate(request: Request):
    """Intercept /api/generate (used by `ollama run`) with memory augmentation."""
    raw_body = await request.body()
    try:
        body = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"JSON parse failed on /api/generate ({e}), trying utf-8 replace...")
        try:
            body = json.loads(raw_body.decode("utf-8", errors="replace"))
        except Exception:
            logger.error("Cannot parse /api/generate body, forwarding raw to Ollama")
            return await _stream_passthrough("POST", "/api/generate", {}, raw_body)

    prompt = body.get("prompt", "")
    system_prompt = body.get("system", "")
    stream = body.get("stream", True)
    model_name = body.get("model", "unknown")

    # Search memory and inject context into system prompt
    results = []
    if prompt and memory.count > 0:
        try:
            query_embedding = await asyncio.to_thread(embedder.embed, prompt)
            results = await asyncio.to_thread(
                memory.search_relevant,
                query_embedding,
                config.search_top_k,
                config.similarity_threshold,
            )
            if results:
                logger.info(
                    f"Memory(/api/generate): {len(results)} results found "
                    f"(best sim: {results[0]['similarity']:.3f})"
                )
        except Exception as e:
            logger.error(f"Memory search failed on /api/generate: {e}")

    # Always inject memory block when memory has data
    context_block = build_memory_block(results, config, memory.count)
    if context_block:
        body["system"] = inject_context_into_system(system_prompt, context_block)

    # Forward to Ollama
    if stream:
        return await _stream_generate(body, prompt, model_name)
    else:
        return await _non_stream_generate(body, prompt, model_name)


async def _stream_generate(body: dict, user_text: str, model_name: str):
    """Handle streaming /api/generate with NDJSON passthrough."""
    collected_response = []

    async def gen():
        async with http_client.stream("POST", "/api/generate", json=body) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                yield line + "\n"

                try:
                    chunk = json.loads(line)
                    content = chunk.get("response", "")
                    if content:
                        collected_response.append(content)
                except json.JSONDecodeError:
                    pass

        # Store in memory after stream completes
        assistant_text = "".join(collected_response)
        asyncio.create_task(
            _store_conversation(user_text, assistant_text, model_name)
        )

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


async def _non_stream_generate(body: dict, user_text: str, model_name: str):
    """Handle non-streaming /api/generate."""
    response = await http_client.post("/api/generate", json=body)
    data = response.json()

    assistant_text = data.get("response", "")
    asyncio.create_task(
        _store_conversation(user_text, assistant_text, model_name)
    )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type="application/json",
    )


async def _store_conversation(
    user_text: str, assistant_text: str, model_name: str
):
    """Store user and assistant messages in memory (background task)."""
    try:
        if user_text and len(user_text) > 5:
            user_emb = await asyncio.to_thread(embedder.embed, user_text)
            await asyncio.to_thread(
                memory.store_message, user_emb, user_text, "user", model_name
            )

        if assistant_text and len(assistant_text) > 20 and not _is_unhelpful(assistant_text):
            asst_emb = await asyncio.to_thread(embedder.embed, assistant_text)
            await asyncio.to_thread(
                memory.store_message,
                asst_emb,
                assistant_text,
                "assistant",
                model_name,
            )

        # Auto-save to disk after each conversation (prevents data loss on crash)
        await asyncio.to_thread(memory.save)

        logger.debug(
            f"Stored+saved: user={len(user_text or '')}ch, "
            f"assistant={len(assistant_text or '')}ch, "
            f"total={memory.count} contexts"
        )
    except Exception as e:
        logger.error(f"Failed to store conversation: {e}")


# ==================================================================
# Catch-all: proxy everything else to Ollama unchanged
# ==================================================================

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_passthrough(request: Request, path: str):
    """Forward any non-chat request to Ollama as-is."""
    url = f"/{path}"
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "transfer-encoding")
    }
    body = await request.body()

    if request.method == "POST":
        return await _stream_passthrough(request.method, url, headers, body)
    else:
        response = await http_client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )


async def _stream_passthrough(method: str, url: str, headers: dict, body: bytes):
    """Stream a POST response from Ollama back to the client."""

    async def generate():
        async with http_client.stream(
            method, url, headers=headers, content=body
        ) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# ==================================================================
# Helpers
# ==================================================================

_UNHELPFUL_PHRASES = [
    "i don't have access",
    "i don't have persistent memory",
    "i don't have a persistent memory",
    "i cannot remember",
    "i can't remember previous",
    "i don't have any actual information",
    "nu am acces la",
    "nu am memorie",
    "nu pot accesa",
    "nu am informatii",
]


def _is_unhelpful(text: str) -> bool:
    """Check if an assistant response is a generic refusal that shouldn't be stored."""
    lower = text.lower()
    return any(phrase in lower for phrase in _UNHELPFUL_PHRASES)


def _extract_last_user_message(messages: list) -> Optional[str]:
    """Extract the text of the most recent user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Multimodal: extract text parts
                content = " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and "text" in part
                )
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None
