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
from context_injection import format_memory_context, inject_context_into_messages

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
    body = await request.json()

    messages = body.get("messages", [])
    stream = body.get("stream", True)
    model_name = body.get("model", "unknown")

    # Step 1: Extract latest user message
    user_text = _extract_last_user_message(messages)

    # Step 2: Generate embedding and search memory
    context_block = ""
    if user_text and memory.count > 0:
        query_embedding = await asyncio.to_thread(embedder.embed, user_text)
        results = await asyncio.to_thread(
            memory.search_relevant,
            query_embedding,
            config.search_top_k,
            config.similarity_threshold,
        )
        if results:
            context_block = format_memory_context(results, config)
            logger.info(
                f"Memory: {len(results)} results injected "
                f"(best sim: {results[0]['similarity']:.3f})"
            )

    # Step 3: Inject context into messages
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


async def _store_conversation(
    user_text: str, assistant_text: str, model_name: str
):
    """Store user and assistant messages in memory (background task)."""
    try:
        if user_text:
            user_emb = await asyncio.to_thread(embedder.embed, user_text)
            await asyncio.to_thread(
                memory.store_message, user_emb, user_text, "user", model_name
            )

        if assistant_text:
            asst_emb = await asyncio.to_thread(embedder.embed, assistant_text)
            await asyncio.to_thread(
                memory.store_message,
                asst_emb,
                assistant_text,
                "assistant",
                model_name,
            )

        logger.debug(
            f"Stored: user={len(user_text or '')}ch, "
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
