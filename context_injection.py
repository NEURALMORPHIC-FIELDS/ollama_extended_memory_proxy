import time
from typing import List, Dict, Any

from config import ProxyConfig


def format_memory_context(
    results: List[Dict[str, Any]],
    config: ProxyConfig,
) -> str:
    """Format search results into a context block for the system message."""
    if not results:
        return ""

    lines = []
    total_chars = 0

    for r in results[: config.max_context_items]:
        meta = r["metadata"]
        text = meta.get("text", "")
        role = meta.get("role", "unknown")
        sim = r["similarity"]
        ts = meta.get("timestamp", 0)

        age = _format_age(ts)

        remaining_budget = config.max_context_chars - total_chars
        if remaining_budget <= 0:
            break

        if len(text) > remaining_budget:
            text = text[:remaining_budget] + "..."

        lines.append(f"[{role}] ({age}, relevance: {sim:.0%}): {text}")
        total_chars += len(lines[-1])

    if not lines:
        return ""

    return (
        "The following are relevant excerpts from previous conversations "
        "that may provide useful context. Use them if helpful, ignore if not relevant.\n\n"
        + "\n".join(lines)
    )


def inject_context_into_messages(
    messages: List[Dict[str, str]],
    context_block: str,
) -> List[Dict[str, str]]:
    """Inject memory context into the message list.

    If a system message exists as the first message, appends to it.
    Otherwise prepends a new system message.
    """
    if not context_block:
        return messages

    messages = [m.copy() for m in messages]

    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = (
            messages[0]["content"] + "\n\n---\n" + context_block
        )
    else:
        messages.insert(0, {"role": "system", "content": context_block})

    return messages


def _format_age(timestamp: float) -> str:
    if timestamp <= 0:
        return "unknown time"
    delta = time.time() - timestamp
    if delta < 60:
        return "just now"
    elif delta < 3600:
        return f"{int(delta / 60)}m ago"
    elif delta < 86400:
        return f"{int(delta / 3600)}h ago"
    else:
        return f"{int(delta / 86400)}d ago"
