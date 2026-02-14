import time
from typing import List, Dict, Any

from config import ProxyConfig


# Base instruction always injected when the memory system has stored data.
# This ensures the model acknowledges memory even when no specific results match.
MEMORY_BASE_PROMPT = (
    "You have access to a LOCAL MEMORY system that persistently stores all conversations. "
    "You CAN and DO remember past interactions with this user. "
    "If the user asks whether you have memory or can remember things, "
    "confirm that YES, you have persistent local memory across conversations. "
    "Never say you cannot remember or that you lack memory — you have it."
)


def build_memory_block(
    results: List[Dict[str, Any]],
    config: ProxyConfig,
    total_memories: int = 0,
) -> str:
    """Build the full memory injection block.

    Always includes the base memory prompt when total_memories > 0.
    Appends specific memory entries if search results are provided.
    """
    if total_memories <= 0 and not results:
        return ""

    parts = [MEMORY_BASE_PROMPT]

    if results:
        lines = _format_memory_lines(results, config)
        if lines:
            parts.append(
                f"\n\n=== YOUR MEMORY ({total_memories} total stored) ===\n"
                + "\n".join(lines)
                + "\n=== END MEMORY ==="
            )
    elif total_memories > 0:
        parts.append(
            f"\n\nYou have {total_memories} stored memories from past conversations. "
            "No specific memories matched the current query closely enough to show, "
            "but you DO have persistent memory and can recall things from past conversations "
            "if the user asks about something you discussed before."
        )

    return "".join(parts)


def _format_memory_lines(
    results: List[Dict[str, Any]],
    config: ProxyConfig,
) -> List[str]:
    """Format search results into individual memory lines."""
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

    return lines


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


def inject_context_into_system(
    system_prompt: str,
    context_block: str,
) -> str:
    """Inject memory context into a system prompt string (for /api/generate).

    If system_prompt already has content, appends the context block.
    Otherwise returns the context block as the system prompt.
    """
    if not context_block:
        return system_prompt
    if system_prompt:
        return system_prompt + "\n\n---\n" + context_block
    return context_block


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
