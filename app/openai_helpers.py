from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI


MessageContent = Union[str, List[Dict[str, Any]]]
MessageDict = Dict[str, MessageContent]
MessageLike = Union[str, MessageDict]


def _format_messages(
    messages: Sequence[MessageLike],
) -> List[Dict[str, MessageContent]]:
    formatted: List[Dict[str, MessageContent]] = []
    for msg in messages:
        if isinstance(msg, str):
            formatted.append({"role": "user", "content": msg})
            continue
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if not role or content is None:
            continue
        if isinstance(content, (list, tuple)):
            formatted.append({"role": role, "content": list(content)})
        else:
            formatted.append({"role": role, "content": str(content)})
    return formatted


def call_responses(
    client: OpenAI,
    model: str,
    messages: Sequence[MessageLike],
    *,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> Any:
    payload: Dict[str, Any] = {
        "model": model,
        "input": _format_messages(messages),
        "temperature": temperature,
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning_effort and model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": reasoning_effort}
    return client.responses.create(**payload)


def response_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text:
        return text.strip()

    data: Dict[str, Any] = {}
    try:
        data = resp.model_dump()  # type: ignore[attr-defined]
    except Exception:
        try:
            data = resp.to_dict()  # type: ignore[attr-defined]
        except Exception:
            pass

    outputs = data.get("output") if isinstance(data, dict) else None
    if not isinstance(outputs, list):
        return ""

    chunks: List[str] = []
    for item in outputs:
        content_items = item.get("content") if isinstance(item, dict) else None
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                value = content.get("text")
                if isinstance(value, dict):
                    chunks.append(str(value.get("value", "")))
                elif isinstance(value, str):
                    chunks.append(value)
                else:
                    chunks.append(str(value or ""))
            elif "text" in content:
                value = content.get("text")
                if isinstance(value, dict):
                    chunks.append(str(value.get("value", "")))
                else:
                    chunks.append(str(value))
    return "".join(chunks).strip()
