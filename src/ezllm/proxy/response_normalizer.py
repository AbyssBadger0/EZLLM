import json


def normalize_text_piece(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def extract_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                block_type = item.get("type", "unknown")
                if block_type == "text":
                    parts.append(normalize_text_piece(item.get("text")))
                elif block_type == "thinking":
                    parts.append(normalize_text_piece(item.get("thinking") or item.get("text")))
                elif block_type == "tool_use":
                    parts.append(
                        f"[tool_use:{item.get('name', 'unknown')}] "
                        f"{json.dumps(item.get('input', {}), ensure_ascii=False)}"
                    )
                elif block_type == "tool_result":
                    parts.append(f"[tool_result] {extract_content(item.get('content'))}")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def parse_openai_payload_for_log(payload: dict, reasoning_parts: list[str], content_parts: list[str]) -> None:
    choices = payload.get("choices", [])
    if not choices:
        return

    choice = choices[0]
    delta = choice.get("delta")
    if isinstance(delta, dict):
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        if reasoning:
            reasoning_parts.append(normalize_text_piece(reasoning))
        if content:
            content_parts.append(normalize_text_piece(content))
        return

    message = choice.get("message")
    if isinstance(message, dict):
        reasoning = message.get("reasoning_content")
        content = message.get("content")
        if reasoning:
            reasoning_parts.append(normalize_text_piece(reasoning))
        if content:
            content_parts.append(extract_content(content))


def parse_anthropic_block_for_log(block: dict, reasoning_parts: list[str], content_parts: list[str]) -> None:
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        if text:
            content_parts.append(normalize_text_piece(text))
    elif block_type == "thinking":
        thinking = block.get("thinking") or block.get("text")
        if thinking:
            reasoning_parts.append(normalize_text_piece(thinking))
    elif block_type == "tool_use":
        content_parts.append(
            f"[tool_use:{block.get('name', 'unknown')}] {json.dumps(block.get('input', {}), ensure_ascii=False)}"
        )
    elif block_type == "tool_result":
        content_parts.append(f"[tool_result] {extract_content(block.get('content'))}")


def parse_anthropic_payload_for_log(payload: dict, reasoning_parts: list[str], content_parts: list[str]) -> None:
    payload_type = payload.get("type")

    if payload_type == "content_block_delta":
        delta = payload.get("delta", {})
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text")
            if text:
                content_parts.append(normalize_text_piece(text))
        elif delta_type == "thinking_delta":
            thinking = delta.get("thinking") or delta.get("text")
            if thinking:
                reasoning_parts.append(normalize_text_piece(thinking))
        return

    if payload_type == "message_start":
        message = payload.get("message", {})
        for block in message.get("content", []):
            if isinstance(block, dict):
                parse_anthropic_block_for_log(block, reasoning_parts, content_parts)
        return

    if isinstance(payload.get("content"), list):
        for block in payload.get("content", []):
            if isinstance(block, dict):
                parse_anthropic_block_for_log(block, reasoning_parts, content_parts)
