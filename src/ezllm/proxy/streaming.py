from ezllm.proxy.response_normalizer import parse_anthropic_payload_for_log, parse_openai_payload_for_log


def append_payload_text(payload: dict, *, upstream_kind: str, reasoning_parts: list[str], content_parts: list[str]) -> None:
    if upstream_kind == "anthropic":
        parse_anthropic_payload_for_log(payload, reasoning_parts, content_parts)
        return
    parse_openai_payload_for_log(payload, reasoning_parts, content_parts)
