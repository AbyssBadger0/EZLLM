import html
import json
import math
import re
from pathlib import Path


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


def build_display_messages(req_raw: dict) -> list[dict]:
    messages = []
    system = req_raw.get("system")

    if system:
        messages.append({"role": "system", "content": system})

    for msg in req_raw.get("messages", []):
        messages.append(msg)

    return messages


def flatten_text_content(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") in {"text", "thinking"}:
                    parts.append(normalize_text_piece(item.get("text") or item.get("thinking")))
                else:
                    parts.append(normalize_text_piece(item))
            else:
                parts.append(normalize_text_piece(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return normalize_text_piece(value)
    return ""


def detect_request_kind(req_raw: dict, path: str) -> str:
    normalized_path = (path or "").strip().lower()
    if normalized_path.endswith("/count_tokens"):
        return "count-tokens"

    system = req_raw.get("system")
    system_text = flatten_text_content(system).lower()
    if "generate a concise, sentence-case title" in system_text and '"title"' in system_text:
        return "session-title"

    if normalized_path.endswith("/messages") or normalized_path.endswith("/chat/completions"):
        return "chat"

    return "request"


def parse_user_content_html(content_str: str) -> str:
    meta_pattern = r"Sender \(untrusted metadata\):\s*```json\s*(.*?)\s*```"
    timestamp_pattern = r"^\[(.*?)\]"

    meta_match = re.search(meta_pattern, content_str, re.DOTALL)
    if meta_match:
        try:
            meta_json = json.loads(meta_match.group(1))
            username = meta_json.get("username", meta_json.get("name", "Unknown"))
        except Exception:
            username = "Unknown"

        clean_content = re.sub(meta_pattern, "", content_str, flags=re.DOTALL).strip()
        ts_match = re.match(timestamp_pattern, clean_content)

        if ts_match:
            timestamp = ts_match.group(1)
            actual_msg = clean_content[len(ts_match.group(0)) :].strip()
            return (
                '<div style="margin-bottom: 8px;"><span style="background:#21262d; padding:2px 6px; '
                'border-radius:4px; font-size:11px; color:#39c5cf; margin-right:8px; border: 1px solid '
                f'#00aba9;">👤 {html.escape(username)}</span><span style="background:#21262d; padding:2px 6px; '
                'border-radius:4px; font-size:11px; color:#8b949e; border: 1px solid #30363d;">🕒 '
                f'{html.escape(timestamp)}</span></div><div>{html.escape(actual_msg)}</div>'
            )

        return (
            '<div style="margin-bottom: 8px;"><span style="background:#21262d; padding:2px 6px; '
            'border-radius:4px; font-size:11px; color:#39c5cf; border: 1px solid #00aba9;">👤 '
            f'{html.escape(username)}</span></div><div>{html.escape(clean_content)}</div>'
        )

    return html.escape(content_str)


def get_request_model(req_raw: dict | None) -> str:
    if not isinstance(req_raw, dict):
        return ""
    model = req_raw.get("model")
    return model.strip() if isinstance(model, str) else ""


def read_log_entries(history_file: Path) -> tuple[list[dict], int]:
    if not history_file.exists():
        return [], 0
    all_lines = history_file.read_text(encoding="utf-8").splitlines()
    return all_lines, len(all_lines)


def project_log_entry(entry: dict) -> dict:
    req_raw = entry.get("request_raw", {})
    resp_raw = entry.get("response_raw", {})
    messages = build_display_messages(req_raw)
    parsed_messages = []
    for msg in messages:
        role = msg.get("role", "unknown")
        raw_content = extract_content(msg.get("content", ""))
        if role == "user":
            body = parse_user_content_html(raw_content)
        else:
            body = html.escape(raw_content)
        parsed_messages.append({"role": role, "body": body})

    return {
        "timestamp": entry.get("timestamp", ""),
        "duration_sec": entry.get("duration_sec", 0),
        "path": entry.get("path", "-"),
        "upstream": entry.get("upstream", "-"),
        "request_model": get_request_model(req_raw),
        "request_kind": detect_request_kind(req_raw, entry.get("path", "-")),
        "messages": parsed_messages,
        "reasoning": resp_raw.get("reasoning", ""),
        "content": resp_raw.get("content", ""),
        "request_raw": req_raw,
        "response_raw": resp_raw,
    }


def paginate_entries(lines: list[str], *, page: int, size: int) -> dict:
    total = len(lines)
    pages = math.ceil(total / size) if total else 0
    start = (page - 1) * size
    end = start + size
    entries = []
    page_lines = list(reversed(lines))[start:end]

    for line in page_lines:
        try:
            entries.append(project_log_entry(json.loads(line)))
        except Exception:
            pass

    return {
        "page": page,
        "size": size,
        "total": total,
        "pages": pages,
        "entries": entries,
    }
