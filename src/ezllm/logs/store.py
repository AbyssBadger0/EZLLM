import json
import time
from pathlib import Path
from typing import Any


def history_file_for(log_dir: Path) -> Path:
    return Path(log_dir) / "chat_history.jsonl"


def summarize_data_url(url: str) -> str:
    head, _, _ = url.partition(",")
    mime = "unknown"
    if head.startswith("data:"):
        mime = head[5:].split(";", 1)[0] or "unknown"
    return f"<data-url mime={mime} chars={len(url)}>"


def sanitize_media_value_for_log(value: Any):
    if isinstance(value, list):
        return [sanitize_media_value_for_log(item) for item in value]
    if isinstance(value, str):
        if value.startswith("data:"):
            return summarize_data_url(value)
        if value.startswith("file://"):
            return f"<file-url {value[7:]}>"
    return sanitize_payload_for_log(value)


def sanitize_payload_for_log(value: Any):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key in {"url", "image", "video"}:
                sanitized[key] = sanitize_media_value_for_log(item)
            else:
                sanitized[key] = sanitize_payload_for_log(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_payload_for_log(item) for item in value]
    if isinstance(value, str) and value.startswith("data:"):
        return summarize_data_url(value)
    return value


def append_history_entry(history_file: Path, entry: dict[str, Any]) -> Path:
    history_path = Path(history_file)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return history_path


def read_history_entries(history_file: Path) -> list[dict[str, Any]]:
    history_path = Path(history_file)
    if not history_path.exists():
        return []

    entries = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def save_raw_log(
    *,
    log_dir: Path,
    req_j,
    reasoning,
    content,
    duration,
    path,
    upstream,
    timestamp: str | None = None,
) -> dict[str, Any]:
    entry = {
        "timestamp": timestamp or time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(duration, 2),
        "path": path,
        "upstream": upstream,
        "request_raw": sanitize_payload_for_log(req_j),
        "response_raw": sanitize_payload_for_log(
            {
                "reasoning": reasoning,
                "content": content,
            }
        ),
    }
    append_history_entry(history_file_for(Path(log_dir)), entry)
    return entry
