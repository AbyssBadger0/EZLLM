import json
import math
from pathlib import Path


def read_log_entries(history_file: Path) -> list[dict]:
    if not history_file.exists():
        return []
    return [
        json.loads(line)
        for line in history_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def paginate_entries(entries: list[dict], *, page: int, size: int) -> dict:
    total = len(entries)
    pages = max(1, math.ceil(total / size)) if size else 1
    start = (page - 1) * size
    end = start + size
    return {
        "page": page,
        "size": size,
        "total": total,
        "pages": pages,
        "entries": entries[start:end],
    }
