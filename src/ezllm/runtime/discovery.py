from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal


LLAMA_SERVER_NAMES = {"llama-server", "llama-server.exe"}


def scan_model_dirs(raw_dirs: Iterable[str], *, limit: int = 300) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source_dir in _existing_dirs(raw_dirs):
        for model_path in sorted(source_dir.rglob("*.gguf"), key=lambda item: str(item).lower()):
            if len(models) >= limit:
                return models
            resolved_model = _safe_resolve(model_path)
            if resolved_model in seen or _is_mmproj(model_path):
                continue
            seen.add(resolved_model)
            models.append(
                {
                    "name": model_path.name,
                    "path": str(model_path),
                    "directory": str(model_path.parent),
                    "source_dir": str(source_dir),
                    "size_bytes": model_path.stat().st_size,
                    "mmproj_path": _find_mmproj(model_path.parent),
                }
            )
    return models


def scan_llama_binaries(raw_dirs: Iterable[str], *, limit: int = 100) -> list[dict[str, Any]]:
    binaries: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source_dir in _existing_dirs(raw_dirs):
        for binary_path in sorted(source_dir.rglob("*"), key=lambda item: str(item).lower()):
            if len(binaries) >= limit:
                return binaries
            if not binary_path.is_file() or binary_path.name.lower() not in LLAMA_SERVER_NAMES:
                continue
            resolved_binary = _safe_resolve(binary_path)
            if resolved_binary in seen:
                continue
            seen.add(resolved_binary)
            binaries.append(
                {
                    "name": binary_path.name,
                    "path": str(binary_path),
                    "directory": str(binary_path.parent),
                    "source_dir": str(source_dir),
                    "size_bytes": binary_path.stat().st_size,
                }
            )
    return binaries


def browse_directory(
    raw_path: str | None,
    *,
    file_filter: Literal["all", "models", "llama", "directories"] = "all",
) -> dict[str, Any]:
    path = Path(raw_path).expanduser() if raw_path else Path.home()
    if path.is_file():
        path = path.parent
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(str(path))

    entries = [_browser_entry(child) for child in path.iterdir() if _include_browser_entry(child, file_filter)]
    entries.sort(key=lambda item: (item["type"] != "directory", item["name"].lower()))
    parent = path.parent if path.parent != path else None
    return {
        "path": str(path),
        "parent": str(parent) if parent else None,
        "entries": entries,
    }


def _existing_dirs(raw_dirs: Iterable[str]) -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for raw_dir in raw_dirs:
        if not raw_dir:
            continue
        path = Path(str(raw_dir)).expanduser()
        if not path.exists() or not path.is_dir():
            continue
        resolved = _safe_resolve(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        dirs.append(path)
    return dirs


def _find_mmproj(directory: Path) -> str | None:
    for candidate in sorted(directory.glob("*.gguf"), key=lambda item: item.name.lower()):
        if _is_mmproj(candidate):
            return str(candidate)
    return None


def _is_mmproj(path: Path) -> bool:
    return "mmproj" in path.name.lower()


def _include_browser_entry(path: Path, file_filter: str) -> bool:
    if path.is_dir():
        return True
    if file_filter == "directories":
        return False
    if file_filter == "models":
        return path.suffix.lower() == ".gguf"
    if file_filter == "llama":
        return path.name.lower() in LLAMA_SERVER_NAMES
    return True


def _browser_entry(path: Path) -> dict[str, Any]:
    is_dir = path.is_dir()
    payload: dict[str, Any] = {
        "name": path.name,
        "path": str(path),
        "type": "directory" if is_dir else "file",
    }
    if not is_dir:
        payload["size_bytes"] = path.stat().st_size
    return payload


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()
