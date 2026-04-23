from collections.abc import Mapping
from typing import Any


DEFAULT_BASE_URL = "https://openrouter.ai/api"
KNOWN_NAMES = {"openrouter", "or"}


def normalize_model_id(model: Any) -> str:
    normalized = str(model or "").strip()
    if normalized.startswith("openrouter/"):
        return normalized[len("openrouter/") :]
    return normalized


def normalize_provider_config(config: Any, *, name: str = "openrouter") -> dict[str, Any]:
    raw = config if isinstance(config, Mapping) else {}
    models = raw.get("models") if isinstance(raw.get("models"), Mapping) else {}
    base_url = str(raw.get("base_url") or raw.get("api_base") or DEFAULT_BASE_URL).rstrip("/")
    return {
        "name": name,
        "label": "OpenRouter",
        "base_url": base_url,
        "models": {str(family).strip().lower(): normalize_model_id(model) for family, model in models.items() if model},
    }
