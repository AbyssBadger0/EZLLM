from collections.abc import Mapping
from typing import Any


KNOWN_NAMES = {"cc", "sub2", "sub2api"}


def normalize_provider_config(config: Any, *, name: str = "sub2") -> dict[str, Any]:
    raw = config if isinstance(config, Mapping) else {}
    models = raw.get("models") if isinstance(raw.get("models"), Mapping) else {}
    base_url = str(raw.get("base_url") or raw.get("api_base") or "").rstrip("/")
    return {
        "name": name,
        "label": "sub2api",
        "base_url": base_url,
        "models": {str(family).strip().lower(): str(model).strip() for family, model in models.items() if model},
    }
