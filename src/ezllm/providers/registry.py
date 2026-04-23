from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ezllm.providers.openrouter import KNOWN_NAMES as OPENROUTER_NAMES
from ezllm.providers.openrouter import normalize_provider_config as normalize_openrouter_config
from ezllm.providers.sub2 import KNOWN_NAMES as SUB2_NAMES
from ezllm.providers.sub2 import normalize_provider_config as normalize_sub2_config


def _get_value(source: Any, key: str, default=None):
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _as_mapping(source: Any) -> dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, dict):
        return dict(source)
    data = getattr(source, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    return {}


def _coerce_names(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [part.strip() for part in raw_value.split(",") if part.strip()]
    if isinstance(raw_value, (list, tuple, set)):
        names = []
        for item in raw_value:
            text = str(item).strip()
            if text:
                names.append(text)
        return names
    text = str(raw_value).strip()
    return [text] if text else []


def _dedupe_preserving_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered = []
    for value in values:
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(value)
    return tuple(ordered)


def _normalize_provider(name: str, config: Any) -> "ProviderConfig":
    normalized_name = (name or "").strip()
    config_mapping = _as_mapping(config)
    lowered = normalized_name.lower()

    if lowered in OPENROUTER_NAMES:
        payload = normalize_openrouter_config(config_mapping, name=normalized_name)
    elif lowered in SUB2_NAMES:
        payload = normalize_sub2_config(config_mapping, name=normalized_name)
    else:
        models = config_mapping.get("models") if isinstance(config_mapping.get("models"), dict) else {}
        payload = {
            "name": normalized_name,
            "label": str(config_mapping.get("label") or normalized_name),
            "base_url": str(config_mapping.get("base_url") or config_mapping.get("api_base") or "").rstrip("/"),
            "models": {str(family).strip().lower(): str(model).strip() for family, model in models.items() if model},
        }

    return ProviderConfig(
        name=payload["name"],
        label=payload["label"],
        base_url=payload["base_url"],
        models=payload["models"],
    )


def _iter_provider_configs(providers: Any) -> dict[str, Any]:
    configs = _as_mapping(providers)
    configs.pop("active", None)
    return configs


def _build_local_aliases(settings: Any, canonical_local_model: str) -> tuple[str, ...]:
    aliases = []
    if canonical_local_model:
        aliases.append(canonical_local_model)

    llama = _get_value(settings, "llama")
    model_path = _get_value(llama, "model_path", "")
    if model_path:
        aliases.append(Path(model_path).name)

    alias_settings = _get_value(settings, "aliases")
    aliases.extend(_coerce_names(_get_value(alias_settings, "local", [])))
    return _dedupe_preserving_order(aliases)


def _build_cloud_alias_map(settings: Any, providers: dict[str, "ProviderConfig"]) -> dict[str, str]:
    alias_settings = _get_value(settings, "aliases")
    cloud_aliases = _get_value(alias_settings, "cloud", {})
    native_models = _get_value(alias_settings, "native", None)
    if native_models is None:
        native_models = _get_value(alias_settings, "native_models", {})
    alias_map: dict[str, str] = {}

    if isinstance(cloud_aliases, dict):
        for key, value in cloud_aliases.items():
            family = str(value).strip()
            key_text = str(key).strip()
            if not key_text:
                continue

            if isinstance(value, (list, tuple, set)):
                family = key_text.lower()
                for alias in _coerce_names(value):
                    alias_map[alias.lower()] = family
                continue

            if key_text.lower() in {"sonnet", "opus", "haiku"} and family:
                alias_map[family.lower()] = key_text.lower()
                continue

            if family:
                alias_map[key_text.lower()] = family.lower()

    if isinstance(native_models, dict):
        for family, value in native_models.items():
            family_name = str(family).strip().lower()
            if not family_name:
                continue
            for native_model in _coerce_names(value):
                alias_map[native_model.lower()] = family_name

    for provider in providers.values():
        for family, model_name in provider.models.items():
            if model_name:
                alias_map[model_name.lower()] = family

    return alias_map


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    label: str
    base_url: str
    models: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderRegistry:
    local_model_name: str
    local_aliases: tuple[str, ...]
    cloud_alias_to_family: dict[str, str] = field(default_factory=dict)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    active_provider_name: str | None = None

    @property
    def active_provider(self) -> ProviderConfig | None:
        if not self.active_provider_name:
            return None
        return self.providers.get(self.active_provider_name)

    def is_local_alias(self, model: str) -> bool:
        normalized = (model or "").strip().lower()
        if not normalized:
            return False
        return normalized in {alias.lower() for alias in self.local_aliases}

    def cloud_family_for(self, model: str) -> str | None:
        normalized = (model or "").strip().lower()
        if not normalized:
            return None
        return self.cloud_alias_to_family.get(normalized)

    def model_for_family(self, family: str) -> str | None:
        normalized = (family or "").strip().lower()
        if not normalized:
            return None
        provider = self.active_provider
        if provider is None:
            return None
        return provider.models.get(normalized)


def _resolve_active_provider_name(active_provider_name: str | None, providers: dict[str, ProviderConfig]) -> str | None:
    openrouter_provider_name = next((name for name in providers if name.lower() in OPENROUTER_NAMES), None)

    if not active_provider_name:
        return openrouter_provider_name
    if active_provider_name in providers:
        return active_provider_name

    lowered = active_provider_name.lower()
    for name in providers:
        if name.lower() == lowered:
            return name

    if lowered in OPENROUTER_NAMES:
        for name in providers:
            if name.lower() in OPENROUTER_NAMES:
                return name

    if lowered in SUB2_NAMES:
        for name in providers:
            if name.lower() in SUB2_NAMES:
                return name

    return openrouter_provider_name


def build_provider_registry(settings: Any) -> ProviderRegistry:
    providers_config = _get_value(settings, "providers")
    active_provider_name = _get_value(providers_config, "active")
    providers = {
        name: _normalize_provider(name, config)
        for name, config in _iter_provider_configs(providers_config).items()
    }

    proxy = _get_value(settings, "proxy")
    local_model_name = str(_get_value(proxy, "local_model_name", "") or "").strip()
    if not local_model_name:
        llama = _get_value(settings, "llama")
        model_path = str(_get_value(llama, "model_path", "") or "").strip()
        local_model_name = Path(model_path).name if model_path else ""

    return ProviderRegistry(
        local_model_name=local_model_name,
        local_aliases=_build_local_aliases(settings, local_model_name),
        cloud_alias_to_family=_build_cloud_alias_map(settings, providers),
        providers=providers,
        active_provider_name=_resolve_active_provider_name(active_provider_name, providers),
    )
