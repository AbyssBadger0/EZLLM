import os
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .defaults import default_config_path, default_log_dir, default_state_dir
from .models import Settings


def _config_path() -> Path:
    env_path = os.environ.get("EZLLM_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return default_config_path()


def _base_payload() -> dict[str, dict]:
    return {
        "runtime": {
            "host": "127.0.0.1",
            "proxy_port": 8888,
            "llama_port": 8889,
            "log_dir": default_log_dir(),
            "state_dir": default_state_dir(),
        },
        "llama": {},
    }


def _load_payload(path: str | Path | None = None) -> dict[str, dict]:
    payload = _base_payload()
    config_path = Path(path).expanduser() if path is not None else _config_path()
    if config_path.exists():
        with config_path.open("rb") as handle:
            file_payload = tomllib.load(handle)
        payload["runtime"] |= file_payload.get("runtime", {})
        payload["llama"] |= file_payload.get("llama", {})

    runtime = payload.setdefault("runtime", {})
    llama = payload.setdefault("llama", {})
    if os.environ.get("EZLLM_PROXY_PORT"):
        runtime["proxy_port"] = int(os.environ["EZLLM_PROXY_PORT"])
    if os.environ.get("EZLLM_LLAMA_PORT"):
        runtime["llama_port"] = int(os.environ["EZLLM_LLAMA_PORT"])
    if os.environ.get("EZLLM_SERVER_BIN"):
        llama["server_bin"] = os.environ["EZLLM_SERVER_BIN"]
    if os.environ.get("EZLLM_MODEL_PATH"):
        llama["model_path"] = os.environ["EZLLM_MODEL_PATH"]
    if os.environ.get("EZLLM_MMPROJ_PATH"):
        llama["mmproj_path"] = os.environ["EZLLM_MMPROJ_PATH"]

    env_overrides: dict[str, tuple[str, type]] = {
        "EZLLM_CTX_SIZE": ("ctx_size", int),
        "EZLLM_N_PREDICT": ("n_predict", int),
        "EZLLM_PARALLEL": ("parallel", int),
        "EZLLM_GPU_LAYERS": ("gpu_layers", int),
        "EZLLM_BATCH_SIZE": ("batch_size", int),
        "EZLLM_FLASH_ATTN": ("flash_attn", str),
        "EZLLM_CACHE_K_TYPE": ("cache_k_type", str),
        "EZLLM_CACHE_V_TYPE": ("cache_v_type", str),
        "EZLLM_TEMP": ("temp", str),
        "EZLLM_TOP_P": ("top_p", str),
        "EZLLM_TOP_K": ("top_k", str),
        "EZLLM_REASONING": ("reasoning", str),
        "EZLLM_REASONING_FORMAT": ("reasoning_format", str),
        "EZLLM_REASONING_BUDGET": ("reasoning_budget", str),
    }
    for env_name, (config_key, caster) in env_overrides.items():
        if os.environ.get(env_name) is not None:
            llama[config_key] = caster(os.environ[env_name])

    return payload


def load_settings(path: str | Path | None = None) -> Settings:
    return Settings.model_validate(_load_payload(path))


def load_runtime_settings(path: str | Path | None = None):
    payload = _load_payload(path)
    runtime = SimpleNamespace(**payload["runtime"])
    llama = SimpleNamespace(**payload.get("llama", {}))
    return SimpleNamespace(runtime=runtime, llama=llama)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _escape_toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    return _escape_toml_string(str(value))


def _render_toml_table(name: str, payload: dict[str, Any]) -> list[str]:
    lines = [f"[{name}]"]
    scalar_items = [(key, value) for key, value in payload.items() if not isinstance(value, dict) and value is not None]
    nested_items = [(key, value) for key, value in payload.items() if isinstance(value, dict)]
    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")
    for key, value in nested_items:
        lines.append("")
        lines.extend(_render_toml_table(f"{name}.{key}", value))
    return lines


def _render_toml(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            if lines:
                lines.append("")
            lines.extend(_render_toml_table(key, value))
        elif value is not None:
            lines.append(f"{key} = {_format_toml_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def update_config_values(updates: dict[str, Any], *, path: str | Path | None = None) -> Path:
    config_path = Path(path).expanduser() if path is not None else _config_path()
    payload: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    _deep_update(payload, updates)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_render_toml(payload), encoding="utf-8")
    return config_path


def parse_config_value(raw_value: str) -> Any:
    try:
        return tomllib.loads(f"value = {raw_value}")["value"]
    except tomllib.TOMLDecodeError:
        return raw_value


def set_config_key(key_path: str, value: Any, *, path: str | Path | None = None) -> Path:
    parts = [part.strip() for part in key_path.split(".") if part.strip()]
    if len(parts) < 2:
        raise ValueError("config key must include a section, for example llama.ctx_size")
    leaf: dict[str, Any] = value
    for part in reversed(parts[1:]):
        leaf = {part: leaf}
    return update_config_values({parts[0]: leaf}, path=path)


def _insert_or_replace_active_value(section_text: str, provider_name: str) -> str:
    active_line = f'active = "{provider_name}"'
    active_pattern = re.compile(r'(?m)^(?P<indent>\s*)active\s*=\s*.*$')
    match = active_pattern.search(section_text)
    if match:
        indent = match.group("indent")
        return active_pattern.sub(f"{indent}{active_line}", section_text, count=1)
    prefix = "" if section_text.endswith("\n") else "\n"
    return f"{section_text}{prefix}{active_line}\n"


def set_active_provider(name: str) -> Path:
    provider_name = name.strip()
    if not provider_name:
        raise ValueError("provider name must not be empty")

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ""

    section_header = "[providers]"
    section_start = text.find(section_header)
    if section_start != -1:
        section_body_start = section_start + len(section_header)
        next_header_match = re.search(r"(?m)^\[[^\]]+\]\s*$", text[section_body_start:])
        section_end = section_body_start + next_header_match.start() if next_header_match else len(text)
        updated_section = _insert_or_replace_active_value(text[section_start:section_end], provider_name)
        updated_text = f"{text[:section_start]}{updated_section}{text[section_end:]}"
    else:
        descendant_match = re.search(r"(?m)^\[providers\.[^\]]+\]\s*$", text)
        block = f'{section_header}\nactive = "{provider_name}"\n'
        if descendant_match:
            prefix = "" if descendant_match.start() == 0 or text[descendant_match.start() - 1] == "\n" else "\n"
            updated_text = f"{text[:descendant_match.start()]}{prefix}{block}\n{text[descendant_match.start():]}"
        elif text.strip():
            suffix = "" if text.endswith("\n") else "\n"
            updated_text = f"{text}{suffix}\n{block}"
        else:
            updated_text = block

    path.write_text(updated_text, encoding="utf-8")
    return path
