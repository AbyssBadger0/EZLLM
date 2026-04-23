from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ezllm.runtime.state import load_runtime_state


def build_runtime_config_payload(
    settings,
    provider_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "display_model_name": Path(settings.llama.model_path).name,
        "proxy": {
            "host": settings.runtime.host,
            "port": settings.runtime.proxy_port,
        },
        "llama": {
            "port": settings.runtime.llama_port,
            "server_bin": settings.llama.server_bin,
            "model_path": settings.llama.model_path,
        },
        "cloud": dict(provider_summary or {}),
        "logs": {
            "dir": settings.runtime.log_dir,
        },
    }


def load_runtime_summary(settings) -> dict[str, Any]:
    state = load_runtime_state(Path(settings.runtime.state_dir))
    if state is None:
        return {
            "proxy_pid": None,
            "llama_pid": None,
            "llama_status": "not-running",
        }
    return {
        "proxy_pid": state.proxy_pid,
        "llama_pid": state.llama_pid,
        "llama_status": state.status,
    }


def build_health_payload(
    settings,
    runtime_config: Mapping[str, Any],
    *,
    proxy_pid: int | None,
    llama_pid: int | None,
    llama_status: str,
) -> dict[str, Any]:
    return {
        "proxy": "ok",
        "llama": llama_status,
        "runtime": dict(runtime_config),
        "pids": {
            "proxy": proxy_pid,
            "llama": llama_pid,
        },
        "proxy_port": settings.runtime.proxy_port,
    }
