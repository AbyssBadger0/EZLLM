from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ezllm.logs.store import history_file_for
from ezllm.runtime.state import load_runtime_state


def legacy_model_file_name(model_path: str | Path) -> str:
    normalized = str(model_path).rstrip("\\/")
    if not normalized:
        return ""

    candidates = []
    for candidate in (PureWindowsPath(normalized).name, PurePosixPath(normalized).name):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        if "/" not in candidate and "\\" not in candidate:
            return candidate
    return candidates[0] if candidates else normalized


def build_cloud_summary(
    settings,
    provider_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    local_model_name = legacy_model_file_name(settings.llama.model_path)
    summary = {
        "provider": "",
        "base_url": "",
        "api_key_configured": False,
        "local_model_name": local_model_name,
        "upstream_model_name": "",
    }
    if provider_summary:
        for key in summary:
            if key in provider_summary:
                summary[key] = provider_summary[key]

    if not summary["local_model_name"]:
        summary["local_model_name"] = local_model_name
    return summary


def _service_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def build_runtime_config_payload(
    settings,
    provider_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proxy_url = _service_url(settings.runtime.host, settings.runtime.proxy_port)
    llama_url = _service_url(settings.runtime.host, settings.runtime.llama_port)
    cloud_summary = build_cloud_summary(settings, provider_summary)
    history_path = history_file_for(Path(settings.runtime.log_dir))

    return {
        "display_model_name": cloud_summary["local_model_name"],
        "proxy": {
            "url": proxy_url,
            "host": settings.runtime.host,
            "port": settings.runtime.proxy_port,
            "healthz": f"{proxy_url}/healthz",
            "logs_page": f"{proxy_url}/logs",
        },
        "llama": {
            "url": llama_url,
            "port": settings.runtime.llama_port,
            "binary": settings.llama.server_bin,
            "model_path": settings.llama.model_path,
            "model_file": legacy_model_file_name(settings.llama.model_path),
            "ctx_size": settings.llama.ctx_size,
            "n_predict": settings.llama.n_predict,
        },
        "cloud": cloud_summary,
        "logs": {
            "dir": settings.runtime.log_dir,
            "history": str(history_path),
        },
    }


def load_runtime_summary(settings) -> dict[str, Any]:
    state = load_runtime_state(Path(settings.runtime.state_dir))
    if state is None:
        return {
            "proxy_pid": None,
            "llama_pid": None,
            "llama_status": "not-running",
            "started_at": None,
        }
    return {
        "proxy_pid": state.proxy_pid,
        "llama_pid": state.llama_pid,
        "llama_status": state.status,
        "started_at": getattr(state, "started_at", None),
    }


def build_health_payload(
    settings,
    runtime_config: Mapping[str, Any],
    *,
    proxy_pid: int | None,
    llama_pid: int | None,
    llama_status: str,
    started_at: str | None,
) -> dict[str, Any]:
    cloud_summary = dict(runtime_config.get("cloud", {}))
    display_model_name = runtime_config["display_model_name"]

    return {
        "proxy": "ok",
        "started_at": started_at,
        "llama_port": settings.runtime.llama_port,
        "proxy_port": settings.runtime.proxy_port,
        "display_model_name": display_model_name,
        "local_model_name": cloud_summary.get("local_model_name", display_model_name),
        "upstream_model_name": cloud_summary.get("upstream_model_name", ""),
        "runtime": dict(runtime_config),
        "pids": {
            "proxy": proxy_pid,
            "llama": llama_pid,
        },
        "llama_status": llama_status,
    }
