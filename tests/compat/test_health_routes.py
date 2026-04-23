from pathlib import Path

from fastapi.testclient import TestClient

import ezllm.proxy.app as proxy_app
import ezllm.runtime.health as runtime_health
from ezllm.config.models import LlamaConfig, RuntimeConfig, Settings


def _provider_summary() -> dict:
    return {
        "provider": "openai",
        "base_url": "https://api.example.test/v1",
        "api_key_configured": True,
        "upstream_model_name": "gpt-4.1-mini",
    }


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        runtime=RuntimeConfig(
            host="127.0.0.1",
            proxy_port=8890,
            llama_port=8891,
            log_dir=str(tmp_path / "logs"),
            state_dir=str(tmp_path / "state"),
        ),
        llama=LlamaConfig(
            server_bin="llama-server",
            model_path=r"C:\models\legacy-model.gguf",
        ),
    )


def _build_client(tmp_path: Path, provider_summary: dict | None = None) -> TestClient:
    return TestClient(
        proxy_app.build_app(
            log_dir=tmp_path,
            settings=_build_settings(tmp_path),
            provider_summary=provider_summary,
        )
    )


def _expected_runtime_payload(tmp_path: Path) -> dict:
    return {
        "display_model_name": "legacy-model.gguf",
        "proxy": {
            "url": "http://127.0.0.1:8890",
            "host": "127.0.0.1",
            "port": 8890,
            "healthz": "http://127.0.0.1:8890/healthz",
            "logs_page": "http://127.0.0.1:8890/logs",
        },
        "llama": {
            "url": "http://127.0.0.1:8891",
            "port": 8891,
            "binary": "llama-server",
            "model_path": r"C:\models\legacy-model.gguf",
            "model_file": "legacy-model.gguf",
            "ctx_size": 32768,
            "n_predict": 4096,
        },
        "cloud": {
            "provider": "openai",
            "base_url": "https://api.example.test/v1",
            "api_key_configured": True,
            "local_model_name": "legacy-model.gguf",
            "upstream_model_name": "gpt-4.1-mini",
        },
        "logs": {
            "dir": str(tmp_path),
            "history": str(tmp_path / "chat_history.jsonl"),
        },
    }


def test_runtime_config_restores_legacy_runtime_contract(tmp_path):
    client = _build_client(
        tmp_path,
        provider_summary={
            **_provider_summary(),
            "ignored": "not-part-of-legacy-shape",
        },
    )

    response = client.get("/runtime-config")

    assert response.status_code == 200
    payload = response.json()
    expected = _expected_runtime_payload(tmp_path)

    assert set(payload) == set(expected)
    assert payload == expected


def test_healthz_restores_legacy_top_level_field_names(tmp_path):
    client = _build_client(
        tmp_path,
        provider_summary=_provider_summary(),
    )

    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    expected = {
        "proxy": "ok",
        "started_at": None,
        "llama_port": 8891,
        "proxy_port": 8890,
        "display_model_name": "legacy-model.gguf",
        "local_model_name": "legacy-model.gguf",
        "upstream_model_name": "gpt-4.1-mini",
        "runtime": _expected_runtime_payload(tmp_path),
        "pids": {"proxy": None, "llama": None},
        "llama_status": "not-running",
    }

    assert set(payload) == set(expected)
    assert payload["runtime"] == expected["runtime"]
    assert payload == expected


def test_legacy_model_file_name_handles_windows_style_paths_on_any_host():
    helper = getattr(runtime_health, "legacy_model_file_name", None)

    assert callable(helper)
    assert helper(r"C:\models\legacy-model.gguf") == "legacy-model.gguf"
    assert helper("/models/legacy-model.gguf") == "legacy-model.gguf"
