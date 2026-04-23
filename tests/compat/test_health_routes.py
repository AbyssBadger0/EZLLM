from pathlib import Path

from fastapi.testclient import TestClient

import ezllm.proxy.app as proxy_app
from ezllm.config.models import LlamaConfig, RuntimeConfig, Settings


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


def _build_client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(proxy_app, "load_settings", lambda: _build_settings(tmp_path), raising=False)
    return TestClient(proxy_app.build_app(log_dir=tmp_path))


def test_runtime_config_returns_legacy_named_sections(tmp_path, monkeypatch):
    client = _build_client(tmp_path, monkeypatch)

    response = client.get("/runtime-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["display_model_name"] == "legacy-model.gguf"
    assert {"display_model_name", "proxy", "llama", "cloud", "logs"} <= payload.keys()


def test_healthz_returns_legacy_health_fields(tmp_path, monkeypatch):
    client = _build_client(tmp_path, monkeypatch)

    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["proxy"] == "ok"
    assert payload["runtime"]["display_model_name"] == "legacy-model.gguf"
    assert payload["pids"] == {"proxy": None, "llama": None}
    assert payload["proxy_port"] == 8890
