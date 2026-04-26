from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ezllm.config.loader import load_settings
from ezllm.config.models import LlamaConfig, RuntimeConfig, Settings
from ezllm.proxy.app import build_app


def _settings(tmp_path: Path) -> Settings:
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
            mmproj_path=r"C:\models\mmproj.gguf",
        ),
    )


def test_control_page_and_config_api_expose_full_llama_parameters(tmp_path):
    client = TestClient(
        build_app(
            log_dir=tmp_path,
            settings=_settings(tmp_path),
            config_path=tmp_path / "config.toml",
        )
    )

    page = client.get("/control")
    payload = client.get("/api/control/config").json()

    assert page.status_code == 200
    assert "EZLLM Control" in page.text
    assert payload["runtime"]["proxy_port"] == 8890
    assert payload["llama"]["server_bin"] == "llama-server"
    assert payload["llama"]["model_path"] == r"C:\models\legacy-model.gguf"
    assert payload["llama"]["mmproj_path"] == r"C:\models\mmproj.gguf"
    assert payload["llama"]["ctx_size"] == 200000
    assert payload["llama"]["n_predict"] == 81920
    assert payload["llama"]["cache_k_type"] == "q8_0"
    assert payload["llama"]["cache_v_type"] == "q8_0"
    assert payload["llama"]["reasoning"] == "auto"
    assert payload["llama"]["reasoning_format"] == "deepseek"
    assert payload["llama"]["reasoning_budget"] == "-1"


def test_control_config_update_writes_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [runtime]
        state_dir = "state"

        [llama]
        server_bin = "llama-server"
        model_path = "/models/model.gguf"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))
    settings = load_settings()
    client = TestClient(build_app(log_dir=tmp_path, settings=settings, config_path=config_path))

    response = client.put(
        "/api/control/config",
        json={
            "runtime": {"proxy_port": 9000, "llama_port": 9001},
            "llama": {
                "ctx_size": 65536,
                "n_predict": 16384,
                "reasoning": "off",
                "flash_attn": "off",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["restart_required"] is True
    text = config_path.read_text(encoding="utf-8")
    assert "proxy_port = 9000" in text
    assert "llama_port = 9001" in text
    assert "ctx_size = 65536" in text
    assert 'reasoning = "off"' in text


def test_control_restart_and_stop_schedule_runtime_actions(tmp_path):
    actions = []
    controller = SimpleNamespace(
        restart=lambda: actions.append("restart"),
        stop=lambda: actions.append("stop"),
    )
    client = TestClient(
        build_app(
            log_dir=tmp_path,
            settings=_settings(tmp_path),
            config_path=tmp_path / "config.toml",
            control_actions=controller,
        )
    )

    restart = client.post("/api/control/restart")
    stop = client.post("/api/control/stop")

    assert restart.status_code == 202
    assert stop.status_code == 202
    assert actions == ["restart", "stop"]
