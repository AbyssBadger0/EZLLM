from pathlib import Path

from ezllm.config.loader import load_settings


def test_load_settings_merges_file_and_env_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [runtime]
        proxy_port = 8888
        llama_port = 8889

        [llama]
        server_bin = "llama-server"
        model_path = "/models/model.gguf"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))
    monkeypatch.setenv("EZLLM_PROXY_PORT", "9999")

    settings = load_settings()

    assert settings.runtime.proxy_port == 9999
    assert settings.runtime.llama_port == 8889
    assert settings.llama.server_bin == "llama-server"
    assert settings.llama.model_path == "/models/model.gguf"


def test_load_settings_requires_model_and_binary(tmp_path, monkeypatch):
    config_path = tmp_path / "broken.toml"
    config_path.write_text("[runtime]\nproxy_port = 8888\n", encoding="utf-8")
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))

    try:
        load_settings()
    except Exception as exc:
        assert "server_bin" in str(exc)
        assert "model_path" in str(exc)
    else:
        raise AssertionError("load_settings() should reject incomplete llama config")
