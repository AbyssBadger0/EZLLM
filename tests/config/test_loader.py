from pathlib import Path

from ezllm.config import defaults
from ezllm.config.loader import _config_path, load_settings


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


def test_config_defaults_use_windows_appdata(monkeypatch):
    monkeypatch.delenv("EZLLM_CONFIG", raising=False)
    monkeypatch.setattr(defaults.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", r"C:\Users\tester\AppData\Roaming")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

    assert _config_path() == Path(r"C:\Users\tester\AppData\Roaming\EZLLM\config.toml")
    assert Path(defaults.default_log_dir()) == Path(r"C:\Users\tester\AppData\Local\EZLLM\logs")
    assert Path(defaults.default_state_dir()) == Path(r"C:\Users\tester\AppData\Local\EZLLM\state")


def test_config_defaults_use_macos_application_support(monkeypatch):
    monkeypatch.delenv("EZLLM_CONFIG", raising=False)
    monkeypatch.setattr(defaults.sys, "platform", "darwin")
    monkeypatch.setattr(defaults.Path, "home", classmethod(lambda cls: Path("/Users/tester")))

    assert _config_path() == Path("/Users/tester/Library/Application Support/EZLLM/config.toml")
    assert Path(defaults.default_log_dir()) == Path("/Users/tester/Library/Application Support/EZLLM/logs")
    assert Path(defaults.default_state_dir()) == Path("/Users/tester/Library/Application Support/EZLLM/state")


def test_config_defaults_use_linux_xdg_locations(monkeypatch):
    monkeypatch.delenv("EZLLM_CONFIG", raising=False)
    monkeypatch.setattr(defaults.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/config-home")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/data-home")

    assert _config_path() == Path("/tmp/config-home/ezllm/config.toml")
    assert Path(defaults.default_log_dir()) == Path("/tmp/data-home/ezllm/logs")
    assert Path(defaults.default_state_dir()) == Path("/tmp/data-home/ezllm/state")
