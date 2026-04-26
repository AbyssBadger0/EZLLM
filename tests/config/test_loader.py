from pathlib import Path, PureWindowsPath

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
    assert settings.llama.ctx_size == 200000
    assert settings.llama.n_predict == 81920
    assert settings.llama.cache_k_type == "q8_0"
    assert settings.llama.cache_v_type == "q8_0"
    assert settings.llama.flash_attn == "on"
    assert settings.llama.reasoning == "auto"
    assert settings.llama.reasoning_format == "deepseek"
    assert settings.llama.reasoning_budget == "-1"
    assert settings.llama.temp == "0.7"
    assert settings.llama.top_p == "0.95"
    assert settings.llama.top_k == "20"


def test_load_settings_accepts_full_llama_parameter_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [runtime]
        proxy_port = 7777
        llama_port = 7778

        [llama]
        server_bin = "llama-server"
        model_path = "/models/model.gguf"
        mmproj_path = "/models/mmproj.gguf"
        ctx_size = 65536
        n_predict = 16384
        parallel = 2
        gpu_layers = 88
        batch_size = 1024
        flash_attn = "off"
        cache_k_type = "f16"
        cache_v_type = "f16"
        temp = "0.2"
        top_p = "0.9"
        top_k = "40"
        reasoning = "off"
        reasoning_format = "none"
        reasoning_budget = "0"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))

    settings = load_settings()

    assert settings.llama.mmproj_path == "/models/mmproj.gguf"
    assert settings.llama.ctx_size == 65536
    assert settings.llama.n_predict == 16384
    assert settings.llama.parallel == 2
    assert settings.llama.gpu_layers == 88
    assert settings.llama.batch_size == 1024
    assert settings.llama.flash_attn == "off"
    assert settings.llama.cache_k_type == "f16"
    assert settings.llama.cache_v_type == "f16"
    assert settings.llama.temp == "0.2"
    assert settings.llama.top_p == "0.9"
    assert settings.llama.top_k == "40"
    assert settings.llama.reasoning == "off"
    assert settings.llama.reasoning_format == "none"
    assert settings.llama.reasoning_budget == "0"


def test_load_settings_reads_extended_llama_env_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [llama]
        server_bin = "llama-server"
        model_path = "/models/model.gguf"
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))
    monkeypatch.setenv("EZLLM_MMPROJ_PATH", "/models/mmproj.gguf")
    monkeypatch.setenv("EZLLM_CTX_SIZE", "12345")
    monkeypatch.setenv("EZLLM_N_PREDICT", "678")
    monkeypatch.setenv("EZLLM_REASONING", "off")

    settings = load_settings()

    assert settings.llama.mmproj_path == "/models/mmproj.gguf"
    assert settings.llama.ctx_size == 12345
    assert settings.llama.n_predict == 678
    assert settings.llama.reasoning == "off"


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

    assert PureWindowsPath(_config_path()) == PureWindowsPath(
        r"C:\Users\tester\AppData\Roaming\EZLLM\config.toml"
    )
    assert PureWindowsPath(defaults.default_log_dir()) == PureWindowsPath(
        r"C:\Users\tester\AppData\Local\EZLLM\logs"
    )
    assert PureWindowsPath(defaults.default_state_dir()) == PureWindowsPath(
        r"C:\Users\tester\AppData\Local\EZLLM\state"
    )


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
