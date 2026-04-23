from types import SimpleNamespace

from typer.testing import CliRunner

from ezllm.cli import app


def _write_runtime_only_config(path, *, state_dir):
    path.write_text(
        f"""
        [runtime]
        state_dir = "{state_dir.as_posix()}"
        """.strip(),
        encoding="utf-8",
    )


def _write_full_config(path, *, state_dir):
    path.write_text(
        f"""
        [runtime]
        state_dir = "{state_dir.as_posix()}"

        [llama]
        server_bin = "llama-server"
        model_path = "/models/model.gguf"
        """.strip(),
        encoding="utf-8",
    )


def test_cli_app_runs():
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0


def test_status_reports_not_running_with_runtime_only_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_runtime_only_config(config_path, state_dir=tmp_path / "state")
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "not running" in result.stdout.lower()


def test_provider_use_updates_active_provider_in_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_full_config(config_path, state_dir=tmp_path / "state")
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(app, ["provider", "use", "openrouter"])

    assert result.exit_code == 0
    assert 'active = "openrouter"' in config_path.read_text(encoding="utf-8")
    assert "openrouter" in result.stdout.lower()


def test_doctor_prints_useful_lines_with_runtime_only_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_runtime_only_config(config_path, state_dir=tmp_path / "state")
    monkeypatch.setenv("EZLLM_CONFIG", str(config_path))

    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    stdout = result.stdout.lower()
    assert "config file:" in stdout
    assert "state dir:" in stdout
    assert "status:" in stdout


def test_start_is_wired_to_runtime_manager(monkeypatch, tmp_path):
    called = {}

    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), host="127.0.0.1", proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )

    class FakeManager:
        def __init__(self, incoming_settings):
            called["settings"] = incoming_settings

        def start_background(self, *, force=False):
            called["started"] = True
            called["force"] = force
            return "EZLLM started"

    monkeypatch.setattr("ezllm.cli.load_settings", lambda: settings)
    monkeypatch.setattr("ezllm.cli.RuntimeManager", FakeManager)

    runner = CliRunner()
    result = runner.invoke(app, ["start"])

    assert result.exit_code == 0
    assert called["settings"] is settings
    assert called["started"] is True
    assert called["force"] is False
    assert "started" in result.stdout.lower()


def test_start_accepts_force_flag(monkeypatch, tmp_path):
    called = {}

    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), host="127.0.0.1", proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )

    class FakeManager:
        def __init__(self, incoming_settings):
            called["settings"] = incoming_settings

        def start_background(self, *, force=False):
            called["force"] = force
            return "EZLLM starting"

    monkeypatch.setattr("ezllm.cli.load_settings", lambda: settings)
    monkeypatch.setattr("ezllm.cli.RuntimeManager", FakeManager)

    runner = CliRunner()
    result = runner.invoke(app, ["start", "--force"])

    assert result.exit_code == 0
    assert called["settings"] is settings
    assert called["force"] is True


def test_restart_accepts_force_flag(monkeypatch, tmp_path):
    called = []

    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), host="127.0.0.1", proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )

    class FakeManager:
        def __init__(self, incoming_settings):
            called.append(("init", incoming_settings))

        def ensure_startable(self, *, force=False):
            called.append(("ensure_startable", force))

        def stop(self):
            called.append(("stop", None))
            return "EZLLM stopped"

        def start_background(self, *, force=False):
            called.append(("start_background", force))
            return "EZLLM starting"

    monkeypatch.setattr("ezllm.cli.load_settings", lambda: settings)
    monkeypatch.setattr("ezllm.cli.RuntimeManager", FakeManager)

    runner = CliRunner()
    result = runner.invoke(app, ["restart", "--force"])

    assert result.exit_code == 0
    assert called == [
        ("init", settings),
        ("ensure_startable", True),
        ("stop", None),
        ("start_background", True),
    ]


def test_restart_does_not_stop_healthy_instance_when_preflight_fails(monkeypatch, tmp_path):
    called = []

    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), host="127.0.0.1", proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )

    class FakeManager:
        def __init__(self, incoming_settings):
            called.append(("init", incoming_settings))

        def ensure_startable(self, *, force=False):
            called.append(("ensure_startable", force))
            raise RuntimeError("port is busy")

        def stop(self):
            called.append(("stop", None))
            return "EZLLM stopped"

        def start_background(self, *, force=False):
            called.append(("start_background", force))
            return "EZLLM starting"

    monkeypatch.setattr("ezllm.cli.load_settings", lambda: settings)
    monkeypatch.setattr("ezllm.cli.RuntimeManager", FakeManager)

    runner = CliRunner()
    result = runner.invoke(app, ["restart"])

    assert result.exit_code == 1
    assert called == [
        ("init", settings),
        ("ensure_startable", False),
    ]
