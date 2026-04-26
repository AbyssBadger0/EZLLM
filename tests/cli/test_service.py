import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from ezllm.cli import app
from ezllm.platform.linux import install_systemd_service, render_service_unit


def test_service_status_rejects_non_linux(monkeypatch, tmp_path):
    monkeypatch.setattr("ezllm.platform.linux.platform.system", lambda: "Darwin")
    monkeypatch.setattr("ezllm.platform.linux.SYSTEMD_RUNTIME_DIR", tmp_path / "systemd")

    result = CliRunner().invoke(app, ["service", "status"])

    assert result.exit_code == 1
    output = f"{result.stdout}\n{result.stderr}".lower()
    assert "linux/systemd only" in output


def test_service_status_rejects_linux_without_systemd(monkeypatch, tmp_path):
    monkeypatch.setattr("ezllm.platform.linux.platform.system", lambda: "Linux")
    monkeypatch.setattr("ezllm.platform.linux.SYSTEMD_RUNTIME_DIR", tmp_path / "systemd")

    result = CliRunner().invoke(app, ["service", "status"])

    assert result.exit_code == 1
    output = f"{result.stdout}\n{result.stderr}".lower()
    assert "linux/systemd only" in output


def test_service_status_reports_linux_availability(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "systemd"
    runtime_dir.mkdir()
    calls = []

    def fake_systemctl_service(action, name, **kwargs):
        calls.append((action, name, kwargs))
        return subprocess.CompletedProcess(["systemctl", action, name], 0, stdout="active\n", stderr="")

    monkeypatch.setattr("ezllm.platform.linux.platform.system", lambda: "Linux")
    monkeypatch.setattr("ezllm.platform.linux.SYSTEMD_RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr("ezllm.cli.systemctl_service", fake_systemctl_service)

    result = CliRunner().invoke(app, ["service", "status"])

    assert result.exit_code == 0
    assert "active" in result.stdout.lower()
    assert calls == [("status", "ezllm.service", {"check": False})]


def test_models_download_calls_downloader_and_prints_saved_path(monkeypatch, tmp_path):
    called = {}
    saved_path = tmp_path / "model.gguf"

    def fake_download_model_artifact(repo_id, filename, *, revision=None, local_dir=None, repo_type="model"):
        called["repo_id"] = repo_id
        called["filename"] = filename
        called["revision"] = revision
        called["local_dir"] = local_dir
        called["repo_type"] = repo_type
        return str(saved_path)

    monkeypatch.setattr("ezllm.cli.download_model_artifact", fake_download_model_artifact)

    result = CliRunner().invoke(
        app,
        [
            "models",
            "download",
            "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "Qwen2.5-7B-Instruct.Q4_K_M.gguf",
            "--revision",
            "main",
            "--local-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert called == {
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct.Q4_K_M.gguf",
        "revision": "main",
        "local_dir": str(tmp_path),
        "repo_type": "model",
    }
    assert str(saved_path) in result.stdout


def test_render_service_unit_contains_expected_systemd_fields():
    unit = render_service_unit("/usr/bin/python3", "/etc/ezllm/config.toml")

    assert 'ExecStart="/usr/bin/python3" -m ezllm.cli run' in unit
    assert 'Environment="EZLLM_CONFIG=/etc/ezllm/config.toml"' in unit
    assert "WantedBy=multi-user.target" in unit


def test_render_service_unit_quotes_paths_with_spaces():
    unit = render_service_unit("/opt/EZ LLM/bin/python3", "/etc/EZ LLM/config file.toml")

    assert 'ExecStart="/opt/EZ LLM/bin/python3" -m ezllm.cli run' in unit
    assert 'Environment="EZLLM_CONFIG=/etc/EZ LLM/config file.toml"' in unit


def test_render_service_unit_can_run_as_user_group_from_working_directory():
    unit = render_service_unit(
        "/home/abyss/miniconda3/bin/python3",
        "/home/abyss/.config/ezllm/config.toml",
        user="abyss",
        group="abyss",
        working_directory="/home/abyss/EZLLM",
    )

    assert "User=abyss" in unit
    assert "Group=abyss" in unit
    assert "WorkingDirectory=/home/abyss/EZLLM" in unit
    assert "Restart=always" in unit
    assert "RestartSec=5" in unit
    assert "LimitNOFILE=65536" in unit


def test_install_systemd_service_writes_unit_and_reloads_systemd(tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    unit_path = install_systemd_service(
        name="lm-server.service",
        python_executable="/home/abyss/miniconda3/bin/python3",
        config_path="/home/abyss/.config/ezllm/config.toml",
        user="abyss",
        group="abyss",
        working_directory="/home/abyss/EZLLM",
        systemd_dir=tmp_path,
        runner=fake_run,
        use_sudo=False,
    )

    assert unit_path == tmp_path / "lm-server.service"
    unit = unit_path.read_text(encoding="utf-8")
    assert 'ExecStart="/home/abyss/miniconda3/bin/python3" -m ezllm.cli run' in unit
    assert 'Environment="EZLLM_CONFIG=/home/abyss/.config/ezllm/config.toml"' in unit
    assert "User=abyss" in unit
    assert calls == [
        (
            ["systemctl", "daemon-reload"],
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_service_restart_invokes_systemctl_for_named_service(monkeypatch):
    calls = []

    def fake_systemctl_service(action, name, **kwargs):
        calls.append((action, name, kwargs))
        return subprocess.CompletedProcess(["systemctl", action, name], 0, stdout="restarted\n", stderr="")

    monkeypatch.setattr("ezllm.cli.ensure_linux_systemd", lambda: None)
    monkeypatch.setattr("ezllm.cli.systemctl_service", fake_systemctl_service)

    result = CliRunner().invoke(app, ["service", "restart", "--name", "lm-server.service"])

    assert result.exit_code == 0
    assert "restarted" in result.stdout
    assert calls == [("restart", "lm-server.service", {"check": True})]


def test_service_install_rejects_non_linux(monkeypatch, tmp_path):
    monkeypatch.setattr("ezllm.platform.linux.platform.system", lambda: "Windows")
    monkeypatch.setattr("ezllm.platform.linux.SYSTEMD_RUNTIME_DIR", tmp_path / "systemd")

    result = CliRunner().invoke(app, ["service", "install"])

    assert result.exit_code == 1
    output = f"{result.stdout}\n{result.stderr}".lower()
    assert "linux/systemd only" in output


def test_python_module_entrypoint_prints_help():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    pythonpath_entries = [str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    result = subprocess.run(
        [sys.executable, "-m", "ezllm.cli", "--help"],
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    assert "ezllm command line interface" in result.stdout.lower()


def test_console_script_aliases_include_lm():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert 'ezllm = "ezllm.cli:app"' in content
    assert 'lm = "ezllm.cli:app"' in content
