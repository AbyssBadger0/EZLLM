import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from ezllm.cli import app
from ezllm.platform.linux import render_service_unit


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
    monkeypatch.setattr("ezllm.platform.linux.platform.system", lambda: "Linux")
    monkeypatch.setattr("ezllm.platform.linux.SYSTEMD_RUNTIME_DIR", runtime_dir)

    result = CliRunner().invoke(app, ["service", "status"])

    assert result.exit_code == 0
    stdout = result.stdout.lower()
    assert "linux/systemd" in stdout
    assert "available" in stdout


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
