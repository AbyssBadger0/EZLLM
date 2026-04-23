from pathlib import Path

from typer.testing import CliRunner

from ezllm.cli import app


def test_service_status_rejects_non_linux(monkeypatch):
    def fake_ensure_linux_systemd():
        raise RuntimeError("linux/systemd only")

    monkeypatch.setattr("ezllm.cli.ensure_linux_systemd", fake_ensure_linux_systemd)

    result = CliRunner().invoke(app, ["service", "status"])

    assert result.exit_code == 1
    output = f"{result.stdout}\n{result.stderr}".lower()
    assert "linux/systemd only" in output


def test_service_status_reports_linux_availability(monkeypatch):
    monkeypatch.setattr("ezllm.cli.ensure_linux_systemd", lambda: None)

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


def test_console_script_aliases_include_lm():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert 'ezllm = "ezllm.cli:app"' in content
    assert 'lm = "ezllm.cli:app"' in content
