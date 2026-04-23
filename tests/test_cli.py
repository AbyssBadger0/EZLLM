from typer.testing import CliRunner

from ezllm.cli import app


def test_cli_app_runs():
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
