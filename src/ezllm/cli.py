import getpass
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

import typer

from ezllm.config.loader import (
    _config_path,
    load_runtime_settings,
    load_settings,
    parse_config_value,
    set_active_provider,
    set_config_key,
)
from ezllm.models.downloader import download_model_artifact
from ezllm.platform.linux import (
    DEFAULT_SERVICE_NAME,
    ensure_linux_systemd,
    install_systemd_service,
    systemctl_service,
)
from ezllm.runtime.manager import RuntimeManager

app = typer.Typer()
provider_app = typer.Typer()
service_app = typer.Typer()
models_app = typer.Typer()
config_app = typer.Typer()

app.add_typer(provider_app, name="provider")
app.add_typer(service_app, name="service")
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")


@app.callback()
def main() -> None:
    """EZLLM command line interface."""


@app.command()
def run() -> None:
    """Run EZLLM in the foreground."""
    RuntimeManager(load_settings()).run_foreground()


@app.command()
def start(
    force: bool = typer.Option(False, "--force", help="Replace conflicting listeners on EZLLM ports."),
    open_browser: bool = typer.Option(False, "--open", help="Open the browser control page after starting."),
) -> None:
    """Start EZLLM in the background."""
    settings = load_settings()
    try:
        typer.echo(RuntimeManager(settings).start_background(force=force))
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    if open_browser:
        webbrowser.open(_control_url(settings))


@app.command()
def stop() -> None:
    """Stop EZLLM."""
    typer.echo(RuntimeManager(load_runtime_settings()).stop())


@app.command()
def restart(
    force: bool = typer.Option(False, "--force", help="Replace conflicting listeners on EZLLM ports.")
) -> None:
    """Restart EZLLM."""
    manager = RuntimeManager(load_settings())
    try:
        manager.ensure_startable(force=force)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    typer.echo(manager.stop())
    try:
        typer.echo(manager.start_background(force=force))
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Print runtime status."""
    typer.echo(RuntimeManager(load_runtime_settings()).format_status())


@app.command("open")
def open_control() -> None:
    """Open the browser control page."""
    settings = load_runtime_settings()
    url = _control_url(settings)
    webbrowser.open(url)
    typer.echo(url)


@app.command()
def doctor() -> None:
    """Print runtime diagnostics."""
    manager = RuntimeManager(load_runtime_settings())
    for line in manager.doctor_lines(config_path=_config_path()):
        typer.echo(line)


def _control_url(settings) -> str:
    return f"http://{settings.runtime.host}:{settings.runtime.proxy_port}/control"


def _default_service_user() -> str:
    return os.environ.get("SUDO_USER") or getpass.getuser()


def _echo_completed_process(result) -> None:
    output = (getattr(result, "stdout", "") or getattr(result, "stderr", "") or "").rstrip()
    if output:
        typer.echo(output)


def _exit_from_process_error(exc: subprocess.CalledProcessError) -> None:
    output = (exc.stderr or exc.stdout or str(exc)).rstrip()
    if output:
        typer.echo(output, err=True)
    raise typer.Exit(code=exc.returncode or 1)


@provider_app.command("use")
def provider_use(name: str) -> None:
    """Set the active provider."""
    path = set_active_provider(name)
    typer.echo(f'Active provider set to "{name}" in {path}')


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a config value, for example llama.ctx_size 200000."""
    path = set_config_key(key, parse_config_value(value))
    typer.echo(f"Updated {key} in {path}")


@config_app.command("show")
def config_show() -> None:
    """Print the active config file."""
    path = _config_path()
    if not path.exists():
        typer.echo(f"Config file does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    typer.echo(path.read_text(encoding="utf-8"))


@service_app.command("status")
def service_status(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Print Linux systemd service status."""
    try:
        ensure_linux_systemd()
        result = systemctl_service("status", name, check=False)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as exc:
        _exit_from_process_error(exc)
    _echo_completed_process(result)


@service_app.command("install")
def service_install(
    name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name."),
    python_executable: str = typer.Option(sys.executable, "--python", help="Python executable for the service."),
    config_path: str | None = typer.Option(None, "--config", help="EZLLM config file path."),
    user: str | None = typer.Option(None, "--user", help="User account that runs the service."),
    group: str | None = typer.Option(None, "--group", help="Optional group account that runs the service."),
    working_directory: str | None = typer.Option(None, "--working-directory", help="Service working directory."),
    enable: bool = typer.Option(False, "--enable", help="Enable the service at boot after installing."),
    start_service: bool = typer.Option(False, "--start", help="Restart the service after installing."),
) -> None:
    """Install or replace the Linux systemd service unit."""
    try:
        ensure_linux_systemd()
        unit_path = install_systemd_service(
            name=name,
            python_executable=python_executable,
            config_path=str(Path(config_path).expanduser() if config_path else _config_path()),
            user=user if user is not None else _default_service_user(),
            group=group,
            working_directory=str(Path(working_directory).expanduser() if working_directory else Path.cwd()),
            enable=enable,
            start=start_service,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as exc:
        _exit_from_process_error(exc)
    typer.echo(f"Installed {name} at {unit_path}")


def _service_action(action: str, name: str, *, check: bool = True) -> None:
    try:
        ensure_linux_systemd()
        result = systemctl_service(action, name, check=check)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except subprocess.CalledProcessError as exc:
        _exit_from_process_error(exc)
    _echo_completed_process(result)


@service_app.command("start")
def service_start(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Start the Linux systemd service."""
    _service_action("start", name)


@service_app.command("stop")
def service_stop(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Stop the Linux systemd service."""
    _service_action("stop", name)


@service_app.command("restart")
def service_restart(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Restart the Linux systemd service."""
    _service_action("restart", name)


@service_app.command("enable")
def service_enable(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Enable the Linux systemd service at boot."""
    _service_action("enable", name)


@service_app.command("disable")
def service_disable(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Disable the Linux systemd service at boot."""
    _service_action("disable", name)


@service_app.command("is-enabled")
def service_is_enabled(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Print whether the Linux systemd service is enabled."""
    _service_action("is-enabled", name, check=False)


@service_app.command("log")
def service_log(name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name", help="systemd service unit name.")) -> None:
    """Follow Linux systemd journal logs for the service."""
    try:
        ensure_linux_systemd()
        subprocess.run(["journalctl", "-u", name, "-f"], check=False)
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


@models_app.command("download")
def models_download(
    repo_id: str,
    filename: str,
    revision: str | None = typer.Option(None, "--revision", help="Optional repository revision."),
    local_dir: str | None = typer.Option(None, "--local-dir", help="Optional local directory for the download."),
    repo_type: str = typer.Option("model", "--repo-type", help="Hugging Face repository type."),
) -> None:
    """Download a model artifact from Hugging Face Hub."""
    typer.echo(
        download_model_artifact(
            repo_id,
            filename,
            revision=revision,
            local_dir=local_dir,
            repo_type=repo_type,
        )
    )


if __name__ == "__main__":
    app()
