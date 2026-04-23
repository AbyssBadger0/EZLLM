import typer

from ezllm.config.loader import _config_path, load_runtime_settings, load_settings, set_active_provider
from ezllm.runtime.manager import RuntimeManager

app = typer.Typer()
provider_app = typer.Typer()

app.add_typer(provider_app, name="provider")


@app.callback()
def main() -> None:
    """EZLLM command line interface."""


@app.command()
def run() -> None:
    """Run EZLLM in the foreground."""
    RuntimeManager(load_settings()).run_foreground()


@app.command()
def start(force: bool = typer.Option(False, "--force", help="Replace conflicting listeners on EZLLM ports.")) -> None:
    """Start EZLLM in the background."""
    try:
        typer.echo(RuntimeManager(load_settings()).start_background(force=force))
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)


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


@app.command()
def doctor() -> None:
    """Print runtime diagnostics."""
    manager = RuntimeManager(load_runtime_settings())
    for line in manager.doctor_lines(config_path=_config_path()):
        typer.echo(line)


@provider_app.command("use")
def provider_use(name: str) -> None:
    """Set the active provider."""
    path = set_active_provider(name)
    typer.echo(f'Active provider set to "{name}" in {path}')
