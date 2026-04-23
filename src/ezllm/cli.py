import typer


app = typer.Typer()


@app.callback()
def main() -> None:
    """EZLLM command line interface."""
