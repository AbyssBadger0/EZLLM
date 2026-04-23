from pathlib import Path


def default_log_dir() -> str:
    return str(Path.home() / ".ezllm" / "logs")


def default_state_dir() -> str:
    return str(Path.home() / ".ezllm" / "state")
