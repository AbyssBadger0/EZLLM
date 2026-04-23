from pathlib import Path


def history_file_for(log_dir: Path) -> Path:
    return Path(log_dir) / "chat_history.jsonl"
