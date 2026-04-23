import json
from pathlib import Path

from pydantic import BaseModel, ValidationError


class RuntimeState(BaseModel):
    proxy_pid: int | None = None
    llama_pid: int | None = None
    proxy_port: int
    llama_port: int
    status: str


def save_runtime_state(state_dir: Path, state: RuntimeState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "runtime.json"
    temp_path = state_dir / "runtime.json.tmp"
    temp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    temp_path.replace(path)


def load_runtime_state(state_dir: Path) -> RuntimeState | None:
    path = state_dir / "runtime.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RuntimeState.model_validate(data)
    except (json.JSONDecodeError, ValidationError, OSError):
        return None
