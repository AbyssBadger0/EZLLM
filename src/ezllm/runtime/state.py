import json
from pathlib import Path

from pydantic import BaseModel


class RuntimeState(BaseModel):
    proxy_pid: int | None = None
    llama_pid: int | None = None
    proxy_port: int
    llama_port: int
    status: str


def save_runtime_state(state_dir: Path, state: RuntimeState) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "runtime.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")


def load_runtime_state(state_dir: Path) -> RuntimeState | None:
    path = state_dir / "runtime.json"
    if not path.exists():
        return None
    return RuntimeState.model_validate(json.loads(path.read_text(encoding="utf-8")))
