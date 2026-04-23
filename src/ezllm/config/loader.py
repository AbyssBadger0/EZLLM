import os
import tomllib
from pathlib import Path

from .defaults import default_config_path, default_log_dir, default_state_dir
from .models import Settings


def _config_path() -> Path:
    env_path = os.environ.get("EZLLM_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return default_config_path()


def load_settings() -> Settings:
    payload = {
        "runtime": {
            "host": "127.0.0.1",
            "proxy_port": 8888,
            "llama_port": 8889,
            "log_dir": default_log_dir(),
            "state_dir": default_state_dir(),
        },
        "llama": {},
    }
    path = _config_path()
    if path.exists():
        with path.open("rb") as handle:
            file_payload = tomllib.load(handle)
        payload["runtime"] |= file_payload.get("runtime", {})
        payload["llama"] |= file_payload.get("llama", {})

    runtime = payload.setdefault("runtime", {})
    llama = payload.setdefault("llama", {})
    if os.environ.get("EZLLM_PROXY_PORT"):
        runtime["proxy_port"] = int(os.environ["EZLLM_PROXY_PORT"])
    if os.environ.get("EZLLM_LLAMA_PORT"):
        runtime["llama_port"] = int(os.environ["EZLLM_LLAMA_PORT"])
    if os.environ.get("EZLLM_SERVER_BIN"):
        llama["server_bin"] = os.environ["EZLLM_SERVER_BIN"]
    if os.environ.get("EZLLM_MODEL_PATH"):
        llama["model_path"] = os.environ["EZLLM_MODEL_PATH"]

    return Settings.model_validate(payload)
