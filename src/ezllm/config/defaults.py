import os
import sys
from pathlib import Path


def _home_dir() -> Path:
    return Path.home()


def _config_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "EZLLM"
    if sys.platform == "darwin":
        return _home_dir() / "Library" / "Application Support" / "EZLLM"
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "ezllm"
    return _home_dir() / ".config" / "ezllm"


def _data_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "EZLLM"
    if sys.platform == "darwin":
        return _home_dir() / "Library" / "Application Support" / "EZLLM"
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "ezllm"
    return _home_dir() / ".local" / "share" / "ezllm"


def default_config_path() -> Path:
    return _config_root() / "config.toml"


def default_log_dir() -> str:
    return str(_data_root() / "logs")


def default_state_dir() -> str:
    return str(_data_root() / "state")
