import os
import subprocess
import sys
from collections.abc import Sequence


def spawn_background(
    command: Sequence[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    env_vars = {**os.environ, **(env or {})}
    background_log = env_vars.get("EZLLM_BACKGROUND_LOG")
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    log_handle = None
    if background_log:
        log_handle = open(background_log, "ab")
        stdout = log_handle
        stderr = subprocess.STDOUT
    popen_kwargs = {
        "cwd": cwd,
        "env": env_vars,
        "stdout": stdout,
        "stderr": stderr,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(list(command), **popen_kwargs)
    finally:
        if log_handle is not None:
            log_handle.close()
