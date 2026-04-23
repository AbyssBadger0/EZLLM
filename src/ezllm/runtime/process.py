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
    popen_kwargs = {
        "cwd": cwd,
        "env": {**os.environ, **(env or {})},
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        popen_kwargs["start_new_session"] = True
    return subprocess.Popen(list(command), **popen_kwargs)
