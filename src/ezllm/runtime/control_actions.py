import os
import sys
import time
from pathlib import Path

import psutil

from ezllm.config.loader import load_runtime_settings, load_settings
from ezllm.runtime.process import spawn_background
from ezllm.runtime.state import load_runtime_state


def _terminate_tree(pid: int | None) -> None:
    if not pid or pid == os.getpid():
        return
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        processes = [*process.children(recursive=True), process]
    except psutil.NoSuchProcess:
        return
    alive = []
    for item in processes:
        if item.pid == os.getpid():
            continue
        try:
            item.terminate()
            alive.append(item)
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(alive, timeout=3)
    for item in alive:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(alive, timeout=3)


def _terminate_process_only(pid: int | None) -> None:
    if not pid or pid == os.getpid():
        return
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=3)
    except psutil.NoSuchProcess:
        return
    except psutil.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=3)
        except psutil.NoSuchProcess:
            return


def _clear_runtime_state(state_dir: str | Path) -> None:
    for path in (Path(state_dir) / "runtime.json", Path(state_dir) / "runtime.json.tmp"):
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def run_scheduled_action(action: str, *, delay_seconds: float = 0.75) -> None:
    time.sleep(delay_seconds)
    runtime_settings = load_runtime_settings()
    state = load_runtime_state(Path(runtime_settings.runtime.state_dir))
    if state is not None:
        _terminate_tree(state.llama_pid)
        _terminate_process_only(state.proxy_pid)
        _clear_runtime_state(runtime_settings.runtime.state_dir)
    if action == "restart":
        from ezllm.runtime.manager import RuntimeManager

        RuntimeManager(load_settings()).start_background(force=True)


def schedule_control_action(action: str) -> None:
    if action not in {"restart", "stop"}:
        raise ValueError(f"unsupported control action: {action}")
    spawn_background([sys.executable, "-m", "ezllm.runtime.control_actions", action])


class ScheduledControlActions:
    def restart(self) -> None:
        schedule_control_action("restart")

    def stop(self) -> None:
        schedule_control_action("stop")


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    run_scheduled_action(action)


if __name__ == "__main__":
    main()
