import os
import sys
from pathlib import Path

import uvicorn

from ezllm.platform.linux import LinuxPlatformAdapter
from ezllm.platform.macos import MacOSPlatformAdapter
from ezllm.platform.windows import WindowsPlatformAdapter
from ezllm.proxy.app import build_app
from ezllm.runtime.process import spawn_background
from ezllm.runtime.state import RuntimeState, save_runtime_state
from ezllm.runtime.state import load_runtime_state


def _default_platform_adapter():
    if sys.platform == "win32":
        return WindowsPlatformAdapter()
    if sys.platform == "darwin":
        return MacOSPlatformAdapter()
    return LinuxPlatformAdapter()


class RuntimeManager:
    def __init__(self, settings, platform_adapter=None):
        self.settings = settings
        self.platform_adapter = platform_adapter or _default_platform_adapter()
        self.state_dir = Path(settings.runtime.state_dir)

    def format_status(self) -> str:
        state = load_runtime_state(self.state_dir)
        if state is None:
            return "EZLLM not running"
        if state.status != "running":
            return (
                f"EZLLM {state.status} on proxy:{state.proxy_port} llama:{state.llama_port} "
                f"(proxy pid={state.proxy_pid}, llama pid={state.llama_pid})"
            )
        return (
            f"EZLLM running on proxy:{state.proxy_port} llama:{state.llama_port} "
            f"(proxy pid={state.proxy_pid}, llama pid={state.llama_pid})"
        )

    def run_foreground(self) -> None:
        app = build_app(log_dir=Path(self.settings.runtime.log_dir), settings=self.settings)
        save_runtime_state(
            self.state_dir,
            RuntimeState(
                proxy_pid=os.getpid(),
                llama_pid=None,
                proxy_port=self.settings.runtime.proxy_port,
                llama_port=self.settings.runtime.llama_port,
                status="running",
            ),
        )
        try:
            uvicorn.run(
                app,
                host=self.settings.runtime.host,
                port=self.settings.runtime.proxy_port,
            )
        finally:
            self._clear_state()

    def start_background(self) -> str:
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "import uvicorn; "
                "from ezllm.proxy.app import build_app; "
                f"uvicorn.run(build_app(log_dir=Path(r'{self.settings.runtime.log_dir}')), "
                f"host={self.settings.runtime.host!r}, port={self.settings.runtime.proxy_port!r})"
            ),
        ]
        process = spawn_background(command)
        save_runtime_state(
            self.state_dir,
            RuntimeState(
                proxy_pid=process.pid,
                llama_pid=None,
                proxy_port=self.settings.runtime.proxy_port,
                llama_port=self.settings.runtime.llama_port,
                status="running",
            ),
        )
        return self.format_status()

    def stop(self) -> str:
        state = load_runtime_state(self.state_dir)
        if state is None:
            return "EZLLM not running"

        pids = {
            pid
            for pid in (
                state.proxy_pid,
                state.llama_pid,
                *self.platform_adapter.find_listening_pids(state.proxy_port),
                *self.platform_adapter.find_listening_pids(state.llama_port),
            )
            if pid
        }
        for pid in pids:
            self.platform_adapter.terminate_tree(pid)

        self._clear_state()
        return "EZLLM stopped"

    def doctor_lines(self, *, config_path: str | Path | None = None) -> list[str]:
        runtime = self.settings.runtime
        llama = getattr(self.settings, "llama", None)
        lines = []
        if config_path is not None:
            lines.append(f"Config file: {Path(config_path)}")
        lines.extend(
            [
                f"State dir: {self.state_dir}",
                f"Log dir: {getattr(runtime, 'log_dir', '<not configured>')}",
                f"Proxy port: {runtime.proxy_port}",
                f"Llama port: {runtime.llama_port}",
                f"Status: {self.format_status()}",
                f"Llama binary: {getattr(llama, 'server_bin', '<not configured>')}",
                f"Model path: {getattr(llama, 'model_path', '<not configured>')}",
            ]
        )
        return lines

    def _clear_state(self) -> None:
        for path in (self.state_dir / "runtime.json", self.state_dir / "runtime.json.tmp"):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
