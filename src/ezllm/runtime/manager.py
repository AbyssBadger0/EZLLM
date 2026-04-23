import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil
import uvicorn

from ezllm.platform.linux import LinuxPlatformAdapter
from ezllm.platform.macos import MacOSPlatformAdapter
from ezllm.platform.windows import WindowsPlatformAdapter
from ezllm.proxy.app import build_app
from ezllm.runtime.ports import choose_port_conflict_action
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
        pid = os.getpid()
        state = load_runtime_state(self.state_dir)
        listeners_by_port = self._listeners_by_port(state)
        listeners = set().union(*listeners_by_port.values()) if listeners_by_port else set()
        owned_pids = self._active_owned_pids(state, listeners_by_port)
        listeners.discard(pid)
        owned_pids.discard(pid)
        if listeners or owned_pids:
            raise RuntimeError("EZLLM proxy port is already in use.")
        app = build_app(log_dir=Path(self.settings.runtime.log_dir), settings=self.settings)
        save_runtime_state(
            self.state_dir,
            RuntimeState(
                proxy_pid=pid,
                llama_pid=None,
                proxy_port=self.settings.runtime.proxy_port,
                llama_port=self.settings.runtime.llama_port,
                status="running",
                started_at=self._started_at_for_pid(pid),
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

    def start_background(self, *, force: bool = False) -> str:
        action, owned_pids, listeners = self._start_plan(force=force)
        if action == "restart":
            self._terminate_pids(owned_pids, force=False)
            self._clear_state()
        elif action == "force":
            self._terminate_pids(listeners, force=True)
            self._clear_state()

        command = self._background_command()
        process = spawn_background(command)
        save_runtime_state(
            self.state_dir,
            RuntimeState(
                proxy_pid=process.pid,
                llama_pid=None,
                proxy_port=self.settings.runtime.proxy_port,
                llama_port=self.settings.runtime.llama_port,
                status="starting",
                started_at=self._started_at_for_pid(process.pid),
            ),
        )
        return self.format_status()

    def stop(self) -> str:
        state = load_runtime_state(self.state_dir)
        if state is None:
            return "EZLLM not running"

        self._terminate_pids(self._verified_owned_pids(state), force=False)

        self._clear_state()
        return "EZLLM stopped"

    def ensure_startable(self, *, force: bool = False) -> str:
        return self._start_plan(force=force)[0]

    def _start_plan(self, *, force: bool = False) -> tuple[str, set[int], set[int]]:
        state = load_runtime_state(self.state_dir)
        listeners_by_port = self._listeners_by_port(state)
        owned_pids = self._active_owned_pids(state, listeners_by_port)
        conflict_listeners_by_port = {port: set(pids) for port, pids in listeners_by_port.items()}
        if state is not None and state.status == "starting" and state.proxy_pid in owned_pids:
            conflict_listeners_by_port.setdefault(state.proxy_port, set()).add(state.proxy_pid)
        action = choose_port_conflict_action(
            requested_force=force,
            owned_pids=owned_pids,
            listeners_by_port=conflict_listeners_by_port,
        )
        if action == "error":
            raise RuntimeError("EZLLM ports are already in use by another process. Re-run with --force.")
        listeners = set().union(*conflict_listeners_by_port.values()) if conflict_listeners_by_port else set()
        return action, owned_pids, listeners

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

    def _background_command(self) -> list[str]:
        return [
            sys.executable,
            "-c",
            (
                "from ezllm.config.loader import load_settings; "
                "from ezllm.runtime.manager import RuntimeManager; "
                "RuntimeManager(load_settings()).run_foreground()"
            ),
        ]

    def _listeners_by_port(self, state=None) -> dict[int, set[int]]:
        ports = {self.settings.runtime.proxy_port}
        if state is not None:
            ports.add(state.proxy_port)
        return {
            port: self.platform_adapter.find_listening_pids(port)
            for port in ports
        }

    def _owned_pids(self, state) -> set[int]:
        if state is None:
            return set()
        return {pid for pid in (state.proxy_pid,) if pid}

    def _active_owned_pids(self, state, listeners_by_port: dict[int, set[int]]) -> set[int]:
        owned_pids = self._owned_pids(state)
        if not owned_pids:
            return set()
        listeners = set().union(*listeners_by_port.values()) if listeners_by_port else set()
        active = {pid for pid in owned_pids if pid in listeners}
        if state is not None and state.status == "starting" and state.proxy_pid in owned_pids:
            if self._starting_state_process_exists(state):
                active.add(state.proxy_pid)
        return active

    def _verified_owned_pids(self, state) -> set[int]:
        listeners_by_port = self._listeners_by_port(state)
        return self._active_owned_pids(state, listeners_by_port)

    def _starting_state_process_exists(self, state) -> bool:
        process = self._process_for_pid(getattr(state, "proxy_pid", None))
        if process is None:
            return False
        return self._starting_state_matches_process(state, process)

    def _process_for_pid(self, pid: int | None):
        if not pid:
            return None
        try:
            return psutil.Process(pid)
        except psutil.Error:
            return None

    def _starting_state_matches_process(self, state, process) -> bool:
        started_at = getattr(state, "started_at", None)
        if not started_at:
            return self._is_legacy_background_process(process)

        expected_started_at = self._parse_started_at(started_at)
        if expected_started_at is None:
            return False

        try:
            actual_started_at = datetime.fromtimestamp(process.create_time(), tz=timezone.utc)
        except psutil.Error:
            return False
        return abs((actual_started_at - expected_started_at).total_seconds()) < 1.0

    def _is_legacy_background_process(self, process) -> bool:
        try:
            cmdline = process.cmdline()
        except psutil.Error:
            return False
        if not cmdline:
            return False

        joined = " ".join(part.lower() for part in cmdline)
        return (
            ("ezllm.runtime.manager" in joined and "run_foreground()" in joined)
            or ("ezllm.proxy.app" in joined and "uvicorn.run" in joined)
        )

    def _started_at_for_pid(self, pid: int) -> str | None:
        try:
            process = psutil.Process(pid)
        except psutil.Error:
            return None
        started_at = datetime.fromtimestamp(process.create_time(), tz=timezone.utc)
        return started_at.isoformat().replace("+00:00", "Z")

    def _parse_started_at(self, started_at: str) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        return parsed

    def _terminate_pids(self, pids: set[int], *, force: bool) -> None:
        for pid in pids:
            self.platform_adapter.terminate_tree(pid, force=force)
