import platform
from pathlib import Path
from textwrap import dedent

import psutil

LINUX_SYSTEMD_ONLY_MESSAGE = "EZLLM service commands are linux/systemd only."
SYSTEMD_RUNTIME_DIR = Path("/run/systemd/system")


class LinuxPlatformAdapter:
    def find_listening_pids(self, port: int) -> set[int]:
        return {
            conn.pid
            for conn in psutil.net_connections(kind="inet")
            if conn.laddr and conn.laddr.port == port and conn.pid and conn.status == psutil.CONN_LISTEN
        }

    def terminate_tree(self, pid: int, *, force: bool = False) -> None:
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

        try:
            processes = [*proc.children(recursive=True), proc]
        except psutil.NoSuchProcess:
            return
        alive = []

        for process in processes:
            try:
                process.kill() if force else process.terminate()
                alive.append(process)
            except psutil.NoSuchProcess:
                continue

        gone, alive = psutil.wait_procs(alive, timeout=3)
        del gone

        if force:
            return

        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                continue

        psutil.wait_procs(alive, timeout=3)


def ensure_linux_systemd() -> None:
    if platform.system().lower() != "linux" or not SYSTEMD_RUNTIME_DIR.is_dir():
        raise RuntimeError(LINUX_SYSTEMD_ONLY_MESSAGE)


def _quote_systemd(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_service_unit(python_executable: str, config_path: str) -> str:
    return dedent(
        f"""\
        [Unit]
        Description=EZLLM local runtime
        After=network.target

        [Service]
        Type=simple
        Environment={_quote_systemd(f"EZLLM_CONFIG={config_path}")}
        ExecStart={_quote_systemd(python_executable)} -m ezllm.cli run
        Restart=on-failure

        [Install]
        WantedBy=multi-user.target
        """
    )
