import os
import platform
import subprocess
from pathlib import Path

import psutil

LINUX_SYSTEMD_ONLY_MESSAGE = "EZLLM service commands are linux/systemd only."
SYSTEMD_RUNTIME_DIR = Path("/run/systemd/system")
SYSTEMD_SYSTEM_DIR = Path("/etc/systemd/system")
DEFAULT_SERVICE_NAME = "ezllm.service"


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


def normalize_service_name(name: str) -> str:
    service_name = (name or "").strip()
    if not service_name:
        raise ValueError("service name must not be empty")
    if "/" in service_name or "\\" in service_name:
        raise ValueError("service name must not contain path separators")
    if not service_name.endswith(".service"):
        service_name = f"{service_name}.service"
    return service_name


def render_service_unit(
    python_executable: str,
    config_path: str,
    *,
    user: str | None = None,
    group: str | None = None,
    working_directory: str | None = None,
) -> str:
    identity_lines = []
    if user:
        identity_lines.append(f"User={user}")
    if group:
        identity_lines.append(f"Group={group}")
    if working_directory:
        identity_lines.append(f"WorkingDirectory={working_directory}")
    lines = [
        "[Unit]",
        "Description=EZLLM local runtime",
        "After=network.target",
        "",
        "[Service]",
        "Type=simple",
        f"Environment={_quote_systemd('PYTHONUNBUFFERED=1')}",
        f"Environment={_quote_systemd(f'EZLLM_CONFIG={config_path}')}",
        *identity_lines,
        f"ExecStart={_quote_systemd(python_executable)} -m ezllm.cli run",
        "Restart=always",
        "RestartSec=5",
        "LimitNOFILE=65536",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    return "\n".join(lines) + "\n"


def _use_sudo(use_sudo: bool) -> bool:
    return use_sudo and hasattr(os, "geteuid") and os.geteuid() != 0


def _with_sudo(command: list[str], *, use_sudo: bool) -> list[str]:
    if _use_sudo(use_sudo):
        return ["sudo", *command]
    return command


def _run(
    command: list[str],
    *,
    runner=subprocess.run,
    check: bool = True,
    input_text: str | None = None,
):
    kwargs = {"check": check, "capture_output": True, "text": True}
    if input_text is not None:
        kwargs["input"] = input_text
    return runner(command, **kwargs)


def systemctl_service(
    action: str,
    name: str = DEFAULT_SERVICE_NAME,
    *,
    runner=subprocess.run,
    check: bool = True,
    use_sudo: bool = True,
):
    service_name = normalize_service_name(name)
    command = _with_sudo(["systemctl", action, service_name], use_sudo=use_sudo)
    return _run(command, runner=runner, check=check)


def systemctl_daemon_reload(*, runner=subprocess.run, use_sudo: bool = True):
    command = _with_sudo(["systemctl", "daemon-reload"], use_sudo=use_sudo)
    return _run(command, runner=runner)


def install_systemd_service(
    *,
    name: str = DEFAULT_SERVICE_NAME,
    python_executable: str,
    config_path: str,
    user: str | None = None,
    group: str | None = None,
    working_directory: str | None = None,
    systemd_dir: str | Path = SYSTEMD_SYSTEM_DIR,
    runner=subprocess.run,
    use_sudo: bool = True,
    enable: bool = False,
    start: bool = False,
) -> Path:
    service_name = normalize_service_name(name)
    unit = render_service_unit(
        python_executable,
        config_path,
        user=user,
        group=group,
        working_directory=working_directory,
    )
    unit_dir = Path(systemd_dir)
    unit_path = unit_dir / service_name

    if unit_dir == SYSTEMD_SYSTEM_DIR and _use_sudo(use_sudo):
        command = ["sudo", "install", "-m", "0644", "/dev/stdin", str(unit_path)]
        _run(command, runner=runner, input_text=unit)
    else:
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(unit, encoding="utf-8")

    systemctl_daemon_reload(runner=runner, use_sudo=use_sudo)
    if enable:
        systemctl_service("enable", service_name, runner=runner, use_sudo=use_sudo)
    if start:
        systemctl_service("restart", service_name, runner=runner, use_sudo=use_sudo)
    return unit_path
