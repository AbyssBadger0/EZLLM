import platform

import psutil

LINUX_SYSTEMD_ONLY_MESSAGE = "EZLLM service commands are linux/systemd only."


class LinuxPlatformAdapter:
    def find_listening_pids(self, port: int) -> set[int]:
        return {
            conn.pid
            for conn in psutil.net_connections(kind="inet")
            if conn.laddr and conn.laddr.port == port and conn.pid
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
    if platform.system().lower() != "linux":
        raise RuntimeError(LINUX_SYSTEMD_ONLY_MESSAGE)
