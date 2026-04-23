import psutil


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

        processes = [*proc.children(recursive=True), proc]
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
