import psutil


class MacOSPlatformAdapter:
    def find_listening_pids(self, port: int) -> set[int]:
        return {
            conn.pid
            for conn in psutil.net_connections(kind="inet")
            if conn.laddr and conn.laddr.port == port and conn.pid
        }

    def terminate_tree(self, pid: int, *, force: bool = False) -> None:
        proc = psutil.Process(pid)
        for child in proc.children(recursive=True):
            child.kill() if force else child.terminate()
        proc.kill() if force else proc.terminate()
