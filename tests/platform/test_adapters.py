from types import SimpleNamespace

import ezllm.platform.linux as linux
import ezllm.platform.macos as macos
import ezllm.platform.windows as windows


def _connections(psutil_module):
    return [
        SimpleNamespace(
            pid=101,
            laddr=SimpleNamespace(port=8888),
            status=psutil_module.CONN_LISTEN,
        ),
        SimpleNamespace(
            pid=202,
            laddr=SimpleNamespace(port=8888),
            status="ESTABLISHED",
        ),
        SimpleNamespace(
            pid=303,
            laddr=SimpleNamespace(port=8889),
            status=psutil_module.CONN_LISTEN,
        ),
    ]


def test_platform_adapters_only_report_listening_pids(monkeypatch):
    scenarios = [
        (windows.psutil, windows.WindowsPlatformAdapter()),
        (linux.psutil, linux.LinuxPlatformAdapter()),
        (macos.psutil, macos.MacOSPlatformAdapter()),
    ]

    for psutil_module, adapter in scenarios:
        monkeypatch.setattr(psutil_module, "net_connections", lambda kind: _connections(psutil_module))

        assert adapter.find_listening_pids(8888) == {101}
