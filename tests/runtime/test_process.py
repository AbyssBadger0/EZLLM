from ezllm.runtime.process import spawn_background
from ezllm.platform.linux import LinuxPlatformAdapter
import ezllm.platform.linux as linux_platform


def test_spawn_background_passes_merged_env_and_session_flags(monkeypatch):
    captured = {}

    class DummyPopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

    monkeypatch.setattr("ezllm.runtime.process.subprocess.Popen", DummyPopen)
    monkeypatch.setattr("ezllm.runtime.process.sys.platform", "linux")
    monkeypatch.setenv("BASE_ENV", "base")

    spawn_background(["llama-server", "--port", "8889"], cwd="/tmp/ezllm", env={"EXTRA_ENV": "1"})

    assert captured["command"] == ["llama-server", "--port", "8889"]
    assert captured["kwargs"]["cwd"] == "/tmp/ezllm"
    assert captured["kwargs"]["env"]["BASE_ENV"] == "base"
    assert captured["kwargs"]["env"]["EXTRA_ENV"] == "1"
    assert captured["kwargs"]["start_new_session"] is True


def test_linux_terminate_tree_ignores_missing_process(monkeypatch):
    adapter = LinuxPlatformAdapter()

    def raise_missing_process(pid):
        raise linux_platform.psutil.NoSuchProcess(pid)

    monkeypatch.setattr("ezllm.platform.linux.psutil.Process", raise_missing_process)

    adapter.terminate_tree(1234)
