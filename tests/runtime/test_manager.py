import os
import sys
from types import SimpleNamespace

import psutil
import pytest

from ezllm.runtime.manager import RuntimeManager
from ezllm.runtime.state import RuntimeState, load_runtime_state, save_runtime_state


def _settings(tmp_path):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            state_dir=str(tmp_path),
            log_dir=str(tmp_path / "logs"),
            host="127.0.0.1",
            proxy_port=8888,
            llama_port=8889,
        ),
        llama=SimpleNamespace(
            server_bin="llama-server",
            model_path="model.gguf",
            mmproj_path=None,
            gpu_layers=999,
            ctx_size=200000,
            n_predict=81920,
            cache_k_type="q8_0",
            cache_v_type="q8_0",
            flash_attn="on",
            batch_size=512,
            parallel=1,
            temp="0.7",
            top_p="0.95",
            top_k="20",
            reasoning="auto",
            reasoning_format="deepseek",
            reasoning_budget="-1",
        ),
    )


def test_save_and_load_runtime_state(tmp_path):
    state = RuntimeState(
        proxy_pid=101,
        llama_pid=202,
        proxy_port=8888,
        llama_port=8889,
        status="running",
    )

    save_runtime_state(tmp_path, state)
    loaded = load_runtime_state(tmp_path)

    assert loaded.proxy_pid == 101
    assert loaded.llama_pid == 202
    assert loaded.status == "running"


def test_save_and_load_runtime_state_preserves_started_at(tmp_path):
    state = RuntimeState(
        proxy_pid=101,
        llama_pid=202,
        proxy_port=8888,
        llama_port=8889,
        status="running",
        started_at="2026-04-24T12:34:56Z",
    )

    save_runtime_state(tmp_path, state)
    loaded = load_runtime_state(tmp_path)

    assert loaded.started_at == "2026-04-24T12:34:56Z"


def test_load_runtime_state_returns_none_for_corrupt_json(tmp_path):
    tmp_path.joinpath("runtime.json").write_text("{not valid json", encoding="utf-8")

    assert load_runtime_state(tmp_path) is None


def test_load_runtime_state_returns_none_for_invalid_utf8(tmp_path):
    tmp_path.joinpath("runtime.json").write_bytes(b"\xff\xfe\xfa")

    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_format_status_reports_running_state(tmp_path):
    settings = _settings(tmp_path)
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=8888,
            llama_port=8889,
            status="running",
        ),
    )

    manager = RuntimeManager(settings)

    assert "running on proxy:8888 llama:8889" in manager.format_status().lower()


def test_runtime_manager_format_status_reports_persisted_non_running_state(tmp_path):
    settings = _settings(tmp_path)
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
        ),
    )

    manager = RuntimeManager(settings)

    assert "starting" in manager.format_status().lower()
    assert "running on" not in manager.format_status().lower()


def test_runtime_manager_format_status_reports_not_running(tmp_path):
    settings = _settings(tmp_path)

    manager = RuntimeManager(settings)

    assert "not running" in manager.format_status().lower()


def test_runtime_manager_stop_only_terminates_owned_proxy_pid_from_state(tmp_path):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {101}

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=8888,
            llama_port=8889,
            status="running",
        ),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert set(terminated) == {(101, False)}
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_terminates_owned_proxy_and_llama_pids(tmp_path):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            if port == 8888:
                return {101}
            if port == 8889:
                return {202}
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=8888,
            llama_port=8889,
            status="running",
        ),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert set(terminated) == {(101, False), (202, False)}
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_ignores_stale_state_when_proxy_listener_is_different(tmp_path):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {999}

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=8888,
            llama_port=8889,
            status="running",
        ),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert terminated == []
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_uses_persisted_proxy_port_when_config_drifted(tmp_path):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {101} if port == 7777 else set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    settings = _settings(tmp_path)
    settings.runtime.proxy_port = 8888
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=7777,
            llama_port=8889,
            status="running",
        ),
    )

    manager = RuntimeManager(settings, platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert terminated == [(101, False)]
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_terminates_starting_proxy_pid_without_listener(tmp_path):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=os.getpid(),
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at=manager._started_at_for_pid(os.getpid()),
        ),
    )

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert terminated == [(os.getpid(), False)]
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_does_not_kill_stale_starting_proxy_pid(tmp_path, monkeypatch):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    class MissingProcess:
        def __init__(self, pid):
            raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr("ezllm.runtime.manager.psutil.Process", MissingProcess)
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
        ),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert terminated == []
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_start_background_rejects_foreign_port_conflicts_without_force(
    tmp_path, monkeypatch
):
    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {999} if port == 8888 else set()

        def terminate_tree(self, pid, *, force=False):
            raise AssertionError("foreign pid should not be terminated without --force")

    spawned = []
    monkeypatch.setattr("ezllm.runtime.manager.spawn_background", lambda command, **kwargs: spawned.append(command))

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    with pytest.raises(RuntimeError, match="in use"):
        manager.start_background(force=False)

    assert spawned == []
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_start_background_restarts_owned_port_conflicts(tmp_path, monkeypatch):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {101} if port == 8888 else {202}

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=8888,
            llama_port=8889,
            status="running",
        ),
    )
    monkeypatch.setattr(
        "ezllm.runtime.manager.spawn_background",
        lambda command, **kwargs: SimpleNamespace(pid=303),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.start_background(force=False)
    state = load_runtime_state(tmp_path)

    assert "starting" in result.lower()
    assert set(terminated) == {(101, False), (202, False)}
    assert state is not None
    assert state.proxy_pid == 303
    assert state.status == "starting"


def test_runtime_manager_start_background_treats_starting_proxy_pid_without_listener_as_owned_conflict(
    tmp_path, monkeypatch
):
    terminated = []
    spawned = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=os.getpid(),
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at=manager._started_at_for_pid(os.getpid()),
        ),
    )
    monkeypatch.setattr(
        "ezllm.runtime.manager.spawn_background",
        lambda command, **kwargs: spawned.append(command) or SimpleNamespace(pid=202),
    )

    result = manager.start_background(force=False)
    state = load_runtime_state(tmp_path)

    assert "starting" in result.lower()
    assert terminated == [(os.getpid(), False)]
    assert len(spawned) == 1
    assert state is not None
    assert state.proxy_pid == 202


def test_runtime_manager_start_background_treats_legacy_background_child_without_started_at_as_owned_conflict(
    tmp_path, monkeypatch
):
    terminated = []
    spawned = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    class LegacyBackgroundProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 1234.0

        def cmdline(self):
            return [
                sys.executable,
                "-c",
                "from ezllm.proxy.app import build_app; import uvicorn; uvicorn.run(build_app(log_dir='logs', settings=None))",
            ]

    monkeypatch.setattr("ezllm.runtime.manager.psutil.Process", LegacyBackgroundProcess)
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at=None,
        ),
    )
    monkeypatch.setattr(
        "ezllm.runtime.manager.spawn_background",
        lambda command, **kwargs: spawned.append(command) or SimpleNamespace(pid=202),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.start_background(force=False)
    state = load_runtime_state(tmp_path)

    assert "starting" in result.lower()
    assert terminated == [(101, False)]
    assert len(spawned) == 1
    assert state is not None
    assert state.proxy_pid == 202


def test_runtime_manager_start_background_restart_does_not_kill_foreign_listener_after_churn(
    tmp_path, monkeypatch
):
    terminated = []
    lookup_count = {8888: 0}

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            if port == 8889:
                return set()
            lookup_count[port] = lookup_count.get(port, 0) + 1
            if port == 8888 and lookup_count[port] == 1:
                return {101}
            return {999}

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=8888,
            llama_port=8889,
            status="running",
        ),
    )
    monkeypatch.setattr(
        "ezllm.runtime.manager.spawn_background",
        lambda command, **kwargs: SimpleNamespace(pid=303),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.start_background(force=False)
    state = load_runtime_state(tmp_path)

    assert "starting" in result.lower()
    assert terminated == [(101, False)]
    assert state is not None
    assert state.proxy_pid == 303


def test_runtime_manager_start_background_rejects_foreign_llama_port_listener_without_force(
    tmp_path, monkeypatch
):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set() if port == 8888 else {505}

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    spawned = []
    monkeypatch.setattr("ezllm.runtime.manager.spawn_background", lambda command, **kwargs: spawned.append(command))

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    with pytest.raises(RuntimeError, match="in use"):
        manager.start_background(force=False)

    assert terminated == []
    assert spawned == []
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_start_background_detects_owned_instance_on_persisted_proxy_port_after_config_drift(
    tmp_path, monkeypatch
):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {101} if port == 7777 else set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    settings = _settings(tmp_path)
    settings.runtime.proxy_port = 8888
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=202,
            proxy_port=7777,
            llama_port=8889,
            status="running",
        ),
    )
    monkeypatch.setattr(
        "ezllm.runtime.manager.spawn_background",
        lambda command, **kwargs: SimpleNamespace(pid=303),
    )

    manager = RuntimeManager(settings, platform_adapter=FakePlatformAdapter())

    result = manager.start_background(force=False)
    state = load_runtime_state(tmp_path)

    assert "starting" in result.lower()
    assert terminated == [(101, False)]
    assert state is not None
    assert state.proxy_pid == 303
    assert state.proxy_port == 8888


def test_runtime_manager_run_foreground_refuses_to_clobber_existing_running_state(tmp_path, monkeypatch):
    uvicorn_called = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {101}

        def terminate_tree(self, pid, *, force=False):
            raise AssertionError("run_foreground should not terminate processes")

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="running",
        ),
    )
    monkeypatch.setattr("ezllm.runtime.manager.uvicorn.run", lambda *args, **kwargs: uvicorn_called.append(True))

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    with pytest.raises(RuntimeError, match="already in use"):
        manager.run_foreground()

    state = load_runtime_state(tmp_path)
    assert uvicorn_called == []
    assert state is not None
    assert state.proxy_pid == 101
    assert state.status == "running"


def test_runtime_manager_run_foreground_does_not_self_conflict_with_matching_starting_state(
    tmp_path, monkeypatch
):
    calls = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            raise AssertionError("run_foreground should not terminate processes")

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())
    current_pid = os.getpid()
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=current_pid,
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at=manager._started_at_for_pid(current_pid),
        ),
    )
    monkeypatch.setattr("ezllm.runtime.manager.build_app", lambda **kwargs: object())
    monkeypatch.setattr(
        "ezllm.runtime.manager.start_llama_server",
        lambda settings, log_dir: SimpleNamespace(
            pid=222,
            poll=lambda: None,
            terminate=lambda: None,
            wait=lambda timeout=None: None,
        ),
    )

    def fake_uvicorn_run(*args, **kwargs):
        state = load_runtime_state(tmp_path)
        calls.append((args, kwargs, state.proxy_pid if state else None, state.status if state else None))

    monkeypatch.setattr("ezllm.runtime.manager.uvicorn.run", fake_uvicorn_run)

    manager.run_foreground()

    assert len(calls) == 1
    assert calls[0][2] == current_pid
    assert calls[0][3] == "running"
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 8888
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_run_foreground_allows_starting_wrapper_pid_without_listener(
    tmp_path, monkeypatch
):
    calls = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            raise AssertionError("run_foreground should not terminate processes")

    class WrapperProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 1234.0

        def cmdline(self):
            return [
                sys.executable,
                "-c",
                "from ezllm.config.loader import load_settings; from ezllm.runtime.manager import RuntimeManager; RuntimeManager(load_settings()).run_foreground()",
            ]

    monkeypatch.setattr("ezllm.runtime.manager.psutil.Process", WrapperProcess)
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at=None,
        ),
    )
    monkeypatch.setattr("ezllm.runtime.manager.build_app", lambda **kwargs: object())
    monkeypatch.setattr(
        "ezllm.runtime.manager.start_llama_server",
        lambda settings, log_dir: SimpleNamespace(
            pid=222,
            poll=lambda: None,
            terminate=lambda: None,
            wait=lambda timeout=None: None,
        ),
    )
    monkeypatch.setattr("ezllm.runtime.manager.uvicorn.run", lambda *args, **kwargs: calls.append(kwargs))

    RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter()).run_foreground()

    assert calls == [{"host": "127.0.0.1", "port": 8888}]
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_run_foreground_starts_llama_server_and_persists_pid(tmp_path, monkeypatch):
    calls = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            raise AssertionError("run_foreground should not terminate through the adapter")

    monkeypatch.setattr("ezllm.runtime.manager.build_app", lambda **kwargs: object())
    monkeypatch.setattr(
        "ezllm.runtime.manager.start_llama_server",
        lambda settings, log_dir: calls.append(("start_llama", settings, log_dir))
        or SimpleNamespace(pid=222, poll=lambda: None, terminate=lambda: None, wait=lambda timeout=None: None),
    )

    def fake_uvicorn_run(*args, **kwargs):
        state = load_runtime_state(tmp_path)
        calls.append(("uvicorn", kwargs, state.llama_pid if state else None, state.status if state else None))

    monkeypatch.setattr("ezllm.runtime.manager.uvicorn.run", fake_uvicorn_run)

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    manager.run_foreground()

    assert calls[0][0] == "start_llama"
    assert calls[0][2] == tmp_path / "logs"
    assert calls[1] == ("uvicorn", {"host": "127.0.0.1", "port": 8888}, 222, "running")
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_does_not_treat_unrelated_live_process_without_started_at_as_owned(tmp_path):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=os.getpid(),
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at=None,
        ),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert terminated == []
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_treats_naive_started_at_as_non_match(tmp_path):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=os.getpid(),
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at="2026-04-24T12:34:56",
        ),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert terminated == []
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_stop_treats_legacy_background_child_without_started_at_as_owned(tmp_path, monkeypatch):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return set()

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    class LegacyBackgroundProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 1234.0

        def cmdline(self):
            return [
                sys.executable,
                "-c",
                "from ezllm.config.loader import load_settings; from ezllm.runtime.manager import RuntimeManager; RuntimeManager(load_settings()).run_foreground()",
            ]

    monkeypatch.setattr("ezllm.runtime.manager.psutil.Process", LegacyBackgroundProcess)
    save_runtime_state(
        tmp_path,
        RuntimeState(
            proxy_pid=101,
            llama_pid=None,
            proxy_port=8888,
            llama_port=8889,
            status="starting",
            started_at=None,
        ),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.stop()

    assert result == "EZLLM stopped"
    assert terminated == [(101, False)]
    assert load_runtime_state(tmp_path) is None


def test_runtime_manager_start_background_force_terminates_foreign_listeners(tmp_path, monkeypatch):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {404} if port == 8888 else {505}

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    monkeypatch.setattr(
        "ezllm.runtime.manager.spawn_background",
        lambda command, **kwargs: SimpleNamespace(pid=606),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    manager.start_background(force=True)
    state = load_runtime_state(tmp_path)

    assert set(terminated) == {(404, True), (505, True)}
    assert state is not None
    assert state.proxy_pid == 606
    assert state.status == "starting"
