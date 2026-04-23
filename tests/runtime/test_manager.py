from types import SimpleNamespace

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
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
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


def test_runtime_manager_stop_only_terminates_owned_pids_from_state(tmp_path):
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
    assert set(terminated) == {(101, False), (202, False)}
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
    monkeypatch.setattr("ezllm.runtime.manager.spawn_background", lambda command: spawned.append(command))

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
        lambda command: SimpleNamespace(pid=303),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    result = manager.start_background(force=False)
    state = load_runtime_state(tmp_path)

    assert "starting" in result.lower()
    assert set(terminated) == {(101, False), (202, False)}
    assert state is not None
    assert state.proxy_pid == 303
    assert state.status == "starting"


def test_runtime_manager_start_background_force_terminates_foreign_listeners(tmp_path, monkeypatch):
    terminated = []

    class FakePlatformAdapter:
        def find_listening_pids(self, port):
            return {404} if port == 8888 else {505}

        def terminate_tree(self, pid, *, force=False):
            terminated.append((pid, force))

    monkeypatch.setattr(
        "ezllm.runtime.manager.spawn_background",
        lambda command: SimpleNamespace(pid=606),
    )

    manager = RuntimeManager(_settings(tmp_path), platform_adapter=FakePlatformAdapter())

    manager.start_background(force=True)
    state = load_runtime_state(tmp_path)

    assert set(terminated) == {(404, True), (505, True)}
    assert state is not None
    assert state.proxy_pid == 606
    assert state.status == "starting"
