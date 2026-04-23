from types import SimpleNamespace

from ezllm.runtime.manager import RuntimeManager
from ezllm.runtime.state import RuntimeState, load_runtime_state, save_runtime_state


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
    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )
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
    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )
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
    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )

    manager = RuntimeManager(settings)

    assert "not running" in manager.format_status().lower()
