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


def test_runtime_manager_format_status_reports_not_running(tmp_path):
    settings = SimpleNamespace(
        runtime=SimpleNamespace(state_dir=str(tmp_path), proxy_port=8888, llama_port=8889),
        llama=SimpleNamespace(server_bin="llama-server", model_path="model.gguf"),
    )

    manager = RuntimeManager(settings)

    assert "not running" in manager.format_status().lower()
