from pathlib import Path

from ezllm.runtime.state import load_runtime_state


class RuntimeManager:
    def __init__(self, settings, platform_adapter=None):
        self.settings = settings
        self.platform_adapter = platform_adapter
        self.state_dir = Path(settings.runtime.state_dir)

    def format_status(self) -> str:
        state = load_runtime_state(self.state_dir)
        if state is None:
            return "EZLLM not running"
        return (
            f"EZLLM running on proxy:{state.proxy_port} llama:{state.llama_port} "
            f"(proxy pid={state.proxy_pid}, llama pid={state.llama_pid})"
        )
