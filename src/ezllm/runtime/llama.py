import subprocess
from collections.abc import Sequence
from pathlib import Path


def _append_option(command: list[str], flag: str, value) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def build_llama_command(settings) -> list[str]:
    runtime = settings.runtime
    llama = settings.llama
    command = [
        llama.server_bin,
        "-m",
        llama.model_path,
    ]
    _append_option(command, "--mmproj", getattr(llama, "mmproj_path", None))
    command.extend(
        [
            "-ngl",
            str(llama.gpu_layers),
            "-c",
            str(llama.ctx_size),
            "-n",
            str(llama.n_predict),
            "-ctk",
            str(llama.cache_k_type),
            "-ctv",
            str(llama.cache_v_type),
            "-fa",
            str(llama.flash_attn),
            "-b",
            str(llama.batch_size),
            "--parallel",
            str(llama.parallel),
            "--temp",
            str(llama.temp),
            "--top-p",
            str(llama.top_p),
            "--top-k",
            str(llama.top_k),
            "--reasoning",
            str(llama.reasoning),
            "--reasoning-format",
            str(llama.reasoning_format),
            "--reasoning-budget",
            str(llama.reasoning_budget),
            "--host",
            runtime.host,
            "--port",
            str(runtime.llama_port),
        ]
    )
    return command


def start_llama_server(settings, log_dir: str | Path) -> subprocess.Popen:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "llama-server.log"
    command: Sequence[str] = build_llama_command(settings)
    with log_path.open("ab") as handle:
        return subprocess.Popen(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
