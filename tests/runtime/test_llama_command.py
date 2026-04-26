from types import SimpleNamespace

from ezllm.runtime.llama import build_llama_command


def _settings(**llama_overrides):
    llama = {
        "server_bin": "llama-server",
        "model_path": r"C:\models\gemma.gguf",
        "mmproj_path": r"C:\models\mmproj.gguf",
        "gpu_layers": 999,
        "ctx_size": 200000,
        "n_predict": 81920,
        "cache_k_type": "q8_0",
        "cache_v_type": "q8_0",
        "flash_attn": "on",
        "batch_size": 512,
        "parallel": 1,
        "temp": "0.7",
        "top_p": "0.95",
        "top_k": "20",
        "reasoning": "auto",
        "reasoning_format": "deepseek",
        "reasoning_budget": "-1",
    }
    llama.update(llama_overrides)
    return SimpleNamespace(
        runtime=SimpleNamespace(host="127.0.0.1", llama_port=8889),
        llama=SimpleNamespace(**llama),
    )


def test_build_llama_command_includes_legacy_runtime_parameters():
    command = build_llama_command(_settings())

    assert command == [
        "llama-server",
        "-m",
        r"C:\models\gemma.gguf",
        "--mmproj",
        r"C:\models\mmproj.gguf",
        "-ngl",
        "999",
        "-c",
        "200000",
        "-n",
        "81920",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "-fa",
        "on",
        "-b",
        "512",
        "--parallel",
        "1",
        "--temp",
        "0.7",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--reasoning",
        "auto",
        "--reasoning-format",
        "deepseek",
        "--reasoning-budget",
        "-1",
        "--host",
        "127.0.0.1",
        "--port",
        "8889",
    ]


def test_build_llama_command_omits_empty_mmproj_path():
    command = build_llama_command(_settings(mmproj_path=""))

    assert "--mmproj" not in command
