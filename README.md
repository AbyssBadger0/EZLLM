# EZLLM

Cross-platform local LLM runtime and CLI with legacy `lm_core` compatibility.

For the current project handoff, architecture notes, implemented surface area,
known gaps, and next milestone plan, see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).

## Current Status

EZLLM currently provides a CLI-managed local runtime that launches both the
FastAPI workbench proxy and a configured `llama-server` process. It exposes a
browser workbench, the llama.cpp Web UI, logs, health, runtime metadata, and
OpenAI-compatible llama.cpp API proxying.

## Platform Support

- Windows: primary development and smoke-test platform.
- Linux: supported by the Python runtime, config paths, process manager, and CI
  test suite. Install or build `llama-server` manually, then point EZLLM at it.
- macOS: basic runtime paths are supported. Install or build `llama-server`
  manually.

## Install EZLLM

### Windows PowerShell

```powershell
cd C:\path\to\EZLLM
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

Check the CLI:

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli --help
.\.venv\Scripts\python.exe -m ezllm.cli doctor
```

### Linux or macOS

```bash
cd /path/to/EZLLM
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Check the CLI:

```bash
python -m ezllm.cli --help
python -m ezllm.cli doctor
```

## Install llama.cpp Runtime

Do not commit llama.cpp release zips, DLLs, models, or build outputs into this
repository. Windows CUDA bundles can be hundreds of MB because they include
NVIDIA runtime DLLs. This repo keeps those files out of Git and downloads them
locally when needed.

### Windows: Download Official Release Assets

On Windows, run:

```powershell
# Auto-pick cuda13 for Blackwell/compute capability 12.x NVIDIA GPUs,
# cuda12 for other NVIDIA GPUs, otherwise CPU.
.\scripts\install_llama_cpp.ps1

# Or choose explicitly.
.\scripts\install_llama_cpp.ps1 -Backend cuda13
.\scripts\install_llama_cpp.ps1 -Backend cuda12
.\scripts\install_llama_cpp.ps1 -Backend cpu
.\scripts\install_llama_cpp.ps1 -Backend vulkan
```

The script downloads the latest official `ggml-org/llama.cpp` release into:

```text
vendor/llama/win-x64-<backend>/
```

For CUDA backends it downloads both the llama.cpp binary zip and the matching
`cudart-...` runtime zip. You still need a compatible NVIDIA driver installed,
but you do not need Visual Studio, CMake, or the CUDA Toolkit just to run the
downloaded binary.

Use `-DryRun` to see what would be downloaded:

```powershell
.\scripts\install_llama_cpp.ps1 -Backend cuda13 -DryRun
```

### Manual Download

You can also download packages manually from:

```text
https://github.com/ggml-org/llama.cpp/releases
```

Choose one Windows x64 package:

```text
llama-<tag>-bin-win-cpu-x64.zip
llama-<tag>-bin-win-cuda-13.1-x64.zip
llama-<tag>-bin-win-cuda-12.4-x64.zip
llama-<tag>-bin-win-vulkan-x64.zip
```

For CUDA, also download the matching runtime package:

```text
cudart-llama-bin-win-cuda-13.1-x64.zip
cudart-llama-bin-win-cuda-12.4-x64.zip
```

Extract the files into a local folder such as:

```text
vendor/llama/win-x64-cuda13/
```

Keep `llama-server.exe` together with all DLLs from the same release.

### Build From Source

Use this route if you want a custom CUDA architecture, a newer source commit
than the latest release, or local build flags.

Windows CUDA example:

```powershell
git clone https://github.com/ggml-org/llama.cpp C:\Users\$env:USERNAME\llama.cpp
cd C:\Users\$env:USERNAME\llama.cpp

cmake -S . -B build-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-cuda --target llama-server -j 16
```

The built server is usually here:

```text
C:\Users\<you>\llama.cpp\build-cuda\bin\llama-server.exe
```

### Linux: Build llama.cpp

Install your distribution's build tools, CMake, and a compiler first. CPU build:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-cpu --target llama-server -j "$(nproc)"
```

CUDA build:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-cuda --target llama-server -j "$(nproc)"
```

The built server is usually here:

```text
/home/<you>/llama.cpp/build-cpu/bin/llama-server
/home/<you>/llama.cpp/build-cuda/bin/llama-server
```

You can also extract an official Linux release into an ignored local folder such
as `vendor/llama/linux-x64-cpu/` and point `llama.server_bin` at the extracted
`llama-server` binary.

### macOS: Build llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build --target llama-server -j 8
```

The built server is usually here:

```text
/Users/<you>/llama.cpp/build/bin/llama-server
```

## Configure EZLLM

Create a config file at the platform default location:

```text
Windows: C:\Users\<you>\AppData\Roaming\EZLLM\config.toml
Linux:   ~/.config/ezllm/config.toml
macOS:   ~/Library/Application Support/EZLLM/config.toml
```

Windows example:

```toml
[runtime]
host = "127.0.0.1"
proxy_port = 8888
llama_port = 8889

[llama]
server_bin = 'C:\path\to\EZLLM\vendor\llama\win-x64-cuda13\llama-server.exe'
model_path = 'C:\path\to\your-model.gguf'
mmproj_path = 'C:\path\to\mmproj.gguf'
model_scan_dirs = ['C:\Users\<you>\.lmstudio\models', 'D:\Models']
llama_cpp_dirs = ['C:\path\to\EZLLM\vendor\llama', 'C:\Users\<you>\llama.cpp']
ctx_size = 200000
n_predict = 81920
parallel = 1
gpu_layers = 999
batch_size = 512
flash_attn = "on"
cache_k_type = "q8_0"
cache_v_type = "q8_0"
temp = "0.7"
top_p = "0.95"
top_k = "20"
reasoning = "auto"
reasoning_format = "deepseek"
reasoning_budget = "-1"
```

Linux example:

```toml
[runtime]
host = "127.0.0.1"
proxy_port = 8888
llama_port = 8889

[llama]
server_bin = "/home/<you>/llama.cpp/build-cuda/bin/llama-server"
model_path = "/home/<you>/models/your-model.gguf"
mmproj_path = "/home/<you>/models/mmproj.gguf"
model_scan_dirs = ["/home/<you>/models"]
llama_cpp_dirs = ["/home/<you>/llama.cpp", "/path/to/EZLLM/vendor/llama"]
ctx_size = 200000
n_predict = 81920
parallel = 1
gpu_layers = 999
batch_size = 512
flash_attn = "on"
cache_k_type = "q8_0"
cache_v_type = "q8_0"
temp = "0.7"
top_p = "0.95"
top_k = "20"
reasoning = "auto"
reasoning_format = "deepseek"
reasoning_budget = "-1"
```

You can also point `model_path` at a GGUF file downloaded by LM Studio, for
example under:

```text
C:\Users\<you>\.lmstudio\models\
```

## Run

Start EZLLM in the foreground:

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli run
```

Linux/macOS:

```bash
python -m ezllm.cli run
```

Start it in the background and open the browser control page:

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli start --open
```

Linux/macOS:

```bash
python -m ezllm.cli start --open
```

The workbench is available at:

```text
http://127.0.0.1:8888/
```

From the workbench, open:

```text
http://127.0.0.1:8888/llama/
http://127.0.0.1:8888/control
http://127.0.0.1:8888/logs
```

EZLLM keeps `8889` as the internal llama.cpp server port, while browser and API
entrypoints can stay on `8888`.

Useful CLI commands:

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli status
.\.venv\Scripts\python.exe -m ezllm.cli stop
.\.venv\Scripts\python.exe -m ezllm.cli restart --force
.\.venv\Scripts\python.exe -m ezllm.cli open
.\.venv\Scripts\python.exe -m ezllm.cli config set llama.ctx_size 65536
```

Linux/macOS:

```bash
python -m ezllm.cli status
python -m ezllm.cli stop
python -m ezllm.cli restart --force
python -m ezllm.cli open
python -m ezllm.cli config set llama.ctx_size 65536
```

Useful diagnostics:

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli doctor
Invoke-RestMethod http://127.0.0.1:8888/healthz
Invoke-RestMethod http://127.0.0.1:8888/runtime-config
```

Linux/macOS:

```bash
curl http://127.0.0.1:8888/healthz
curl http://127.0.0.1:8888/runtime-config
```

EZLLM manages `llama-server` internally after `run` or `start`. You can send
local chat smoke tests through EZLLM on port `8888`:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8888/v1/chat/completions `
  -ContentType "application/json" `
  -Body '{"model":"local","messages":[{"role":"user","content":"Say hello in one short sentence."}],"stream":false}'
```

Linux/macOS:

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Say hello in one short sentence."}],"stream":false}'
```

EZLLM also accepts a provider-neutral reasoning control parameter and maps it to
the active llama.cpp backend before forwarding:

```json
{
  "reasoning": {
    "effort": "off"
  }
}
```

Supported effort values are `off`, `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `extra_high`, `extra high`, and `auto`. For llama.cpp, `off` maps to
`chat_template_kwargs.enable_thinking=false` and `thinking_budget_tokens=0`.
Other effort values enable thinking and set a reasoning budget for the request.
EZLLM also accepts OpenAI Chat Completions style `reasoning_effort`. If neither
`reasoning` nor `reasoning_effort` is sent, EZLLM defaults chat requests to
thinking disabled.

The browser page can edit the same runtime and llama parameters, then schedule a
restart so the saved settings take effect.

The Control page also includes local discovery helpers:

- Add one or more `Model Scan Directories`, then click `Scan Models` to fill a
  model dropdown from `.gguf` files. Selecting a model writes `llama.model_path`
  and automatically fills `llama.mmproj_path` when a matching `mmproj*.gguf`
  file is found in the same folder.
- Add one or more `llama.cpp Directories`, then click `Scan llama.cpp` to find
  `llama-server` or `llama-server.exe` binaries. Selecting one writes
  `llama.server_bin`.
- Use `Browse` or `Add Folder` in Control to navigate local folders from the
  browser UI when you do not want to paste paths manually.

## Parameter Environment Overrides

The config file can be overridden with environment variables:

```powershell
$env:EZLLM_CONFIG = 'C:\path\to\config.toml'
$env:EZLLM_SERVER_BIN = 'C:\path\to\llama-server.exe'
$env:EZLLM_MODEL_PATH = 'C:\path\to\model.gguf'
$env:EZLLM_MMPROJ_PATH = 'C:\path\to\mmproj.gguf'
$env:EZLLM_CTX_SIZE = '65536'
$env:EZLLM_N_PREDICT = '16384'
$env:EZLLM_REASONING = 'off'
```

Linux/macOS:

```bash
export EZLLM_CONFIG='/path/to/config.toml'
export EZLLM_SERVER_BIN='/path/to/llama-server'
export EZLLM_MODEL_PATH='/path/to/model.gguf'
export EZLLM_MMPROJ_PATH='/path/to/mmproj.gguf'
export EZLLM_CTX_SIZE='65536'
export EZLLM_N_PREDICT='16384'
export EZLLM_REASONING='off'
```
