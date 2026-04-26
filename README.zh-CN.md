# EZLLM

[English](README.md)

EZLLM 是一个跨平台的本地 LLM 运行时、浏览器工作台和 OpenAI-compatible 代理，基于 llama.cpp 管理本地模型服务。

## EZLLM 是什么

EZLLM 现在可以通过 CLI 启动并管理本地运行时：它会启动 FastAPI 工作台代理和配置好的 `llama-server` 进程。它提供浏览器工作台、llama.cpp Web UI、日志页面、健康检查、运行时元信息，以及 OpenAI-compatible 的 llama.cpp API 代理。

EZLLM 不内置模型文件，也不把 llama.cpp 二进制文件提交到仓库。你需要自己准备 GGUF 模型，并把 EZLLM 指向一个可用的 `llama-server`。

## 功能特性

- 通过 EZLLM CLI 管理本地 `llama-server` 进程
- 在一个本地端口上提供浏览器工作台
- 把 llama.cpp 原生 Web UI 代理到 `/llama/`
- 在 `/v1/chat/completions` 提供 OpenAI-compatible 聊天代理
- 记录聊天请求和响应，方便在日志页面检查
- 通过浏览器 Control 页面配置模型路径、llama.cpp 二进制文件和 reasoning 行为
- 扫描本地模型目录和 llama.cpp 目录，同时避免把大模型或二进制文件提交到仓库

## 运行要求

- Python 3.11 或更高版本
- 一个兼容的 llama.cpp `llama-server` 二进制文件
- 至少一个本地 GGUF 模型文件
- GPU 和对应运行时不是必须的，但大模型强烈建议使用

## 平台支持

- Windows：已验证 CLI 托管运行时和随仓库提供的 llama.cpp 安装脚本。
- Linux：Python 运行时、配置路径、进程管理和 CI 测试均已覆盖。需要手动安装或编译 `llama-server`。
- macOS：基础运行时路径可用。需要手动安装或编译 `llama-server`。

## 安装 EZLLM

### Windows PowerShell

```powershell
cd C:\path\to\EZLLM
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

检查 CLI：

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli --help
.\.venv\Scripts\python.exe -m ezllm.cli doctor
```

### Linux 或 macOS

```bash
cd /path/to/EZLLM
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

检查 CLI：

```bash
python -m ezllm.cli --help
python -m ezllm.cli doctor
```

## 安装 llama.cpp 运行时

请不要把 llama.cpp release 压缩包、DLL、模型或编译产物提交到这个仓库。Windows CUDA 包可能有数百 MB，因为它们包含 NVIDIA 运行时 DLL。EZLLM 会把这些文件放在本地目录，并通过 `.gitignore` 忽略。

### Windows：下载官方 release

在 Windows 上运行：

```powershell
# 自动选择后端：Blackwell/compute capability 12.x NVIDIA GPU 使用 cuda13，
# 其他 NVIDIA GPU 使用 cuda12，否则使用 CPU。
.\scripts\install_llama_cpp.ps1

# 也可以手动指定。
.\scripts\install_llama_cpp.ps1 -Backend cuda13
.\scripts\install_llama_cpp.ps1 -Backend cuda12
.\scripts\install_llama_cpp.ps1 -Backend cpu
.\scripts\install_llama_cpp.ps1 -Backend vulkan
```

脚本会把最新的 `ggml-org/llama.cpp` 官方 release 下载到：

```text
vendor/llama/win-x64-<backend>/
```

CUDA 后端会同时下载 llama.cpp 二进制 zip 和匹配的 `cudart-...` 运行时 zip。你仍然需要安装兼容的 NVIDIA 驱动，但仅运行下载好的二进制包时，不需要 Visual Studio、CMake 或 CUDA Toolkit。

可以用 `-DryRun` 预览下载内容：

```powershell
.\scripts\install_llama_cpp.ps1 -Backend cuda13 -DryRun
```

### Windows：手动下载

也可以从以下地址手动下载：

```text
https://github.com/ggml-org/llama.cpp/releases
```

选择一个 Windows x64 包：

```text
llama-<tag>-bin-win-cpu-x64.zip
llama-<tag>-bin-win-cuda-13.1-x64.zip
llama-<tag>-bin-win-cuda-12.4-x64.zip
llama-<tag>-bin-win-vulkan-x64.zip
```

CUDA 还需要下载匹配的运行时包：

```text
cudart-llama-bin-win-cuda-13.1-x64.zip
cudart-llama-bin-win-cuda-12.4-x64.zip
```

解压到本地目录，例如：

```text
vendor/llama/win-x64-cuda13/
```

请保持 `llama-server.exe` 和同一 release 的 DLL 文件在一起。

### Windows：从源码编译

如果你需要自定义 CUDA 架构、使用比 release 更新的源码，或指定本地编译参数，可以从源码编译。

Windows CUDA 示例：

```powershell
git clone https://github.com/ggml-org/llama.cpp C:\Users\$env:USERNAME\llama.cpp
cd C:\Users\$env:USERNAME\llama.cpp

cmake -S . -B build-cuda -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CUDA=ON `
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-cuda --target llama-server -j 16
```

编译后的 server 通常在：

```text
C:\Users\<you>\llama.cpp\build-cuda\bin\llama-server.exe
```

### Linux：编译 llama.cpp

先安装发行版对应的构建工具、CMake 和编译器。CPU 构建：

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-cpu --target llama-server -j "$(nproc)"
```

CUDA 构建：

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-cuda --target llama-server -j "$(nproc)"
```

编译后的 server 通常在：

```text
/home/<you>/llama.cpp/build-cpu/bin/llama-server
/home/<you>/llama.cpp/build-cuda/bin/llama-server
```

你也可以把官方 Linux release 解压到被 Git 忽略的本地目录，例如 `vendor/llama/linux-x64-cpu/`，然后把 `llama.server_bin` 指向其中的 `llama-server`。

### macOS：编译 llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build --target llama-server -j 8
```

编译后的 server 通常在：

```text
/Users/<you>/llama.cpp/build/bin/llama-server
```

## 配置 EZLLM

在平台默认位置创建配置文件：

```text
Windows: C:\Users\<you>\AppData\Roaming\EZLLM\config.toml
Linux:   ~/.config/ezllm/config.toml
macOS:   ~/Library/Application Support/EZLLM/config.toml
```

Windows 示例：

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

Linux 示例：

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

LM Studio 下载的 GGUF 模型也可以直接使用，例如 Windows 下通常在：

```text
C:\Users\<you>\.lmstudio\models\
```

## 运行

前台启动：

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli run
```

Linux/macOS：

```bash
python -m ezllm.cli run
```

后台启动并打开浏览器控制台：

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli start --open
```

Linux/macOS：

```bash
python -m ezllm.cli start --open
```

工作台地址：

```text
http://127.0.0.1:8888/
```

工作台中可以进入：

```text
http://127.0.0.1:8888/llama/
http://127.0.0.1:8888/control
http://127.0.0.1:8888/logs
```

EZLLM 会把 `8889` 作为内部 llama.cpp server 端口，浏览器和 API 入口可以统一使用 `8888`。

常用 CLI 命令：

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli status
.\.venv\Scripts\python.exe -m ezllm.cli stop
.\.venv\Scripts\python.exe -m ezllm.cli restart --force
.\.venv\Scripts\python.exe -m ezllm.cli open
.\.venv\Scripts\python.exe -m ezllm.cli config set llama.ctx_size 65536
```

Linux/macOS：

```bash
python -m ezllm.cli status
python -m ezllm.cli stop
python -m ezllm.cli restart --force
python -m ezllm.cli open
python -m ezllm.cli config set llama.ctx_size 65536
```

Linux/systemd 服务命令与跨平台运行时命令分开提供。Windows 和非 systemd
主机会直接拒绝这些命令，不会影响原有 Windows CLI 运行方式。

```bash
python -m ezllm.cli service install \
  --name ezllm.service \
  --python /path/to/python \
  --config ~/.config/ezllm/config.toml \
  --user "$USER" \
  --working-directory /path/to/EZLLM \
  --enable

python -m ezllm.cli service restart --name ezllm.service
python -m ezllm.cli service status --name ezllm.service
python -m ezllm.cli service log --name ezllm.service
```

诊断命令：

```powershell
.\.venv\Scripts\python.exe -m ezllm.cli doctor
Invoke-RestMethod http://127.0.0.1:8888/healthz
Invoke-RestMethod http://127.0.0.1:8888/runtime-config
```

Linux/macOS：

```bash
curl http://127.0.0.1:8888/healthz
curl http://127.0.0.1:8888/runtime-config
```

EZLLM 在 `run` 或 `start` 后会托管 `llama-server`。你可以通过 `8888` 发送本地聊天测试：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8888/v1/chat/completions `
  -ContentType "application/json" `
  -Body '{"model":"local","messages":[{"role":"user","content":"Say hello in one short sentence."}],"stream":false}'
```

Linux/macOS：

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Say hello in one short sentence."}],"stream":false}'
```

## Reasoning / Thinking 控制

EZLLM 支持一个供应商中立的 reasoning 控制字段，并会在转发给 llama.cpp 前映射到 llama.cpp 参数：

```json
{
  "reasoning": {
    "effort": "off"
  }
}
```

支持的 effort 值包括：`off`、`none`、`minimal`、`low`、`medium`、`high`、`xhigh`、`extra_high`、`extra high` 和 `auto`。

对 llama.cpp 来说：

- `off` 会映射为 `chat_template_kwargs.enable_thinking=false` 和 `thinking_budget_tokens=0`
- 其他 effort 值会启用 thinking，并设置请求级 reasoning budget
- 也兼容 OpenAI Chat Completions 风格的 `reasoning_effort`
- 如果请求里既没有 `reasoning`，也没有 `reasoning_effort`，EZLLM 默认关闭思考模式

## 浏览器 Control 页面

Control 页面可以编辑运行时和 llama 参数，并安排重启让配置生效。它还提供本地发现能力：

- 添加一个或多个 `Model Scan Directories`，点击 `Scan Models`，从 `.gguf` 文件生成模型下拉框
- 选中模型后写入 `llama.model_path`，如果同目录存在 `mmproj*.gguf`，会自动填入 `llama.mmproj_path`
- 添加一个或多个 `llama.cpp Directories`，点击 `Scan llama.cpp`，查找 `llama-server` 或 `llama-server.exe`
- 选中 server 后写入 `llama.server_bin`
- 可以使用 `Browse` 或 `Add Folder` 在网页里浏览本机目录，避免手动复制路径

## 环境变量覆盖

配置文件可以被环境变量覆盖。

Windows PowerShell：

```powershell
$env:EZLLM_CONFIG = 'C:\path\to\config.toml'
$env:EZLLM_SERVER_BIN = 'C:\path\to\llama-server.exe'
$env:EZLLM_MODEL_PATH = 'C:\path\to\model.gguf'
$env:EZLLM_MMPROJ_PATH = 'C:\path\to\mmproj.gguf'
$env:EZLLM_CTX_SIZE = '65536'
$env:EZLLM_N_PREDICT = '16384'
$env:EZLLM_REASONING = 'off'
```

Linux/macOS：

```bash
export EZLLM_CONFIG='/path/to/config.toml'
export EZLLM_SERVER_BIN='/path/to/llama-server'
export EZLLM_MODEL_PATH='/path/to/model.gguf'
export EZLLM_MMPROJ_PATH='/path/to/mmproj.gguf'
export EZLLM_CTX_SIZE='65536'
export EZLLM_N_PREDICT='16384'
export EZLLM_REASONING='off'
```
