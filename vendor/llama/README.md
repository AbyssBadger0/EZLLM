# llama.cpp Runtime Bundles

This directory is the default local destination for downloaded llama.cpp runtime bundles.

The binaries are intentionally ignored by Git. On Windows, use
`scripts/install_llama_cpp.ps1` to download official release assets, or place
your own compiled `llama-server.exe` bundle here. On Linux and macOS, place your
downloaded or compiled `llama-server` bundle here and point `llama.server_bin`
at the binary.

Expected examples:

```text
vendor/llama/win-x64-cpu/
vendor/llama/win-x64-cuda13/
vendor/llama/win-x64-vulkan/
vendor/llama/linux-x64-cpu/
vendor/llama/linux-x64-cuda/
vendor/llama/macos-arm64/
```

Keep `llama-server` or `llama-server.exe` together with the shared libraries
from the same release or build output directory.
