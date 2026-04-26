# llama.cpp Runtime Bundles

This directory is the default local destination for downloaded llama.cpp runtime bundles.

The binaries are intentionally ignored by Git. Use `scripts/install_llama_cpp.ps1`
to download official release assets, or place your own compiled `llama-server.exe`
bundle here.

Expected examples:

```text
vendor/llama/win-x64-cpu/
vendor/llama/win-x64-cuda13/
vendor/llama/win-x64-vulkan/
```

Keep the `llama-server.exe` file together with the DLL files from the same
release or build output directory.
