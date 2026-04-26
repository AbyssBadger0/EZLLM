# Contributing

Thanks for helping improve EZLLM.

## Development Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Pull Requests

- Keep changes focused.
- Add tests for behavior changes.
- Run `python -m pytest tests -q` before opening a PR.
- Do not commit local models, llama.cpp runtime bundles, logs, or config files.
- Update README documentation when changing setup, configuration, or user-facing
  behavior.
