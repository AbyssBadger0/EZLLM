from pathlib import Path

from fastapi import FastAPI

from ezllm.logs.store import history_file_for
from ezllm.proxy.routes_logs import build_logs_router


def build_app(log_dir: Path) -> FastAPI:
    app = FastAPI()
    app.include_router(build_logs_router(history_file_for(Path(log_dir))))
    return app
