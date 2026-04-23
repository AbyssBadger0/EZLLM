from pathlib import Path

from fastapi import FastAPI

from ezllm.config.loader import load_settings
from ezllm.logs.store import history_file_for
from ezllm.proxy.routes_health import build_health_router
from ezllm.proxy.routes_logs import build_logs_router
from ezllm.proxy.routes_runtime import build_runtime_router


def _with_log_dir(settings, log_dir: Path):
    runtime = settings.runtime.model_copy(update={"log_dir": str(Path(log_dir))})
    return settings.model_copy(update={"runtime": runtime})


def build_app(*, log_dir: Path, settings=None, provider_summary=None) -> FastAPI:
    effective_settings = settings if settings is not None else load_settings()
    effective_settings = _with_log_dir(effective_settings, Path(log_dir))
    app = FastAPI()
    app.include_router(build_logs_router(history_file_for(Path(log_dir))))
    app.include_router(build_runtime_router(effective_settings, provider_summary))
    app.include_router(build_health_router(effective_settings, provider_summary))
    return app
