from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter

from ezllm.runtime.health import (
    build_health_payload,
    build_runtime_config_payload,
    load_runtime_summary,
)


def build_health_router(
    settings,
    provider_summary: Mapping[str, Any] | None = None,
    *,
    proxy_pid: int | None = None,
    llama_pid: int | None = None,
    llama_status: str | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    def healthz() -> dict[str, Any]:
        runtime_config = build_runtime_config_payload(settings, provider_summary)
        runtime_summary = load_runtime_summary(settings)
        return build_health_payload(
            settings,
            runtime_config,
            proxy_pid=proxy_pid if proxy_pid is not None else runtime_summary["proxy_pid"],
            llama_pid=llama_pid if llama_pid is not None else runtime_summary["llama_pid"],
            llama_status=llama_status if llama_status is not None else runtime_summary["llama_status"],
        )

    return router
