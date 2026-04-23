from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter

from ezllm.runtime.health import build_runtime_config_payload


def build_runtime_router(
    settings,
    provider_summary: Mapping[str, Any] | None = None,
) -> APIRouter:
    router = APIRouter()

    @router.get("/runtime-config")
    def runtime_config() -> dict[str, Any]:
        return build_runtime_config_payload(settings, provider_summary)

    return router
