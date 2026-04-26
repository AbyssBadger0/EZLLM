from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from ezllm.compat.control_page import render_control_page
from ezllm.config.loader import _config_path, load_settings, update_config_values
from ezllm.config.models import Settings
from ezllm.runtime.control_actions import ScheduledControlActions


def _serializable_settings(settings: Settings, config_path: Path) -> dict[str, Any]:
    payload = settings.model_dump()
    payload["config_path"] = str(config_path)
    return payload


def _filter_updates(payload: Mapping[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for section in ("runtime", "llama"):
        value = payload.get(section)
        if isinstance(value, Mapping):
            updates[section] = dict(value)
    return updates


def build_control_router(
    settings: Settings,
    *,
    config_path: str | Path | None = None,
    control_actions=None,
) -> APIRouter:
    router = APIRouter()
    path = Path(config_path).expanduser() if config_path is not None else _config_path()
    actions = control_actions or ScheduledControlActions()

    @router.get("/control", response_class=HTMLResponse)
    def control_page() -> HTMLResponse:
        return HTMLResponse(render_control_page())

    @router.get("/api/control/config")
    def read_config() -> dict[str, Any]:
        return _serializable_settings(settings, path)

    @router.put("/api/control/config")
    def write_config(payload: dict[str, Any]) -> dict[str, Any]:
        updates = _filter_updates(payload)
        if not updates:
            raise HTTPException(status_code=400, detail="No runtime or llama settings provided.")
        update_config_values(updates, path=path)
        try:
            updated_settings = load_settings(path)
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response = _serializable_settings(updated_settings, path)
        response["restart_required"] = True
        return response

    @router.post("/api/control/restart", status_code=202)
    def restart_runtime() -> dict[str, str]:
        actions.restart()
        return {"status": "restart-scheduled"}

    @router.post("/api/control/stop", status_code=202)
    def stop_runtime() -> dict[str, str]:
        actions.stop()
        return {"status": "stop-scheduled"}

    return router
