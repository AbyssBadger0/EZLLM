from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from ezllm.compat.api_contracts import to_legacy_logs_payload
from ezllm.compat.logs_page import render_logs_page
from ezllm.logs.reader import paginate_entries, read_log_entries


def build_logs_router(history_file: Path) -> APIRouter:
    router = APIRouter()

    @router.get("/logs", response_class=HTMLResponse)
    def logs_page() -> HTMLResponse:
        return HTMLResponse(render_logs_page())

    @router.get("/api/logs")
    def api_logs(page: int = Query(1), size: int = Query(10)) -> dict:
        lines, _ = read_log_entries(history_file)
        payload = paginate_entries(lines, page=page, size=size)
        return to_legacy_logs_payload(payload)

    return router
