from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from ezllm.compat.workbench_page import render_workbench_page


def build_workbench_router() -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    def workbench_page() -> HTMLResponse:
        return HTMLResponse(render_workbench_page())

    return router
