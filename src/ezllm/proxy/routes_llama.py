from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
import httpx


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _filtered_headers(headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }


def _upstream_url(settings: Any, path: str, query_string: bytes) -> str:
    suffix = f"/{path}" if path else "/"
    url = f"http://{settings.runtime.host}:{settings.runtime.llama_port}{suffix}"
    if query_string:
        url = f"{url}?{query_string.decode('latin-1')}"
    return url


def build_llama_proxy_router(settings) -> APIRouter:
    router = APIRouter()

    async def proxy_to_llama(request: Request, path: str = "") -> Response:
        url = _upstream_url(settings, path, request.scope.get("query_string", b""))
        async with httpx.AsyncClient(follow_redirects=False, timeout=None) as client:
            upstream = await client.request(
                request.method,
                url,
                headers=_filtered_headers(request.headers),
                content=await request.body(),
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_filtered_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    async def proxy_llama_ui(request: Request, path: str = "") -> Response:
        return await proxy_to_llama(request, path)

    async def proxy_llama_api(request: Request, path: str) -> Response:
        return await proxy_to_llama(request, f"v1/{path}")

    @router.get("/llama")
    def redirect_llama_root() -> RedirectResponse:
        return RedirectResponse("/llama/")

    router.add_api_route(
        "/llama/",
        proxy_llama_ui,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    router.add_api_route(
        "/llama/{path:path}",
        proxy_llama_ui,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    router.add_api_route(
        "/v1/{path:path}",
        proxy_llama_api,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    return router
