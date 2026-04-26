import json
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
import httpx

from ezllm.logs.store import save_raw_log
from ezllm.proxy.response_normalizer import parse_openai_payload_for_log


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


def _request_json(body: bytes):
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_chat_completion(path: str) -> bool:
    return path.strip("/").lower().endswith("chat/completions")


def _append_sse_payloads(content: bytes, reasoning_parts: list[str], content_parts: list[str]) -> None:
    text = content.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue

        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parse_openai_payload_for_log(payload, reasoning_parts, content_parts)


def _extract_response_text(upstream: httpx.Response) -> tuple[str, str]:
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    content_type = upstream.headers.get("content-type", "").lower()

    if "text/event-stream" in content_type:
        _append_sse_payloads(upstream.content, reasoning_parts, content_parts)
        return "".join(reasoning_parts), "".join(content_parts)

    try:
        payload = upstream.json()
    except Exception:
        return "", ""
    if isinstance(payload, dict):
        parse_openai_payload_for_log(payload, reasoning_parts, content_parts)
    return "".join(reasoning_parts), "".join(content_parts)


def _record_chat_log(
    *,
    settings: Any,
    request_json,
    path: str,
    upstream: httpx.Response,
    duration: float,
) -> None:
    reasoning, content = _extract_response_text(upstream)
    try:
        save_raw_log(
            log_dir=Path(settings.runtime.log_dir),
            req_j=request_json,
            reasoning=reasoning,
            content=content,
            duration=duration,
            path=f"/{path.strip('/')}",
            upstream="local:llama.cpp",
        )
    except Exception:
        return


def build_llama_proxy_router(settings) -> APIRouter:
    router = APIRouter()

    async def proxy_to_llama(request: Request, path: str = "") -> Response:
        url = _upstream_url(settings, path, request.scope.get("query_string", b""))
        body = await request.body()
        started = time.perf_counter()
        async with httpx.AsyncClient(follow_redirects=False, timeout=None) as client:
            upstream = await client.request(
                request.method,
                url,
                headers=_filtered_headers(request.headers),
                content=body,
            )
        duration = time.perf_counter() - started
        if request.method == "POST" and _is_chat_completion(path):
            _record_chat_log(
                settings=settings,
                request_json=_request_json(body),
                path=path,
                upstream=upstream,
                duration=duration,
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
