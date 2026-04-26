import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from ezllm.config.models import LlamaConfig, RuntimeConfig, Settings
from ezllm.logs.store import history_file_for
from ezllm.proxy.app import build_app


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        runtime=RuntimeConfig(
            host="127.0.0.1",
            proxy_port=8890,
            llama_port=8891,
            log_dir=str(tmp_path / "logs"),
            state_dir=str(tmp_path / "state"),
        ),
        llama=LlamaConfig(
            server_bin="llama-server",
            model_path=r"C:\models\legacy-model.gguf",
        ),
    )


class FakeAsyncClient:
    requests = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def request(self, method, url, *, headers=None, content=None):
        self.requests.append(
            {
                "method": method,
                "url": str(url),
                "headers": dict(headers or {}),
                "content": content,
            }
        )
        if str(url).endswith("/v1/chat/completions") and content and b'"stream":true' in content:
            return httpx.Response(
                200,
                content=(
                    b'data: {"choices":[{"delta":{"reasoning_content":"thinking "}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                    b'data: {"choices":[{"delta":{"content":" from stream"}}]}\n\n'
                    b"data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        if str(url).endswith("/v1/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "reasoning_content": "thinking",
                                "content": "hello from llama",
                            }
                        }
                    ]
                },
                headers={"content-type": "application/json"},
            )
        if str(url).endswith("/bundle.js?cache=1"):
            return httpx.Response(200, content=b"console.log('llama')", headers={"content-type": "text/javascript"})
        return httpx.Response(200, content=b"<html>llama.cpp</html>", headers={"content-type": "text/html"})


def test_root_path_renders_ezllm_workbench(tmp_path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr("ezllm.proxy.routes_llama.httpx.AsyncClient", FakeAsyncClient)
    client = TestClient(build_app(log_dir=tmp_path, settings=_settings(tmp_path)))

    response = client.get("/")

    assert response.status_code == 200
    assert "EZLLM Workbench" in response.text
    assert 'href="/llama/"' in response.text
    assert 'href="/control"' in response.text
    assert 'href="/logs"' in response.text
    assert FakeAsyncClient.requests == []


def test_llama_path_proxies_to_llama_cpp_web_ui(tmp_path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr("ezllm.proxy.routes_llama.httpx.AsyncClient", FakeAsyncClient)
    client = TestClient(build_app(log_dir=tmp_path, settings=_settings(tmp_path)))

    response = client.get("/llama/")

    assert response.status_code == 200
    assert response.text == "<html>llama.cpp</html>"
    assert FakeAsyncClient.requests[0]["method"] == "GET"
    assert FakeAsyncClient.requests[0]["url"] == "http://127.0.0.1:8891/"


def test_llama_static_assets_proxy_to_matching_upstream_path(tmp_path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr("ezllm.proxy.routes_llama.httpx.AsyncClient", FakeAsyncClient)
    client = TestClient(build_app(log_dir=tmp_path, settings=_settings(tmp_path)))

    response = client.get("/llama/bundle.js?cache=1")

    assert response.status_code == 200
    assert response.text == "console.log('llama')"
    assert FakeAsyncClient.requests[0]["url"] == "http://127.0.0.1:8891/bundle.js?cache=1"


def test_ezllm_control_routes_are_not_shadowed_by_llama_proxy(tmp_path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr("ezllm.proxy.routes_llama.httpx.AsyncClient", FakeAsyncClient)
    client = TestClient(build_app(log_dir=tmp_path, settings=_settings(tmp_path)))

    response = client.get("/control")

    assert response.status_code == 200
    assert "EZLLM Control" in response.text
    assert FakeAsyncClient.requests == []


def test_openai_api_request_can_proxy_to_llama_server(tmp_path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr("ezllm.proxy.routes_llama.httpx.AsyncClient", FakeAsyncClient)
    client = TestClient(build_app(log_dir=tmp_path, settings=_settings(tmp_path)))

    response = client.post("/v1/chat/completions", json={"model": "local"})

    assert response.status_code == 200
    assert FakeAsyncClient.requests[0]["method"] == "POST"
    assert FakeAsyncClient.requests[0]["url"] == "http://127.0.0.1:8891/v1/chat/completions"
    assert FakeAsyncClient.requests[0]["content"] == b'{"model":"local"}'


def test_openai_chat_proxy_persists_logs_for_logs_page(tmp_path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr("ezllm.proxy.routes_llama.httpx.AsyncClient", FakeAsyncClient)
    client = TestClient(build_app(log_dir=tmp_path, settings=_settings(tmp_path)))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    assert response.status_code == 200
    history_file = history_file_for(tmp_path)
    assert history_file.exists()
    raw_entry = json.loads(history_file.read_text(encoding="utf-8").strip())
    assert raw_entry["path"] == "/v1/chat/completions"
    assert raw_entry["upstream"] == "local:llama.cpp"
    assert raw_entry["request_raw"]["messages"][0]["content"] == "ping"
    assert raw_entry["response_raw"] == {
        "reasoning": "thinking",
        "content": "hello from llama",
    }

    logs_response = client.get("/api/logs?page=1&size=10")
    assert logs_response.json()["total"] == 1


def test_streaming_openai_chat_proxy_persists_logs_for_logs_page(tmp_path, monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr("ezllm.proxy.routes_llama.httpx.AsyncClient", FakeAsyncClient)
    client = TestClient(build_app(log_dir=tmp_path, settings=_settings(tmp_path)))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local",
            "stream": True,
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    assert response.status_code == 200
    history_file = history_file_for(tmp_path)
    assert history_file.exists()
    raw_entry = json.loads(history_file.read_text(encoding="utf-8").strip())
    assert raw_entry["response_raw"] == {
        "reasoning": "thinking ",
        "content": "hello from stream",
    }
    assert client.get("/api/logs?page=1&size=10").json()["entries"][0]["content"] == "hello from stream"
