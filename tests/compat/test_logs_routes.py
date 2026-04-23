from fastapi.testclient import TestClient

from ezllm.proxy.app import build_app


def test_logs_page_serves_legacy_markup(tmp_path):
    app = build_app(log_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/logs")

    assert response.status_code == 200
    assert "LM Core Monitor" in response.text
    assert 'id="logs"' in response.text
    assert "/api/logs?page=${state.page}&size=${state.size}" in response.text


def test_api_logs_returns_legacy_shape(tmp_path):
    history = tmp_path / "chat_history.jsonl"
    history.write_text(
        '{"timestamp":"2026-04-23T00:00:00Z","duration_sec":0.1,"path":"/v1/chat/completions","messages":[],"reasoning":"","content":"ok","request_raw":{},"response_raw":{},"upstream":"local:openai"}\n',
        encoding="utf-8",
    )
    app = build_app(log_dir=tmp_path)
    client = TestClient(app)

    payload = client.get("/api/logs?page=1&size=10").json()

    assert payload["page"] == 1
    assert payload["size"] == 10
    assert payload["total"] == 1
    assert payload["pages"] == 1
    assert payload["entries"][0]["content"] == "ok"
