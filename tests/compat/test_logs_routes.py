import json
from pathlib import Path

from fastapi.testclient import TestClient

from ezllm.proxy.app import build_app


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _extract_legacy_logs_html() -> str:
    source = Path("C:/Users/abyss/GraphiteUI/scripts/lm_core0.py").read_text(encoding="utf-8")
    marker = 'return HTMLResponse("""'
    start = source.index(marker) + len(marker)
    end = source.index('""")', start)
    return source[start:end]


def test_logs_page_serves_exact_legacy_markup(tmp_path):
    app = build_app(log_dir=tmp_path)
    client = TestClient(app)

    response = client.get("/logs")

    assert response.status_code == 200
    assert response.text == _extract_legacy_logs_html()


def test_api_logs_returns_legacy_projected_shape_in_reverse_order(tmp_path):
    history = tmp_path / "chat_history.jsonl"
    fixture_text = (FIXTURES_DIR / "log_entries.jsonl").read_text(encoding="utf-8")
    fixture_entries = [json.loads(line) for line in fixture_text.splitlines() if line.strip()]
    history.write_text(fixture_text, encoding="utf-8")
    app = build_app(log_dir=tmp_path)
    client = TestClient(app)

    payload = client.get("/api/logs?page=1&size=10").json()

    assert payload["page"] == 1
    assert payload["size"] == 10
    assert payload["total"] == 2
    assert payload["pages"] == 1

    second = fixture_entries[1]
    entry = payload["entries"][0]
    assert entry["timestamp"] == second["timestamp"]
    assert entry["request_model"] == "gpt-compat"
    assert entry["request_kind"] == "chat"
    assert entry["reasoning"] == "latest reasoning"
    assert entry["content"] == "latest output"
    assert entry["request_raw"] == second["request_raw"]
    assert entry["response_raw"] == second["response_raw"]
    assert entry["messages"] == [
        {"role": "system", "body": "System instructions"},
        {"role": "user", "body": "latest question"},
        {"role": "assistant", "body": "latest prior answer"},
    ]


def test_api_logs_returns_zero_pages_when_history_is_empty(tmp_path):
    app = build_app(log_dir=tmp_path)
    client = TestClient(app)

    payload = client.get("/api/logs?page=1&size=10").json()

    assert payload["entries"] == []
    assert payload["total"] == 0
    assert payload["pages"] == 0
