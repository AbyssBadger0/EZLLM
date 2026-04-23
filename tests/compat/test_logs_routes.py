import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ezllm.proxy.app import build_app


FIXTURES_DIR = Path(__file__).parent / "fixtures"
LOGS_PAGE_ASSET = Path(__file__).resolve().parents[2] / "assets" / "logs_page" / "index.html"


def _build_client(tmp_path: Path) -> TestClient:
    return TestClient(build_app(log_dir=tmp_path))


def _write_history(tmp_path: Path, entries: list[dict] | None = None, *, raw_text: str | None = None) -> None:
    history = tmp_path / "chat_history.jsonl"
    if raw_text is not None:
        history.write_text(raw_text, encoding="utf-8")
        return
    payload = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in (entries or []))
    history.write_text(payload, encoding="utf-8")


def test_logs_page_serves_exact_frozen_markup(tmp_path):
    client = _build_client(tmp_path)

    response = client.get("/logs")

    assert response.status_code == 200
    assert response.text == LOGS_PAGE_ASSET.read_text(encoding="utf-8")


def test_api_logs_returns_legacy_projected_shape_in_reverse_order(tmp_path):
    fixture_text = (FIXTURES_DIR / "log_entries.jsonl").read_text(encoding="utf-8")
    fixture_entries = [json.loads(line) for line in fixture_text.splitlines() if line.strip()]
    _write_history(tmp_path, raw_text=fixture_text)
    client = _build_client(tmp_path)

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


def test_api_logs_renders_metadata_tagged_user_content(tmp_path):
    _write_history(
        tmp_path,
        entries=[
            {
                "timestamp": "2026-04-24T00:00:00Z",
                "duration_sec": 0.1,
                "path": "/v1/chat/completions",
                "upstream": "local:openai",
                "request_raw": {
                    "model": "meta-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": 'Sender (untrusted metadata): ```json {"username":"Alice"} ```\n[2026-04-24 09:30] hello',
                        }
                    ],
                },
                "response_raw": {"reasoning": "", "content": "ok"},
            }
        ],
    )
    client = _build_client(tmp_path)

    entry = client.get("/api/logs?page=1&size=10").json()["entries"][0]
    body = entry["messages"][0]["body"]

    assert "Alice" in body
    assert "2026-04-24 09:30" in body
    assert "hello" in body
    assert "<div" in body


def test_api_logs_detects_count_tokens_request_kind(tmp_path):
    _write_history(
        tmp_path,
        entries=[
            {
                "timestamp": "2026-04-24T00:00:00Z",
                "duration_sec": 0.1,
                "path": "/v1/count_tokens",
                "upstream": "local:openai",
                "request_raw": {"model": "counter", "messages": []},
                "response_raw": {"reasoning": "", "content": "7"},
            }
        ],
    )
    client = _build_client(tmp_path)

    entry = client.get("/api/logs?page=1&size=10").json()["entries"][0]

    assert entry["request_kind"] == "count-tokens"


def test_api_logs_detects_session_title_requests_from_system_prompt(tmp_path):
    _write_history(
        tmp_path,
        entries=[
            {
                "timestamp": "2026-04-24T00:00:00Z",
                "duration_sec": 0.1,
                "path": "/v1/messages",
                "upstream": "local:anthropic",
                "request_raw": {
                    "model": "titler",
                    "system": 'Generate a concise, sentence-case title and reply with JSON containing a "title" field.',
                    "messages": [],
                },
                "response_raw": {"reasoning": "", "content": '{"title":"Hello"}'},
            }
        ],
    )
    client = _build_client(tmp_path)

    entry = client.get("/api/logs?page=1&size=10").json()["entries"][0]

    assert entry["request_kind"] == "session-title"


def test_api_logs_skips_malformed_jsonl_lines_without_failing(tmp_path):
    valid_entry = {
        "timestamp": "2026-04-24T00:00:00Z",
        "duration_sec": 0.1,
        "path": "/v1/chat/completions",
        "upstream": "local:openai",
        "request_raw": {"model": "good-model", "messages": []},
        "response_raw": {"reasoning": "", "content": "survives"},
    }
    _write_history(
        tmp_path,
        raw_text="{not json}\n" + json.dumps(valid_entry, ensure_ascii=False),
    )
    client = _build_client(tmp_path)

    payload = client.get("/api/logs?page=1&size=10").json()

    assert payload["total"] == 2
    assert payload["entries"] == [
        {
            "timestamp": "2026-04-24T00:00:00Z",
            "duration_sec": 0.1,
            "path": "/v1/chat/completions",
            "upstream": "local:openai",
            "request_model": "good-model",
            "request_kind": "chat",
            "messages": [],
            "reasoning": "",
            "content": "survives",
            "request_raw": {"model": "good-model", "messages": []},
            "response_raw": {"reasoning": "", "content": "survives"},
        }
    ]


def test_api_logs_out_of_range_page_returns_empty_entries_with_stable_metadata(tmp_path):
    fixture_text = (FIXTURES_DIR / "log_entries.jsonl").read_text(encoding="utf-8")
    _write_history(tmp_path, raw_text=fixture_text)
    client = _build_client(tmp_path)

    payload = client.get("/api/logs?page=3&size=1").json()

    assert payload["page"] == 3
    assert payload["size"] == 1
    assert payload["total"] == 2
    assert payload["pages"] == 2
    assert payload["entries"] == []


def test_api_logs_size_zero_matches_legacy_failure_behavior(tmp_path):
    fixture_text = (FIXTURES_DIR / "log_entries.jsonl").read_text(encoding="utf-8")
    _write_history(tmp_path, raw_text=fixture_text)
    client = _build_client(tmp_path)

    with pytest.raises(ZeroDivisionError):
        client.get("/api/logs?page=1&size=0")


def test_api_logs_returns_zero_pages_when_history_is_empty(tmp_path):
    client = _build_client(tmp_path)

    payload = client.get("/api/logs?page=1&size=10").json()

    assert payload["entries"] == []
    assert payload["total"] == 0
    assert payload["pages"] == 0
