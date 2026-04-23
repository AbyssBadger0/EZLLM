import json

from ezllm.logs.store import append_raw_log, history_file_for, read_all_logs, save_raw_log
from ezllm.proxy.response_normalizer import parse_openai_payload_for_log


def test_append_raw_log_and_read_all_logs_round_trip_legacy_jsonl_entries(tmp_path):
    history_file = history_file_for(tmp_path)
    entry = {
        "timestamp": "2026-04-24 00:00:00",
        "duration_sec": 0.13,
        "path": "/v1/chat/completions",
        "upstream": "local:openai",
        "request_raw": {"model": "lm-local", "messages": []},
        "response_raw": {"reasoning": "", "content": "done"},
    }

    append_raw_log(history_file=history_file, entry=entry)

    raw_entry = json.loads(history_file.read_text(encoding="utf-8").strip())
    assert raw_entry == entry
    assert read_all_logs(history_file) == [entry]


def test_save_raw_log_persists_legacy_raw_request_and_response_keys(tmp_path):
    request_json = {
        "model": "lm-local",
        "messages": [{"role": "user", "content": "hello"}],
    }

    save_raw_log(
        log_dir=tmp_path,
        req_j=request_json,
        reasoning="step by step",
        content="done",
        duration=0.126,
        path="/v1/chat/completions",
        upstream="local:openai",
        timestamp="2026-04-24 00:00:00",
    )

    history_file = history_file_for(tmp_path)
    assert history_file.exists()

    raw_entry = json.loads(history_file.read_text(encoding="utf-8").strip())
    assert "request_raw" in raw_entry
    assert "response_raw" in raw_entry

    entries = read_all_logs(history_file)
    assert entries == [
        {
            "timestamp": "2026-04-24 00:00:00",
            "duration_sec": 0.13,
            "path": "/v1/chat/completions",
            "upstream": "local:openai",
            "request_raw": request_json,
            "response_raw": {
                "reasoning": "step by step",
                "content": "done",
            },
        }
    ]


def test_parse_openai_payload_for_log_ignores_malformed_json_shapes_without_raising():
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    parse_openai_payload_for_log(
        {"choices": [None]},
        reasoning_parts,
        content_parts,
    )

    assert reasoning_parts == []
    assert content_parts == []
