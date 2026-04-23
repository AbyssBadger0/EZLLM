import json

from ezllm.logs.store import history_file_for, read_history_entries, save_raw_log


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

    entries = read_history_entries(history_file)
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
