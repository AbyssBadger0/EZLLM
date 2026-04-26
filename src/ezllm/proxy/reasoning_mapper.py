import json
from typing import Any


EFFORT_ALIASES = {
    "0": "off",
    "false": "off",
    "disabled": "off",
    "none": "off",
    "off": "off",
    "minimal": "minimal",
    "minimum": "minimal",
    "min": "minimal",
    "low": "low",
    "medium": "medium",
    "normal": "medium",
    "high": "high",
    "extra-high": "xhigh",
    "extra_high": "xhigh",
    "extra high": "xhigh",
    "xhigh": "xhigh",
    "max": "xhigh",
    "maximum": "xhigh",
    "auto": "auto",
}

EFFORT_BUDGETS = {
    "minimal": 512,
    "low": 2048,
    "medium": 8192,
    "high": 32768,
    "xhigh": -1,
}

TEMPLATE_EFFORTS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


def _normalize_effort(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "medium" if value else "off"
    text = str(value).strip().lower()
    return EFFORT_ALIASES.get(text)


def _extract_unified_effort(payload: dict[str, Any]) -> str:
    reasoning = payload.get("reasoning")
    if isinstance(reasoning, dict):
        effort = _normalize_effort(reasoning.get("effort"))
        if effort:
            return effort
    else:
        effort = _normalize_effort(reasoning)
        if effort:
            return effort

    return _normalize_effort(payload.get("reasoning_effort")) or "off"


def _copy_payload(body: bytes) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _template_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    raw_kwargs = payload.get("chat_template_kwargs")
    return dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}


def map_unified_reasoning_for_llama(body: bytes) -> bytes:
    payload = _copy_payload(body)
    if payload is None:
        return body

    effort = _extract_unified_effort(payload)
    if not effort or effort == "auto":
        return body

    payload.pop("reasoning", None)
    payload.pop("reasoning_effort", None)

    template_kwargs = _template_kwargs(payload)
    if effort == "off":
        template_kwargs["enable_thinking"] = False
        payload["thinking_budget_tokens"] = 0
    else:
        template_kwargs["enable_thinking"] = True
        template_kwargs["reasoning_effort"] = TEMPLATE_EFFORTS[effort]
        payload["thinking_budget_tokens"] = EFFORT_BUDGETS[effort]

    payload["chat_template_kwargs"] = template_kwargs
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
