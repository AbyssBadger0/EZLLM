def to_legacy_logs_payload(page_payload: dict) -> dict:
    return {
        "page": page_payload["page"],
        "size": page_payload["size"],
        "total": page_payload["total"],
        "pages": page_payload["pages"],
        "entries": page_payload["entries"],
    }
