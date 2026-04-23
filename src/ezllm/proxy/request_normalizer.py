from ezllm.providers.registry import ProviderRegistry


def get_request_model(request_json: dict | None) -> str:
    if not isinstance(request_json, dict):
        return ""
    model = request_json.get("model")
    return model.strip() if isinstance(model, str) else ""


def should_route_to_local(request_json: dict | None, *, registry: ProviderRegistry) -> bool:
    model = get_request_model(request_json)
    if not model:
        return True
    return registry.is_local_alias(model)


def should_route_to_cloud(request_json: dict | None, *, registry: ProviderRegistry) -> bool:
    model = get_request_model(request_json)
    return registry.cloud_family_for(model) is not None


def rewrite_request_model(request_json: dict | None, *, registry: ProviderRegistry) -> dict | None:
    if not isinstance(request_json, dict):
        return request_json

    payload = dict(request_json)
    model = get_request_model(payload)
    if model and registry.is_local_alias(model):
        payload["model"] = registry.local_model_name
        return payload

    family = registry.cloud_family_for(model)
    if family:
        rewritten_model = registry.model_for_family(family)
        if rewritten_model:
            payload["model"] = rewritten_model

    return payload
