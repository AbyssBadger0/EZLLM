from types import SimpleNamespace

from ezllm.providers.registry import build_provider_registry
from ezllm.proxy.request_normalizer import rewrite_request_model


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        proxy=SimpleNamespace(local_model_name="lm-local"),
        llama=SimpleNamespace(model_path=r"C:\models\display-model.gguf"),
        aliases=SimpleNamespace(
            local=["legacy-display", "LM-LOCAL-ALIAS"],
            cloud={"or-sonnet": "sonnet"},
        ),
        providers=AttrDict(
            active="openrouter",
            openrouter={
                "base_url": "https://openrouter.example.test/api/v1",
                "models": {"sonnet": "anthropic/claude-sonnet-4.5"},
            },
        ),
    )


def test_rewrite_request_model_normalizes_local_alias_to_canonical_local_model():
    registry = build_provider_registry(_settings())
    request_json = {"model": "legacy-display", "messages": []}

    rewritten = rewrite_request_model(request_json, registry=registry)

    assert rewritten["model"] == "lm-local"
    assert request_json["model"] == "lm-local"


def test_registry_tracks_cloud_alias_family_and_active_provider_model():
    registry = build_provider_registry(_settings())

    assert registry.cloud_family_for("or-sonnet") == "sonnet"
    assert registry.model_for_family("sonnet") == "anthropic/claude-sonnet-4.5"
