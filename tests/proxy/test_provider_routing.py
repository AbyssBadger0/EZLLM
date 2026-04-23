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

    assert rewritten is not request_json
    assert rewritten["model"] == "lm-local"
    assert request_json["model"] == "legacy-display"


def test_registry_tracks_cloud_alias_family_and_active_provider_model():
    registry = build_provider_registry(_settings())

    assert registry.cloud_family_for("or-sonnet") == "sonnet"
    assert registry.model_for_family("sonnet") == "anthropic/claude-sonnet-4.5"


def test_openrouter_registry_uses_legacy_default_base_and_strips_model_prefix():
    settings = SimpleNamespace(
        proxy=SimpleNamespace(local_model_name="lm-local"),
        llama=SimpleNamespace(model_path=r"C:\models\display-model.gguf"),
        aliases=SimpleNamespace(local=[], cloud={"or-sonnet": "sonnet"}),
        providers=AttrDict(
            active="or",
            **{
                "or": {
                    "models": {"sonnet": "openrouter/anthropic/claude-sonnet-4.6"},
                }
            },
        ),
    )

    registry = build_provider_registry(settings)

    assert registry.active_provider is not None
    assert registry.active_provider.base_url == "https://openrouter.ai/api"
    assert registry.model_for_family("sonnet") == "anthropic/claude-sonnet-4.6"


def test_registry_resolves_legacy_sub2_provider_aliases_for_active_provider():
    scenarios = [
        ("cc", "sub2"),
        ("sub2api", "sub2"),
        ("sub2", "cc"),
    ]

    for active_name, provider_key in scenarios:
        settings = SimpleNamespace(
            proxy=SimpleNamespace(local_model_name="lm-local"),
            llama=SimpleNamespace(model_path=r"C:\models\display-model.gguf"),
            aliases=SimpleNamespace(local=[], cloud={}),
            providers=AttrDict(
                active=active_name,
                **{
                    provider_key: {
                        "base_url": "https://sub2.example.test",
                        "models": {"sonnet": "claude-sonnet-4.6"},
                    }
                },
            ),
        )

        registry = build_provider_registry(settings)

        assert registry.active_provider is not None
        assert registry.active_provider.base_url == "https://sub2.example.test"
        assert registry.model_for_family("sonnet") == "claude-sonnet-4.6"
