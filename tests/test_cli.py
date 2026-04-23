import importlib


def test_cli_module_exposes_app():
    module = importlib.import_module("ezllm.cli")

    assert hasattr(module, "app")
