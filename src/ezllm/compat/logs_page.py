from importlib import resources


def render_logs_page() -> str:
    return resources.files("ezllm.compat").joinpath("logs_page.html").read_text(encoding="utf-8")
