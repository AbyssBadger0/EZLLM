from importlib import resources


def render_control_page() -> str:
    return resources.files("ezllm.compat").joinpath("control_page.html").read_text(encoding="utf-8")
