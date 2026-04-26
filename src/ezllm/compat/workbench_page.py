from importlib import resources


def render_workbench_page() -> str:
    return resources.files("ezllm.compat").joinpath("workbench_page.html").read_text(encoding="utf-8")
