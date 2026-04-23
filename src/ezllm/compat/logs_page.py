from pathlib import Path


def render_logs_page() -> str:
    asset_path = Path(__file__).resolve().parents[3] / "assets" / "logs_page" / "index.html"
    return asset_path.read_text(encoding="utf-8")
