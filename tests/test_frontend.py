from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


def test_frontend_contains_full_video_workflow():
    html = (DEMO / "index.html").read_text()
    assert '<html lang="en">' in html
    for element_id in (
        "camera-preview",
        "video-file",
        "people-list",
        "assistant-track",
        "user-track",
        "assistant-layer",
        "user-layer",
        "intent-grid",
        "realization-grid",
        "codec-grid",
    ):
        assert f'id="{element_id}"' in html


def test_frontend_uses_runtime_catalog_and_generic_stage_names():
    app = (DEMO / "app.js").read_text()
    client = (DEMO / "client.js").read_text()
    assert "/api/frontend-catalog" in app
    assert "catalog.schema" in app
    assert "intent" in client
    assert "realization" in client
    assert "codec" in client


def test_frontend_does_not_embed_private_names_or_experiment_labels():
    sources = "\n".join(
        path.read_text() for path in sorted(DEMO.glob("*")) if path.suffix in {".html", ".js"}
    )
    forbidden = (
        r"experiment[-_ ]?\d+",
        r"exp[-_ ]?\d+",
        r"step[-_ ]?\d{2,}",
        r"(?:^|[\"'])/(?:home|mnt|workspace)/",
    )
    for pattern in forbidden:
        assert re.search(pattern, sources, re.IGNORECASE) is None


def test_interface_copy_is_english_outside_case_content():
    sources = "\n".join(
        path.read_text() for path in sorted(DEMO.glob("*")) if path.suffix in {".html", ".js"}
    )
    assert re.search(r"[\u4e00-\u9fff]", sources) is None
    assert "Case 4" not in sources
    assert "原有" not in sources


def test_catalog_has_only_a_generic_assistant_reference():
    catalog = (DEMO / "catalog.json").read_text()
    docs = (ROOT / "docs" / "frontend.md").read_text()
    assert "case_4" not in catalog + docs
    assert "user_image" not in catalog + docs
    assert "assistant_video" not in catalog + docs
    assert (DEMO / "assets" / "assistant-reference.jpg").is_file()
    assert not any(DEMO.glob("assets/*user*.jpg"))


def test_frontend_uses_chunk_terminology_in_visible_copy():
    sources = "\n".join(
        path.read_text() for path in sorted(DEMO.glob("*")) if path.suffix in {".html", ".js"}
    )
    visible_block_terms = (
        "Block alignment",
        "block alignment",
        "Block details",
        "Global block",
        "Current block",
        "blocks ready",
        "blocks generated",
        "blocks ·",
        ">blocks<",
    )
    assert not any(term in sources for term in visible_block_terms)
