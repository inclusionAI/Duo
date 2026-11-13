import json
import os
from pathlib import Path
import re
import sys

from duo.config import ServiceConfig, load_service_config
from duo.models.realization.modules.media_features import read_image
from duo.preprocessing.defaults import DEFAULT_FORCED_ALIGNER, DEFAULT_SPEECH_RECOGNIZER
from duo.runtime.components.checkpoint_schema import (
    intent_identity,
    public_family,
    realization_identity,
    resolve_family,
)
from duo.runtime.components.inference_config import parser
from duo.serving.cases import Cases
from duo.serving.video_state import chunk_view


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_checkpoint_schema_maps_to_public_component_names():
    manifest = {
        "identity": {
            "planner": {"chunk_token": "<chunk_708>"},
            "generator": {"condition_dropout": 0.0},
        },
        "models": {"generator": {"mode": "lora"}},
    }
    assert intent_identity(manifest) == manifest["identity"]["planner"]
    assert realization_identity(manifest) == manifest["identity"]["generator"]
    assert resolve_family(manifest, "realization") == "generator"
    assert public_family("generator") == "realization"


def test_current_checkpoint_schema_takes_precedence_when_both_are_present():
    manifest = {
        "identity": {
            "intent": {"schema": "current"},
            "planner": {"schema": "legacy"},
            "realization": {"schema": "current"},
            "generator": {"schema": "legacy"},
        },
        "models": {"realization": {}, "generator": {}},
    }
    assert intent_identity(manifest)["schema"] == "current"
    assert realization_identity(manifest)["schema"] == "current"
    assert resolve_family(manifest, "realization") == "realization"


def test_public_runtime_exposes_only_declared_attention_backend():
    action = parser("dmd")._option_string_actions["--attention"]
    assert action.default == "sdpa"
    assert action.choices == ("sdpa",)


def test_default_speech_models_are_reproducible_public_dependencies():
    config = load_service_config(ROOT / "configs" / "inference.example.json")
    assert config.weights.speech_recognizer == "weights/speech/recognizer"
    assert config.weights.forced_aligner == "weights/speech/forced-aligner"
    assert DEFAULT_SPEECH_RECOGNIZER == config.weights.speech_recognizer
    assert DEFAULT_FORCED_ALIGNER == config.weights.forced_aligner
    dependencies = (ROOT / "pyproject.toml").read_text()
    assert "qwen-asr==0.0.6" in dependencies
    assert "torch==2.11.0" in dependencies


def test_public_and_service_clocks_are_the_same_seventeen_frame_chunk():
    from duo.config import TimelineConfig
    from duo.models.realization.modules.codec_spec import spec
    from duo.serving.common import FPS, FRAMES

    clock = TimelineConfig()
    assert (clock.frames_per_second, clock.frames_per_block) == (FPS, FRAMES) == (24, 17)
    codec = spec("duo_5_17")
    assert (codec.latents, codec.frames) == (5, 17)


def test_causal_sampling_defaults_to_fifty_steps():
    from duo.serving.models import _inference

    config = load_service_config(ROOT / "configs" / "inference.example.json")
    assert _inference({"stage": "causal"}, config) == (50, 1.0)
    assert _inference({"stage": "dmd"}, config) == (4, 5.0)
    assert _inference(
        {"stage": "causal", "inference": {"sampling_steps": 32, "flow_shift": 2.0}},
        config,
    ) == (32, 2.0)


def test_readme_surfaces_release_status_and_runtime_links():
    readme = (ROOT / "README.md").read_text()
    for text in (
        "## 🔥 News",
        "Publish the [Duo paper on arXiv](https://arxiv.org/abs/2611.15131)",
        "`2026/11/13`: Release **Duo v1**",
        "offline inference code",
        "inference/vllm-omni-duo",
        "https://github.com/inclusionAI/Duo/tree/inference/vllm-omni-duo",
        "vLLM-Omni](https://github.com/inclusionAI/Duo/tree/inference/vllm-omni-duo) (Recommended)",
        "## 📦 Model Releases",
        "## 🤗 Download Checkpoints",
        "@misc{cao2026duoreciprocalworldmodel,",
        "eprint        = {2611.15131}",
    ):
        assert text in readme


def test_example_config_has_no_machine_specific_paths():
    value = json.loads((ROOT / "configs" / "inference.example.json").read_text())
    rendered = json.dumps(value)
    assert "/xboxtrain" not in rendered
    assert "/home/" not in rendered
    assert value["weights"]["realization"] == "weights/realization"
    assert value["weights"]["realization_family"] == "auto"
    assert value["weights"]["speech_recognizer"] == "weights/speech/recognizer"


def test_runtime_resources_are_packaged():
    project = (ROOT / "pyproject.toml").read_text()
    for resource in (
        "tables/*.pt",
        "native_vae/*.yaml",
        "sana/*.json",
        "LICENSE.diffusers",
        '"duo.runtime" = ["LICENSE"]',
    ):
        assert resource in project
    assert (ROOT / "duo/runtime/components/inferencer.py").is_file()


def test_case_uses_one_still_assistant_reference(tmp_path):
    cases = Cases(
        settings_path=tmp_path / "settings.json",
        catalog_path=ROOT / "configs" / "cases" / "catalog.json",
    )
    raw = json.loads(cases.catalog_path.read_text())[0]
    assert "assistant_video" not in raw
    assert cases.assistant_image("default") == ROOT / "demo/assets/assistant-reference.jpg"
    assert cases.public()[0]["assistant_image"] == "/api/cases/default/reference-image"
    image = read_image(cases.assistant_image("default"), 576, 768)
    assert image.shape == (576, 768, 3)
    assert str(image.dtype) == "uint8"


def test_relative_virtualenv_executable_keeps_its_symlink(tmp_path):
    link = tmp_path / "env" / "bin" / "python"
    link.parent.mkdir(parents=True)
    link.symlink_to(Path(sys.executable))
    config = ServiceConfig(
        project_root=tmp_path,
        config_path=tmp_path / "config.json",
        runtime_dir=tmp_path / "runtime",
        case_catalog=tmp_path / "cases.json",
        default_case="default",
    )
    assert config.executable("env/bin/python") == str(link)


def test_public_video_state_uses_generic_stage_names():
    chunk = {
        "global": 0,
        "requestId": 1,
        "user": {"url": "./recordings/example.mp4"},
        "intent_states": "ready",
        "dit": "running",
        "vae": "pending",
        "intent_state_rows": [1, 2],
        "steps": 4,
        "dit_metrics": {},
    }
    public = chunk_view(chunk)
    assert (public["intent"], public["realization"], public["codec"]) == (
        "ready",
        "running",
        "pending",
    )
    for private in ("intent_states", "dit", "vae", "intent_state_rows", "dit_metrics"):
        assert private not in public


def test_release_sources_do_not_contain_private_paths_or_experiment_labels():
    suffixes = {".py", ".js", ".html", ".json", ".md", ".toml", ".yaml"}
    ignored = {Path(__file__), ROOT / "configs" / "inference.json"}
    files = []
    for directory, names, filenames in os.walk(ROOT):
        names[:] = [name for name in names if name not in {".git", "runtime", "__pycache__"}]
        files.extend(
            path
            for name in filenames
            if (path := Path(directory) / name).suffix in suffixes and path not in ignored
        )
    text = "\n".join(path.read_text(errors="ignore") for path in files)
    for pattern in (
        r"/xboxtrain",
        r"qwenvl",
        r"qwen-vl",
        r"experiment[-_ ]?\d+",
        r"exp[-_ ]?\d+",
        r"step[-_ ]?\d{2,}",
        r"world response",
        r"\bh50\b",
        r"duoinfer",
        r"alipay\.com",
        r"antgroup\.com",
    ):
        assert re.search(pattern, text, re.IGNORECASE) is None
