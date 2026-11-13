# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Create a new, self-contained runtime configuration for Duo."""

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path
from string import Template
from tempfile import TemporaryDirectory

import yaml
from PIL import Image, ImageOps

p = argparse.ArgumentParser(description=__doc__)
for name in (
    "runtime-dir",
    "model-root",
    "intent-adapter",
    "intent-protocol",
    "asr-model",
    "reference-image",
    "reference-latent",
    "engine-python",
    "web-python",
):
    p.add_argument("--" + name, type=Path, required=True)
realization = p.add_mutually_exclusive_group(required=True)
realization.add_argument("--realization-adapter", type=Path, help="Fixed adapter for the BF16 Realization base")
realization.add_argument("--realization-dcp", type=Path, help="Duo release directory or its realization/DCP directory")
realization.add_argument(
    "--fp8-transformer", type=Path, help="Directory containing the serialized ModelOpt FP8 transformer"
)
p.add_argument(
    "--clean-history-chunks", type=int, help="Maximum previous AV chunks visible to the Realization clean pass"
)
p.add_argument("--asr-overlap-previous-chunk", action=argparse.BooleanOptionalAction, default=True)
p.add_argument(
    "--intent-output-repeat-penalty",
    type=float,
    default=2.0,
    help="Recent assistant bigram logit penalty; 0 disables",
)
p.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
p.add_argument("--fp8-profiles", help="Profile path pattern containing {rank}")
p.add_argument("--fa3-library", type=Path, help="Optional full-API FA3 shared library for FP8")
p.add_argument("--sample-dir", type=Path)
p.add_argument("--recordings-dir", type=Path, help="Directory for call recordings (default: runtime/recordings)")
p.add_argument(
    "--personas-dir",
    type=Path,
    help="Directory containing personas.json and portraits/<key>/{portrait.jpg,reference.pt}",
)
p.add_argument("--web-host", default="127.0.0.1")
p.add_argument("--web-port", type=int, default=18080)
p.add_argument("--backend-port", type=int, default=8091)
p.add_argument("--defer-clean", action=argparse.BooleanOptionalAction, default=True)
p.add_argument("--joint-execution", action="store_true", help="Enable joint mode (requires --no-defer-clean)")
p.add_argument("--decode-backend", choices=("native", "solarx"), default="solarx")
p.add_argument(
    "--reference-window-chunks", type=int, default=14, help="Window including current; 0 uses only the token budget"
)
p.add_argument("--alignment-python", type=Path)
p.add_argument("--alignment-pythonpath", default="")
p.add_argument("--alignment-model", type=Path)
a = p.parse_args()
if a.joint_execution and a.defer_clean:
    p.error("--joint-execution requires --no-defer-clean")
if a.clean_history_chunks is not None and (a.clean_history_chunks < 0 or a.joint_execution):
    p.error("--clean-history-chunks requires a nonnegative count and separate clean execution")
if not 0 <= a.intent_output_repeat_penalty <= 10:
    p.error("--intent-output-repeat-penalty must be between 0 and 10")
if a.precision == "fp8" and not (a.fp8_profiles or a.fp8_transformer):
    p.error("--precision fp8 requires --fp8-profiles")
if a.fa3_library and not a.fp8_profiles:
    p.error("--fa3-library requires --fp8-profiles")
if a.fa3_library and not a.fa3_library.is_file():
    p.error("--fa3-library must be an existing file")
if a.fp8_profiles and "{rank}" not in a.fp8_profiles:
    p.error("--fp8-profiles must contain {rank}")
if bool(a.alignment_python) != bool(a.alignment_model):
    p.error("--alignment-python and --alignment-model must be provided together")
if a.reference_window_chunks < 0:
    p.error("--reference-window-chunks must be nonnegative")
for name in (
    "model_root",
    "intent_adapter",
    "intent_protocol",
    "realization_adapter",
    "realization_dcp",
    "fp8_transformer",
    "asr_model",
    "reference_image",
    "reference_latent",
    "engine_python",
    "web_python",
    "sample_dir",
    "personas_dir",
    "alignment_python",
    "alignment_model",
):
    path = getattr(a, name)
    if path is not None and not path.exists():
        p.error(f"--{name.replace('_', '-')} does not exist: {path}")
if a.fp8_transformer:
    quant_config = json.loads((a.fp8_transformer / "config.json").read_text()).get("quantization_config")
    if not quant_config or quant_config.get("quant_algo") != "FP8_PER_CHANNEL_PER_TOKEN":
        p.error("--fp8-transformer must provide FP8_PER_CHANNEL_PER_TOKEN quantization_config in config.json")

root = Path(__file__).resolve().parent
repo = root.parents[1]
runtime = a.runtime_dir.resolve()
if a.realization_dcp:
    from duo_dcp import export_dcp_adapter, read_dcp_manifest

    if runtime.exists():
        p.error(f"--runtime-dir already exists: {runtime}")
    try:
        realization, manifest = read_dcp_manifest(a.realization_dcp)
        dcp_steps = manifest["inference"]["sampling_steps"]
        if a.joint_execution and dcp_steps != 4:
            p.error("--joint-execution requires a four-step DCP")
        runtime.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=runtime.parent) as temporary:
            exported = Path(temporary) / "realization_adapter.safetensors"
            dcp_modules = export_dcp_adapter(realization, manifest, a.model_root / "Ref2VA" / "transformer", exported)
            runtime.mkdir()
            a.realization_adapter = runtime / exported.name
            exported.replace(a.realization_adapter)
    except (OSError, ValueError, KeyError) as exc:
        p.error(f"--realization-dcp: {exc}")
else:
    runtime.mkdir(parents=True, exist_ok=False)
shutil.copytree(root / "web", runtime / "web")
(runtime / "assets").mkdir()
(runtime / "personas").mkdir()
reference_image = runtime / "assets" / ("reference" + a.reference_image.suffix)
reference_latent = runtime / "assets" / "reference.pt"
shutil.copyfile(a.reference_image, reference_image)
shutil.copyfile(a.reference_latent, reference_latent)
portraits = runtime / "web" / "portraits"
if a.personas_dir:
    presets = json.loads((a.personas_dir / "personas.json").read_text())
    if not isinstance(presets, dict) or not presets:
        p.error("personas.json must contain at least one preset")
    for key, preset in presets.items():
        if not key or Path(key).name != key or key in (".", ".."):
            p.error("preset keys must be simple directory names")
        if (
            not isinstance(preset, dict)
            or not preset.get("label")
            or not (
                preset.get("system_prompt") or all(preset.get(k) for k in ("appearance", "environment", "initial_pose"))
            )
        ):
            p.error(f"preset {key!r} needs a label and system_prompt or three description fields")
        for file in ("portrait.jpg", "reference.pt"):
            if not (a.personas_dir / "portraits" / key / file).is_file():
                p.error(f"preset {key!r} is missing {file}")
    shutil.copyfile(a.personas_dir / "personas.json", runtime / "web" / "personas.json")
    shutil.copytree(a.personas_dir / "portraits", portraits, dirs_exist_ok=True)
else:
    (portraits / "default").mkdir(parents=True)
    with Image.open(a.reference_image) as image:
        ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (576, 768)).save(
            portraits / "default" / "portrait.jpg", quality=94
        )
    shutil.copyfile(reference_latent, portraits / "default" / "reference.pt")
paths = dict(
    RUNTIME_DIR=str(runtime),
    MODEL_ROOT=str(a.model_root.resolve()),
    INTENT_ADAPTER=str(a.intent_adapter.resolve()),
    INTENT_PROTOCOL=str(a.intent_protocol.resolve()),
    REALIZATION_ADAPTER=str(a.realization_adapter.resolve()) if a.realization_adapter else "",
    ASR_MODEL=str(a.asr_model.resolve()),
)
session_values = {k: json.dumps(v)[1:-1] for k, v in paths.items()}
(runtime / "session.json").write_text(Template((root / "session.template.json").read_text()).substitute(session_values))


def render_yaml(value):
    if isinstance(value, str):
        return Template(value).substitute(paths)
    if isinstance(value, list):
        return [render_yaml(item) for item in value]
    if isinstance(value, dict):
        return {key: render_yaml(item) for key, item in value.items()}
    return value


deployment = render_yaml(yaml.safe_load((root / "deploy.template.yaml").read_text()))
session = json.loads((runtime / "session.json").read_text())
session.update(
    reference_image=str(reference_image),
    reference_latent=str(reference_latent),
    reference_image_sha256=hashlib.sha256(reference_image.read_bytes()).hexdigest(),
    realization_defer_clean_until_codec=a.defer_clean,
    realization_joint_execution=a.joint_execution,
    asr_overlap_previous_chunk=a.asr_overlap_previous_chunk,
    intent_output_repeat_penalty=a.intent_output_repeat_penalty,
)
if a.realization_dcp:
    session["realization_num_inference_steps"] = dcp_steps
if a.joint_execution:
    session["realization_scheduling"] = "free"
(runtime / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n")
stage = deployment["stages"][2]
realization_config = stage["model_config"]["duo"]
realization_config.update(
    defer_clean_until_codec=a.defer_clean,
    joint_execution=a.joint_execution,
    fp8_gemm=a.precision == "fp8" and a.fp8_transformer is None,
)
if a.realization_dcp:
    realization_config.update(
        sampling_stage=manifest["stage"], flow_shift=manifest["inference"]["flow_shift"], adapter_modules=dcp_modules
    )
    stage["default_sampling_params"]["num_inference_steps"] = dcp_steps
    if dcp_steps != 4:
        realization_config.update(fused_denoise=False, reuse_static_text=False)
        realization_config.pop("text_graph_buckets", None)
if a.fp8_transformer:
    model = runtime / "realization"
    model.mkdir()
    (model / "transformer").symlink_to(a.fp8_transformer.resolve(), target_is_directory=True)
    shutil.copyfile(a.model_root / "Ref2VA" / "model_index.json", model / "model_index.json")
    stage.update(model=str(model), quantization_config=quant_config)
    realization_config.pop("adapter_path")
    realization_config["merge_fixed_adapter"] = False
if a.clean_history_chunks is not None:
    realization_config["clean_history_chunks"] = a.clean_history_chunks
if a.joint_execution:
    deployment["stages"][2]["max_num_seqs"] = 2
    realization_config.pop("text_graph_buckets", None)
deployment["stages"][3]["model_config"]["duo_codec"].update(next_block_lookahead=False, decode_backend=a.decode_backend)
if a.reference_window_chunks:
    realization_config["reference_window_chunks"] = a.reference_window_chunks
else:
    realization_config.pop("reference_window_chunks", None)
(runtime / "deploy.yaml").write_text(yaml.safe_dump(deployment, sort_keys=False))
preset_path = runtime / "web" / "preset.json"
preset = json.loads(preset_path.read_text())
preset["model"] = paths["ASR_MODEL"]
preset_path.write_text(json.dumps(preset, ensure_ascii=False, indent=2) + "\n")
env = dict(
    DUO_REPO=str(repo),
    DUO_RUNTIME_DIR=str(runtime),
    DUO_MODEL_ROOT=paths["MODEL_ROOT"],
    DUO_ASR_MODEL=paths["ASR_MODEL"],
    DUO_ENGINE_PYTHON=str(a.engine_python.absolute()),
    DUO_WEB_PYTHON=str(a.web_python.absolute()),
    DUO_ENGINE_PYTHONPATH=str(repo),
    DUO_PRECISION=a.precision,
    DUO_ENGINE_URL="http://127.0.0.1:" + str(a.backend_port),
    DUO_BACKEND_PORT=str(a.backend_port),
    DUO_WEB_HOST=a.web_host,
    DUO_WEB_PORT=str(a.web_port),
    DUO_RECORDINGS_DIR=str(a.recordings_dir.resolve() if a.recordings_dir else runtime / "recordings"),
)
if a.sample_dir:
    env["DUO_SAMPLE_DIR"] = str(a.sample_dir.resolve())
if a.fp8_profiles:
    env["DUO_FP8_PROFILES"] = str(Path(a.fp8_profiles).absolute())
if a.fa3_library:
    env["DUO_FA3_LIBRARY"] = str(a.fa3_library.resolve())
if a.alignment_python:
    env.update(
        DUO_ALIGNMENT_PYTHON=str(a.alignment_python.absolute()),
        DUO_ALIGNMENT_PYTHONPATH=a.alignment_pythonpath,
        DUO_ALIGNMENT_MODEL=str(a.alignment_model.resolve()),
        DUO_INTENT_ADAPTER=paths["INTENT_ADAPTER"],
    )
(runtime / "environment.sh").write_text("\n".join("export " + k + "=" + shlex.quote(v) for k, v in env.items()) + "\n")
for name in ("serve.sh", "run-web.sh"):
    shutil.copy2(root / name, runtime / name)
print(runtime)
