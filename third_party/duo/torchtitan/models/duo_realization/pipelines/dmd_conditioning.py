"""Training-only mixed GT/NTP conditions; no Realization optimizer or GAN dependencies."""

import hashlib
import json
from copy import copy, deepcopy
from pathlib import Path

POLICY = "dmd_window_gt_then_ntp_shared_v4"


def _sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def runtime_config(manifest, feature_config=None):
    """Resolve recorded sources instead of inheriting inference's old defaults."""
    p, e = manifest.get("intent_provenance", {}), manifest.get("extraction_config", {})
    if e.get("text_time_mode", "planning") != "planning":
        raise ValueError("Intent rollout requires planning-mode cached text")
    if p.get("h50_layer") != 50 or not p.get("base_model"):
        raise ValueError("Intent rollout requires complete feature Intent provenance")
    if p.get("kind") not in ("finetuned", "origin"):
        raise ValueError("Unknown Intent provenance kind")
    protocol = (feature_config or {}).get("intent_protocol_manifest", "")
    protocol = (
        protocol
        or e.get("intent_protocol_manifest", "")
        or p.get("protocol_manifest_path", "")
    )
    value = dict(
        intent=p["base_model"],
        intent_adapter=p.get("adapter_path", ""),
        origin_intent=p["kind"] == "origin",
        intent_protocol_manifest=protocol,
        intent_chunk_token=manifest["intent"].get("chunk_token", ""),
        decode=manifest["codec"]["mode"],
        intent_max_pixels=e.get("intent_max_pixels", 16384),
        max_intent_tokens=e.get("max_intent_tokens", e.get("max_qwen_tokens", 32768)),
        intent_attention=e.get("intent_attention") or "fa2",
        intent_dtype=e.get("intent_dtype", "bfloat16"),
        expected_intent=manifest["intent"],
        intent_load_device="cpu",
        device="cuda:0",
    )
    for key, folder in (
        ("base_files", value["intent"]),
        ("adapter_files", value["intent_adapter"]),
    ):
        for filename, checksum in p.get(key, {}).items():
            path = Path(folder) / filename
            if _sha256(path) != checksum:
                raise ValueError("Intent rollout source checksum differs: " + str(path))
    expected_protocol = manifest["intent"].get("protocol_manifest_sha256")
    if expected_protocol is not None:
        protocol_path = (
            Path(protocol)
            if protocol
            else Path(value["intent_adapter"]) / "protocol_manifest.json"
        )
        if _sha256(protocol_path) != expected_protocol:
            raise ValueError(
                "Intent rollout protocol checksum differs: " + str(protocol_path)
            )
    return value


def window_record(source, selected, session):
    raw = Path(source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Canonical rollout source changed")
    canonical = json.loads(raw)
    # Canonical files may store dataset-root-relative media paths.
    base = Path(source["path"]).parent
    if canonical.get("path_base") == "dataset_root":
        base = (base / canonical.get("dataset_root", "../../..")).resolve()
    for key in ("directory", "metadata", "full_video", "user_video", "assistant_video"):
        if canonical.get("source", {}).get(key):
            path = Path(canonical["source"][key])
            canonical["source"][key] = str(path if path.is_absolute() else base / path)
    allowed, generated = set(selected["turn_ids"]), set(
        selected["last_assistant_turn_ids"]
    )
    lo = session["blocks"][0]["start_ms"]
    hi = min(
        session["blocks"][-1]["end_ms"],
        selected.get("assistant_end_ms", session["blocks"][-1]["end_ms"]),
    )
    original_tokens = canonical["text_tokens"]
    canonical["text_tokens"] = [
        t
        for t in original_tokens
        if t.get("turn_id") in allowed
        and not (t["role"] == "assistant" and t.get("turn_id") in generated)
        and lo <= t["start_ms" if t["role"] == "assistant" else "commit_ms"] < hi
    ]
    emotions = {}
    for key in ("attachments", "turns"):
        if key in canonical.get("emotions", {}):
            events = []
            for event in canonical["emotions"][key]:
                if (
                    event.get("turn_id") not in allowed
                    or event.get("turn_id") in generated
                ):
                    continue
                timestamp = event.get(
                    "commit_ms"
                    if event.get("role") == "user"
                    else "assistant_target_commit_ms"
                )
                if event.get("role") == "assistant":
                    clocks = [
                        t["start_ms"]
                        for t in original_tokens
                        if t["role"] == "assistant"
                        and t.get("turn_id") == event.get("turn_id")
                        and t["commit_ms"] == timestamp
                    ]
                    timestamp = min(clocks) if clocks else timestamp
                if timestamp is not None and lo <= timestamp < hi:
                    events.append(deepcopy(event))
            emotions[key] = events
    # Do not send unused full scripts/alignment text (including the GT answer)
    # to the worker. The serializer needs only these filtered condition fields.
    canonical = dict(
        sample_id=canonical["sample_id"],
        source={key: canonical["source"][key] for key in ("metadata", "user_video")},
        text_tokens=canonical["text_tokens"],
        emotions=emotions,
    )
    description = session.get("assistant_description", {}).get("fields")
    override = source.get("assistant_description_override")
    if description is not None and override is not None and description != override:
        raise ValueError("Cached/configured Assistant descriptions differ")
    return dict(
        sample_id=session["sample_id"],
        canonical=canonical,
        blocks=[
            dict(
                start_ms=b["start_ms"],
                end_ms=b["end_ms"],
                frame_start=b.get("frame_start", round(b["start_ms"] * 24 / 1000)),
                frame_stop=b.get("frame_stop", round(b["end_ms"] * 24 / 1000)),
            )
            for b in session["blocks"]
        ],
        preparation=session.get("preparation", {}),
        assistant_description_override=(
            description if description is not None else override
        ),
    )


def user_visuals(record, processor):
    """Decode just the selected User window on the original absolute frame clock."""
    import av
    import numpy as np

    from ..modules.intent_conditioning import visual_patch

    blocks = record["blocks"]
    start, stop = blocks[0]["frame_start"], blocks[-1]["frame_stop"]
    path = record["canonical"]["source"]["user_video"]
    frames = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / 24 / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Missing User video PTS")
            index = round(float(frame.pts * frame.time_base) * 24)
            if index >= stop:
                break
            if index >= start:
                frames[index] = frame.to_ndarray(format="rgb24")
    original = record.get("preparation", {}).get("original_frames")
    for index in range(start, stop):
        if index not in frames:
            if (
                record.get("preparation", {}).get("tail_policy") == "pad"
                and original is not None
                and index >= original
                and index - 1 in frames
            ):
                frames[index] = frames[index - 1]
            else:
                raise ValueError(f"Missing original User frame {index}")
    return {
        ("user", k): visual_patch(
            processor,
            np.stack([frames[i] for i in range(b["frame_start"], b["frame_stop"])]),
        )
        for k, b in enumerate(blocks)
    }


def rollout_conditions(
    model, tokenizer, record, visuals, rollout_start, args, *, progress=None
):
    """DMD selects GT/continuation bodies; shared Rollout owns all Intent execution."""
    import torch

    from ..modules import intent_conditioning as qc
    from ..modules.intent_rollout_inputs import RolloutInputs, encode_assistant_body
    from .intent_rollout import Rollout, assemble

    emit = progress or (lambda **values: None)
    if not 0 <= rollout_start < len(record["blocks"]):
        raise ValueError("Invalid natural Assistant rollout start")
    inputs = RolloutInputs.from_visuals(record, tokenizer, visuals, args)
    runner = Rollout(model, inputs, args)
    records, plans = [], []
    try:
        for k, block in enumerate(record["blocks"]):
            speech = qc.speech_ids(record["canonical"], block, "assistant", "planning")
            if k < rollout_start:
                body = qc.body_ids(
                    record["canonical"], block, "assistant", "planning", tokenizer
                )
                body = encode_assistant_body(tokenizer, body, speaking=bool(speech))
                prefix = ()
            else:
                # The filtered canonical contains only earlier Assistant turns,
                # including their known prefix in an overlapping target block.
                body = None
                known = (
                    qc.body_ids(
                        record["canonical"], block, "assistant", "planning", tokenizer
                    )
                    if speech
                    else qc.intent.encode_protocol(
                        tokenizer,
                        qc.emotion(record["canonical"], block, "assistant", "planning"),
                    )
                )
                prefix = (
                    encode_assistant_body(tokenizer, known, speaking=True)
                    if known
                    else ()
                )
            emit(phase="intent_prefill", block=k, completed=False)
            item = runner.step(k, body=body, prefix=prefix)
            records.append(item)
            if k >= rollout_start:
                plan = dict(
                    item["plan"],
                    mode="dmd",
                    intent_input_tokens=item["plan"]["input_tokens"],
                )
                plans.append(plan)
            emit(
                phase="intent_gt" if k < rollout_start else "intent_ntp",
                block=k,
                completed=True,
                prefix_only=item["plan"]["prefix_only"],
                truncated=item["plan"]["truncated"],
            )
        features = assemble(records)
        if (
            features["h50"].ndim != 2
            or features["h50"].shape[1] != 5120
            or not torch.isfinite(features["h50"]).all()
        ):
            raise ValueError("Invalid training rollout h50")
        return dict(
            features=features,
            metadata=dict(
                policy=POLICY,
                generated=plans,
                h50_rows=len(features["h50"]),
                gt_blocks=rollout_start,
                generated_blocks=sum(not p["prefix_only"] for p in plans),
                prefix_only_blocks=sum(p["prefix_only"] for p in plans),
                truncated_blocks=sum(p["truncated"] for p in plans),
                rollout_protocol=inputs.rollout_protocol,
                asr="disabled",
            ),
        )
    finally:
        runner.close()


def prepare_rollout_dataset(dataset):
    """Add immutable small indexes to a private dataset view; never rewrite caches."""
    if getattr(dataset, "_dmd_rollout_prepared", False):
        return dataset
    from data.dmd_turn_windows import dialogue_runs

    from ..modules.codec_spec import spec

    result = copy(dataset)
    result.records = deepcopy(dataset.records)
    result.index = result.records
    result._dmd_rollout_prepared = True
    config = dataset.manifest.get("preparation_config", {})
    root = Path(config["root"]) if config.get("root") else None
    if root is None:
        raise ValueError(
            "Intent rollout requires feature preparation_config.root with canonical records"
        )
    geometry = spec(dataset.codec["mode"])
    for row in result.records:
        row.setdefault("height", dataset.manifest.get("height"))
        row.setdefault("width", dataset.manifest.get("width"))
        path = root / row["sample_id"] / "canonical/canonical.json"
        raw = path.read_bytes()
        canonical = json.loads(raw)
        if canonical["sample_id"] != row["sample_id"]:
            raise ValueError("Canonical/feature sample identity differs")
        row["rollout_source"] = dict(
            path=str(path), sha256=hashlib.sha256(raw).hexdigest()
        )
        override = config.get("assistant_descriptions", {}).get(row["sample_id"])
        if override:
            row["rollout_source"]["assistant_description_override"] = override
        row["dialogue_runs"] = dialogue_runs(canonical)
        if "window_index" not in row:
            import torch
            from data.dataloader_window import window_index

            item = torch.load(
                row["path"], map_location="cpu", mmap=True, weights_only=True
            )
            row["window_index"] = window_index(item)
        video = geometry.latents * (row["height"] // 32) * (row["width"] // 32)
        audio = 2 * geometry.audio_latents
        row["block_text_rows"] = [
            b["packed_rows"] - 2 * (video + audio)
            for b in row["window_index"]["blocks"]
        ]
        if any(n < 0 for n in row["block_text_rows"]):
            raise ValueError("Invalid per-block text reservation index")
    return result
