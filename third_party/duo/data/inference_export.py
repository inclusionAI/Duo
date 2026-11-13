"""Optional, lossless inference exports shared by RAVEN and Torchtitan.

Only the Intent process captures full h50. Distributed callers exchange small
capture references; raw hidden states never become additional Realization conditions.
"""

import hashlib
import json
import uuid
from pathlib import Path

FORMAT = "duplex-latents-v1"
TENSOR_FILES = (
    "video_latents.pt",
    "audio_latents.pt",
    "reference_latents.pt",
    "h50.pt",
)


def settings(args):
    enabled = bool(getattr(args, "save_latents", False))
    return dict(
        save_latents=enabled,
        latent_export_format=(
            (
                "duplex-latents-staged-v1"
                if getattr(args, "pipeline_policy", "") == "staged_no_asr_v1"
                else FORMAT
            )
            if enabled
            else None
        ),
    )


def destination(output, record, multiple):
    root = Path(output) / "saved_latents"
    return root / record["split"] / record["sample_id"] if multiple else root


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_save(value, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_export(folder, *, source_sha256=None, blocks=None):
    """Torch-free reader contract, also used before publishing the OSS marker."""
    folder = Path(folder)
    value = json.loads((folder / "metadata.json").read_text())
    if (
        value.get("format") not in (FORMAT, "duplex-latents-staged-v1")
        or value.get("status") != "complete"
    ):
        raise ValueError("Invalid latent export metadata")
    if source_sha256 is not None and value.get("source_sha256") != source_sha256:
        raise ValueError("Latent export source differs")
    if blocks is not None and value.get("blocks") != blocks:
        raise ValueError("Latent export length differs")
    if set(value.get("files", {})) != set(TENSOR_FILES):
        raise ValueError("Missing latent export files")
    for name, info in value["files"].items():
        path = folder / name
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError("Latent export checksum differs: " + name)
    return value


def publish_staged_export(
    output,
    cache,
    latents,
    features,
    plans,
    states,
    record,
    args,
    source,
    coverage,
    local,
):
    """Cached selected h50 is never mislabeled as a full model input capture."""
    import torch
    from data.inference_options import ROLLING_MEDIA_CACHE_POLICY

    cache_policy = (
        ROLLING_MEDIA_CACHE_POLICY
        if args.cache_mode == "block" and args.realization_inference_mode == "causal"
        else "full_window_recompute_v1"
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = digest(Path(local) / "config.json")
    marker = output / "metadata.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("run_identity") != run_identity:
            raise ValueError("Latent export belongs to another run")
    # Select the actual generated prefix while preserving original input-row
    # positions; cached rows outside that prefix are not claimed as exports.
    features = dict(features)
    features["text_groups"] = [
        g for g in features["text_groups"] if g["block"] < len(latents)
    ]
    rows = sum(g["end"] - g["start"] for g in features["text_groups"])
    features["h50"] = features["h50"][:rows]
    features["text_ids"] = features["text_ids"][:rows]
    mapping = features["input_keep"].nonzero().flatten().tolist()[:rows]
    captures = {}
    if args.mode == "closed_loop":
        capture = features.get("h50_capture")
        if capture is None:
            raise ValueError("Missing full NTP prefix h50 captures")
        value = ExportCache(local).load_capture(capture)
        captures["prefixes"] = dict(h50=value["h50"], input_ids=value["input_ids"])
        scope = (
            "all_executed_ntp_forward_segments"
            if features.get("h50_strategy") == "ntp_forward_layer50_v1"
            else "all_input_tokens_of_legal_ntp_prefixes"
        )
    else:
        captures["cached"] = dict(h50=features["h50"], input_ids=features["text_ids"])
        scope = "cached_selected_conditioning_rows"
    uses = []
    for k, state in enumerate(states):
        first, stop = (
            state["media_window"][name] for name in ("start_block", "stop_block")
        )
        selected = [
            g
            for g in features["text_groups"]
            if g["block"] == -1 or first <= g["block"] < stop
        ]
        rows = [i for g in selected for i in range(g["start"], g["end"])]
        uses.append(
            dict(
                block=k,
                media_window=state["media_window"],
                selected_h50_rows=rows,
                media_cache_policy=cache_policy,
                selected_input_rows=[mapping[i] for i in rows],
                current_refiner_blocks=(
                    [-1, k]
                    if args.cache_mode == "block"
                    and args.realization_inference_mode == "causal"
                    else [g["block"] for g in selected]
                ),
                history_source="generated_latents",
            )
        )
    from torchtitan.models.duo_realization.pipelines.streaming import shared_condition

    values = dict(
        video_latents=torch.stack([p["video"] for p in latents]),
        audio_latents=torch.stack([p["audio"] for p in latents]),
        reference_latents=cache["reference"],
        h50=dict(
            snapshots=captures,
            selected_h50=features["h50"],
            media_cache=dict(
                policy=cache_policy,
                shared=shared_condition(features),
                blocks=[s.get("media_cache_condition") for s in states],
            ),
        ),
    )
    for name, value in values.items():
        atomic_save(value, output / (name + ".pt"))
    metadata = dict(
        format="duplex-latents-staged-v1",
        status="complete",
        run_identity=run_identity,
        source_sha256=run_identity,
        sample_id=record["sample_id"],
        split=record["split"],
        blocks=len(latents),
        checkpoint_source=source,
        codec=cache["codec"],
        media_cache_policy=cache_policy,
        h50=dict(
            layer=50,
            scope=scope,
            selected_row_to_input_row=mapping,
            text_groups=features["text_groups"],
            conditioning_layout=features.get("conditioning_layout"),
            h50_strategy=features.get("h50_strategy"),
            selected_rows=rows,
            exported_condition_blocks=list(range(len(latents))),
        ),
        plans=plans[: len(latents)],
        realization_uses=uses,
        **coverage,
        files={
            name: dict(
                bytes=(output / name).stat().st_size, sha256=digest(output / name)
            )
            for name in TENSOR_FILES
        }
    )
    atomic_json(metadata, marker)
    validate_export(output, blocks=len(latents))
    return str(output)


def bind_conditioning(features, session, targets):
    """Map the actual windowed Realization rows back to full Intent input positions."""
    capture = features.get("h50_capture")
    if capture is None:
        raise ValueError("Missing complete h50 capture; cannot backfill old inference")
    start, stop = (session["media_window"][k] for k in ("start_block", "stop_block"))
    selected, groups, cursor = [], [], 0
    for group in features["text_groups"]:
        if group["block"] == -1 or group["block"] in session.get(
            "conditioning_blocks", range(start, stop)
        ):
            rows = capture["selected_rows"][group["start"] : group["end"]]
            selected.extend(rows)
            groups.append(dict(group, start=cursor, end=cursor + len(rows)))
            cursor += len(rows)
    if cursor != len(session["h50"]):
        raise ValueError("Export row mapping differs from Realization conditioning")
    binding = dict(
        capture={
            k: v
            for k, v in capture.items()
            if k != "selected_rows" or "parts" in capture
        },
        selected_input_rows=selected,
        text_groups=groups,
        h50_strategy=features.get("h50_strategy"),
        target_blocks=list(targets),
        media_window=dict(session["media_window"]),
        media_cache_policy=session.get(
            "media_cache_policy", "full_window_recompute_v1"
        ),
    )
    for key in ("media_cache_condition", "media_cache_shared"):
        if key in session:
            binding[key] = session[key]
    return binding


class ExportCache:
    """Per-sample durable capture cache. Construct only on the writing process."""

    def __init__(self, local):
        self.local = Path(local)
        self.root = self.local / "latent-export"
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(self, hidden, text):
        import torch

        h50 = hidden.detach().to("cpu").contiguous()
        ids = torch.tensor(text["input_ids"], dtype=torch.long)
        if h50.shape != (len(ids), 5120) or not torch.isfinite(h50).all():
            raise ValueError("Invalid full Intent h50")
        spans = text["input_spans"]
        cursor = 0
        for span in spans:
            if span["start"] != cursor or span["end"] <= cursor:
                raise ValueError("Incomplete Intent input span map")
            cursor = span["end"]
        if cursor != len(ids):
            raise ValueError("Intent input spans do not cover all h50 rows")
        identifier = "capture-" + uuid.uuid4().hex
        path = self.root / (identifier + ".pt")
        value = dict(
            h50=h50,
            input_ids=ids,
            input_spans=spans,
            selected_input_rows=[i for i, keep in enumerate(text["keep"]) if keep],
            protocol=text["protocol"],
            video_grid_thw=text["video_grid_thw"],
        )
        atomic_save(value, path)
        return dict(
            id=identifier,
            sha256=digest(path),
            rows=len(ids),
            selected_rows=value["selected_input_rows"],
        )

    def load_capture(self, capture):
        import torch

        if "parts" in capture:
            identity = hashlib.sha256(
                json.dumps(capture["parts"], sort_keys=True).encode()
            ).hexdigest()
            if identity != capture["sha256"]:
                raise ValueError("Composite h50 capture identity differs")
            parts = [self.load_capture(part) for part in capture["parts"]]
            spans, offset = [], 0
            for part in parts:
                for span in part["input_spans"]:
                    shifted = dict(
                        span, start=span["start"] + offset, end=span["end"] + offset
                    )
                    if "video_token_rows" in span:
                        shifted["video_token_rows"] = [
                            i + offset for i in span["video_token_rows"]
                        ]
                    spans.append(shifted)
                offset += len(part["input_ids"])
            if offset != capture["rows"]:
                raise ValueError("Composite h50 row count differs")
            return dict(
                h50=torch.cat([p["h50"] for p in parts]),
                input_ids=torch.cat([p["input_ids"] for p in parts]),
                input_spans=spans,
                selected_input_rows=capture["selected_rows"],
                protocol=parts[0]["protocol"],
                video_grid_thw=[g for p in parts for g in p["video_grid_thw"]],
            )
        identifier = capture["id"]
        if not identifier.startswith("capture-") or Path(identifier).name != identifier:
            raise ValueError("Invalid h50 capture reference")
        path = self.root / (identifier + ".pt")
        if digest(path) != capture["sha256"]:
            raise ValueError("Saved h50 capture checksum differs")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value["h50"].shape != (capture["rows"], 5120):
            raise ValueError("Saved h50 capture shape differs")
        return value

    def prepare(
        self, reference, latents, bindings, record, args, *, framework, source=None
    ):
        """Save a complete export source before decoding the last/joint latent."""
        blocks = len(latents)
        if not blocks or len(bindings) != blocks:
            raise ValueError("Expected one conditioning binding per generated block")
        for k, binding in enumerate(bindings):
            if k not in binding["target_blocks"]:
                raise ValueError("Wrong target block for h50 capture")
        from torchtitan.models.duo_realization.modules.codec_spec import (
            mode_from_args,
            spec,
        )

        geometry = spec(mode_from_args(args))
        timeline = []
        for k, block in enumerate(record["blocks"][:blocks]):
            timeline.append(
                dict(
                    block=k,
                    video_latent_index=k,
                    audio_latent_index=k,
                    video_latent_start=k * geometry.latents,
                    video_latent_stop=(k + 1) * geometry.latents,
                    audio_latent_start=k * geometry.audio_latents,
                    audio_latent_stop=(k + 1) * geometry.audio_latents,
                    video_spans=geometry.spans(block["frame_start"]),
                    frame_start=block["frame_start"],
                    frame_stop=block["frame_stop"],
                    start_ms=block["start_ms"],
                    end_ms=block["end_ms"],
                    audio_start_sample=block["audio_start_sample"],
                    audio_stop_sample=block["audio_stop_sample"],
                )
            )
        keys = (
            "decode",
            "intent_chunk_token",
            "sana_weights",
            "exp_name",
            "checkpoint",
            "checkpoint_family",
            "realization",
            "intent",
            "intent_adapter",
            "seed",
            "steps",
            "flow_shift",
            "mode",
            "realization_inference_mode",
            "stage",
            "height",
            "width",
            "sliding_window_size",
            "video_decode_latent",
            "cache_mode",
        )
        config = {key: getattr(args, key) for key in keys if hasattr(args, key)}
        config.setdefault("realization_inference_mode", "causal")
        run_identity = digest(self.local / "config.json")
        runtime_file = self.local / "codec_runtime.json"
        runtime = json.loads(runtime_file.read_text()) if runtime_file.is_file() else {}
        value = dict(
            codec=runtime.get("codec", record.get("codec", geometry.identity())),
            intent=runtime.get("intent", getattr(args, "intent_identity", {})),
            reference=reference.detach().cpu().contiguous(),
            latents=latents,
            bindings=bindings,
            timeline=timeline,
            config=config,
            source=source,
            framework=framework,
            sample_id=record["sample_id"],
            split=record["split"],
            run_identity=run_identity,
            format=FORMAT,
        )
        atomic_save(value, self.root / "ready.pt")
        atomic_json(
            dict(
                format=FORMAT,
                blocks=blocks,
                run_identity=run_identity,
                sha256=digest(self.root / "ready.pt"),
            ),
            self.root / "ready.json",
        )

    def ensure(self, output, wanted):
        """Validate/rebuild a finished export without Intent, Realization, VAE, or RNG use."""
        import torch

        marker = json.loads((self.root / "ready.json").read_text())
        if marker.get("format") != FORMAT or marker["blocks"] != wanted:
            raise ValueError(
                "Missing full latent export recovery source for requested length"
            )
        if marker.get("run_identity") != digest(self.local / "config.json"):
            raise ValueError("Latent export recovery identity differs")
        if digest(self.root / "ready.pt") != marker["sha256"]:
            raise ValueError("Latent export recovery source checksum differs")
        output = Path(output)
        try:
            return validate_export(
                output, source_sha256=marker["sha256"], blocks=wanted
            )
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            pass
        saved = torch.load(
            self.root / "ready.pt", map_location="cpu", weights_only=True
        )
        if saved["run_identity"] != digest(self.local / "config.json"):
            raise ValueError("Latent export recovery identity differs")
        if (output / "metadata.json").is_file():
            try:
                old = json.loads((output / "metadata.json").read_text())
            except json.JSONDecodeError:
                old = None  # The validated run owns this incomplete publication.
            if old is not None and (
                old.get("run_identity") != saved["run_identity"]
                or old.get("sample_id") != saved["sample_id"]
            ):
                raise ValueError("Latent export directory belongs to another run")
        video = torch.stack([row["video"] for row in saved["latents"]])
        audio = torch.stack([row["audio"] for row in saved["latents"]])
        reference = saved["reference"]
        from torchtitan.models.duo_realization.modules.codec_spec import spec

        geometry = spec(saved["config"].get("decode", "old"))
        if video.shape != (
            wanted,
            24,
            geometry.latents,
            *reference.shape[-2:],
        ) or reference.shape[:2] != (24, 1):
            raise ValueError("Unexpected generated video/reference shape")
        if audio.shape != (wanted, 2, 32, geometry.audio_latents):
            raise ValueError("Unexpected generated audio shape")
        snapshots, snapshot_info, uses = {}, {}, []
        for k, binding in enumerate(saved["bindings"]):
            capture = binding["capture"]
            identifier = capture["id"]
            if identifier not in snapshots:
                full = self.load_capture(capture)
                snapshots[identifier] = dict(
                    h50=full["h50"], input_ids=full["input_ids"]
                )
                snapshot_info[identifier] = dict(
                    rows=capture["rows"],
                    shape=list(full["h50"].shape),
                    dtype=str(full["h50"].dtype),
                    input_ids_shape=list(full["input_ids"].shape),
                    input_ids_dtype=str(full["input_ids"].dtype),
                    protocol=full["protocol"],
                    input_spans=full["input_spans"],
                    video_grid_thw=full["video_grid_thw"],
                    all_selected_input_rows=full["selected_input_rows"],
                )
            if any(
                i < 0 or i >= capture["rows"] for i in binding["selected_input_rows"]
            ):
                raise ValueError("Invalid Realization row index")
            uses.append(
                dict(
                    block=k,
                    snapshot=identifier,
                    selected_input_rows=binding["selected_input_rows"],
                    text_groups=binding["text_groups"],
                    media_window=binding["media_window"],
                    target_blocks=binding["target_blocks"],
                    media_cache_policy=binding.get(
                        "media_cache_policy", "full_window_recompute_v1"
                    ),
                    committed_condition_index=(
                        k if "media_cache_condition" in binding else None
                    ),
                )
            )
        media_cache = None
        if "media_cache_condition" in saved["bindings"][0]:
            media_cache = dict(
                shared=saved["bindings"][0]["media_cache_shared"],
                blocks=[b["media_cache_condition"] for b in saved["bindings"]],
            )
        tensors = dict(
            video_latents=video, audio_latents=audio, reference_latents=reference
        )
        if any(
            t.device.type != "cpu" or not torch.isfinite(t).all()
            for t in tensors.values()
        ):
            raise ValueError("Invalid generated export tensors")
        metadata = dict(
            format=FORMAT,
            status="complete",
            source_sha256=marker["sha256"],
            run_identity=saved["run_identity"],
            sample_id=saved["sample_id"],
            split=saved["split"],
            framework=saved["framework"],
            config=saved["config"],
            checkpoint_source=saved["source"],
            codec=saved.get("codec", geometry.identity()),
            intent=saved.get("intent", {}),
            blocks=wanted,
            video_frames=geometry.frames * wanted,
            video_latents=geometry.latents * wanted,
            reference_latents=1,
            audio_latents_per_channel=geometry.audio_latents * wanted,
            audio_channels=2,
            fps=24,
            audio_sample_rate=32000,
            audio_samples_per_channel=sum(
                x["audio_stop_sample"] - x["audio_start_sample"]
                for x in saved["timeline"]
            ),
            timeline=saved["timeline"],
            normalization=dict(
                video="normalized Realization/VAE latent space",
                audio="normalized Realization/VAE latent space; all padded block positions retained",
                reference="normalized video VAE latent space",
                decode=(
                    "Sana decoder accepts normalized TAE latents directly"
                    if geometry.name == "sana_1_4"
                    else "z * matching VAE latents_std + latents_mean; source is config.realization"
                ),
            ),
            axes=dict(
                video_latents=["block", "channel", "time", "height", "width"],
                audio_latents=["block", "stereo_channel", "channel", "time"],
                reference_latents=["channel", "time", "height", "width"],
                h50=["input_token", "hidden_channel"],
                input_ids=["input_token"],
            ),
            h50=dict(
                layer=50,
                layer_index=49,
                before_final_norm=True,
                scope=(
                    "all_executed_ntp_forward_segments"
                    if any(
                        b.get("h50_strategy") == "ntp_forward_layer50_v1"
                        for b in saved["bindings"]
                    )
                    else "all_input_tokens"
                ),
                ntp_internal_states=any(
                    b.get("h50_strategy") == "ntp_forward_layer50_v1"
                    for b in saved["bindings"]
                ),
                snapshots=snapshot_info,
                realization_uses=uses,
                committed_conditions=(
                    "h50.pt:media_cache" if media_cache is not None else None
                ),
            ),
            files={},
        )
        output.mkdir(parents=True, exist_ok=True)
        for name in TENSOR_FILES:
            value = (
                dict(snapshots=snapshots, media_cache=media_cache)
                if name == "h50.pt"
                else tensors[name[:-3]]
            )
            atomic_save(value, output / name)
            info = dict(
                bytes=(output / name).stat().st_size, sha256=digest(output / name)
            )
            if name != "h50.pt":
                info.update(shape=list(value.shape), dtype=str(value.dtype))
            else:
                info["snapshots"] = len(snapshots)
            metadata["files"][name] = info
        atomic_json(metadata, output / "metadata.json")
        return validate_export(output, source_sha256=marker["sha256"], blocks=wanted)
