"""In-process frozen conditions/codecs and phase-boundary model residency."""

import importlib.util
import time
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import torch
from data.inference_budget import budget_index

from ..models.duo_realization.modules import intent_conditioning as conditioning
from ..models.duo_realization.modules.codec_spec import ceil_fraction, spec
from . import inference_state_realization as state


def select_records(args):
    from data import build_dataset

    from ..models.duo_realization.modules.media_features import frame_records

    records = list(
        build_dataset(
            args.data_config,
            video_codec=args.decode,
            tail_policy=getattr(args, "tail_policy", None),
        )
    )
    by_id = {r["sample_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate dataset sample IDs")
    ids = [args.sample_id] if args.sample_id else args.sample_ids
    if ids:
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated sample IDs")
        if any(i not in by_id for i in ids):
            raise ValueError("Unknown sample ID")
        records = [by_id[i] for i in ids]
        if any(args.split != "all" and r["split"] != args.split for r in records):
            raise ValueError("Selected sample does not belong to split")
    else:
        records = [
            r for r in records if args.split == "all" or r["split"] == args.split
        ]
    limit = args.max_samples or (0 if args.all_samples or ids else 1)
    if limit:
        records = records[:limit]
    if not records:
        raise ValueError("No samples selected")
    for r in records:
        if r["sample_id"] in (".", "..") or Path(r["sample_id"]).name != r["sample_id"]:
            raise ValueError("Unsafe sample ID")
        if r["split"] not in ("train", "validation"):
            raise ValueError("Invalid split")
    return list(frame_records(records, "planning", args.decode))


def clock_block(k, mode):
    geometry = spec(mode)
    start, stop = k * geometry.frames, (k + 1) * geometry.frames
    lo, hi = (ceil_fraction(Fraction(f * 32000, 24)) for f in (start, stop))
    return dict(
        block_id=k,
        frame_start=start,
        frame_stop=stop,
        start_ms=start * 1000 / 24,
        end_ms=stop * 1000 / 24,
        audio_start_sample=lo,
        audio_stop_sample=hi,
        audio_padded_samples=geometry.audio_latents * 800,
        committed={"user": [], "assistant": []},
    )


def valid_frames(record):
    prepared = record.get("preparation", {})
    field = "original_frames" if prepared.get("tail_policy") == "pad" else "kept_frames"
    return int(prepared.get(field, record["canonical"]["video"]["source_frames"]))


def feature_view(item):
    keys = (
        "h50",
        "text_groups",
        "protocol",
        "h50_ends",
        "controls",
        "input_ids",
        "input_keep",
        "text_ids",
        "selected_spans",
        "template_version",
        "assistant_description",
        "h50_capture",
    )
    return {k: item[k] for k in keys if k in item}


class Residency:
    def __init__(self, device, emit=lambda **values: None, pin_memory=True):
        self.device = torch.device(device)
        self.models = {}
        self.emit = emit
        self.pin_memory = pin_memory

    def register(self, name, value, move=None, modules=None):
        self.models[name] = dict(
            value=value,
            move=move or value.to,
            modules=modules or (lambda: [value]),
            device="cpu",
        )

    def size(self, name):
        seen, size = set(), 0
        for module in self.models[name]["modules"]():
            if module is None:
                continue
            for value in (*module.parameters(), *module.buffers()):
                if id(value) not in seen:
                    seen.add(id(value))
                    size += value.numel() * value.element_size()
        return size

    def move(self, name, device, phase):
        row = self.models[name]
        before = time.monotonic()
        row["move"](device)
        if str(device) == "cpu" and self.pin_memory and self.device.type == "cuda":
            for module in row["modules"]():
                if module is not None:
                    for value in (*module.parameters(), *module.buffers()):
                        if value.device.type == "cpu" and not value.is_pinned():
                            value.data = value.data.pin_memory()
        row["device"] = str(device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.emit(
            phase=phase,
            model=name,
            destination=str(device),
            model_transfer_seconds=time.monotonic() - before,
        )

    def missing_bytes(self, name):
        seen, size = set(), 0
        for module in self.models[name]["modules"]():
            if module is not None:
                for value in (*module.parameters(), *module.buffers()):
                    if id(value) not in seen and value.device.type == "cpu":
                        seen.add(id(value))
                        size += value.numel() * value.element_size()
        return size

    def enter(self, phase, active, next_use, workspace_bytes=8_000_000_000):
        """Only a phase boundary calls this; no per-block weight transfers."""
        if self.device.type != "cuda":
            return
        torch.cuda.empty_cache()
        required = workspace_bytes + sum(self.missing_bytes(n) for n in active)
        candidates = sorted(
            (
                n
                for n in self.models
                if n not in active and self.models[n]["device"] != "cpu"
            ),
            key=lambda n: next_use.get(n, float("inf")),
            reverse=True,
        )
        for name in candidates:
            if torch.cuda.mem_get_info(self.device)[0] >= required:
                break
            self.move(name, "cpu", phase)
            torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(self.device)[0]
        if free < required:
            raise MemoryError(
                f"Phase {phase} needs {required} free bytes including workspace, has {free}"
            )
        for name in active:
            if self.missing_bytes(name):
                self.move(name, self.device, phase)
        self.emit(
            phase=phase,
            resident_models=[n for n, r in self.models.items() if r["device"] != "cpu"],
            allocated_bytes=torch.cuda.memory_allocated(self.device),
            reserved_bytes=torch.cuda.memory_reserved(self.device),
        )


class FrozenRuntime:
    def __init__(self, args, device, residency):
        self.args, self.device, self.residency = args, torch.device(device), residency
        self.planner = self.video = self.audio = None
        self.tokenizer = None
        self.intent_identity = None

    def tokenizer_identity(self):
        if self.tokenizer is None:
            from ..models.duo_realization.modules.intent import prepare_tokenizer

            self.tokenizer, _ = prepare_tokenizer(self.args)
            self.intent_identity = deepcopy(self.args.intent_identity)
        return self.intent_identity

    def load_video(self, *, encode=False):
        if self.video is None:
            from ..models.duo_realization.modules.video_vae import VideoVAE

            self.video = VideoVAE(
                self.args,
                vae_dtype=torch.float32,
                video_autocast_dtype=torch.float16,
                feature_dtype=torch.bfloat16,
            )
        # Materialize on CPU before admission; objects/weights survive cases.
        if self.video.spec.name == "sana_1_4":
            previous = self.video.device
            missing = (
                self.video.encoder is None if encode else self.video.decoder is None
            )
            if missing:
                self.video.device = "cpu"
            try:
                self.video._sana_encoder() if encode else self.video._sana_decoder()
            finally:
                self.video.device = previous
        else:
            self.video._native()
        if not encode and hasattr(self.video, "prepare_decode"):
            self.video.prepare_decode()
        if "video" not in self.residency.models:
            self.residency.register(
                "video",
                self.video,
                modules=lambda: [
                    self.video.model,
                    self.video.encoder,
                    self.video.decoder,
                ],
            )
        return self.video

    def close_decoder(self):
        if (
            self.video is not None
            and getattr(self.video, "streaming", None) is not None
        ):
            self.video.streaming.finish()
            self.video.closed = True

    def load_audio(self):
        if self.audio is None:
            from ..models.duo_realization.modules.audio_vae import AudioVAE

            self.audio = AudioVAE(
                self.args, vae_dtype=torch.float32, feature_dtype=torch.bfloat16
            )
            self.residency.register(
                "audio", self.audio, modules=lambda: [self.audio.model]
            )
        return self.audio

    def read_features(self, record):
        from data import build_dataset

        if not (self.args.feature_config or self.args.feature_manifest):
            if self.args.mode == "oracle":
                raise ValueError("Oracle requires compatible cached h50")
            return None
        source = self.args.feature_config or dict(
            source="raven_features", manifest=self.args.feature_manifest
        )
        ds = build_dataset(
            source,
            record["split"],
            feature_root=self.args.feature_root or None,
            video_codec=self.args.decode,
        )
        index = next(
            (i for i, r in enumerate(ds) if r["sample_id"] == record["sample_id"]), None
        )
        if index is None:
            raise ValueError("Selected features do not contain this sample/split")
        row = ds[index]
        if row.get("sha256") and state.digest(row["path"]) != row["sha256"]:
            raise ValueError("Selected feature checksum differs from manifest")
        item = ds.get(index)
        from data.inference_media import validate_checkpoint_codec

        validate_checkpoint_codec(self.args.checkpoint_codec, item["codec"])
        identity = self.tokenizer_identity()
        if item.get("intent") != identity or (
            self.args.checkpoint_intent and self.args.checkpoint_intent != identity
        ):
            raise ValueError("Cached/checkpoint Intent identity differs from runtime")
        from .feature_metadata import intent_provenance

        provenance = item.get("intent_provenance")
        actual = intent_provenance(self.args)
        if not provenance or any(
            provenance.get(k) != actual.get(k)
            for k in (
                "kind",
                "h50_layer",
                "base_files",
                "adapter_files",
                "control_filter",
            )
        ):
            raise ValueError(
                "Cached Intent/adapter/tokenizer provenance differs or is missing"
            )
        if item.get("template_version") != conditioning.TEMPLATE_VERSION:
            raise ValueError("Cached h50 template differs")
        description = conditioning.describe(record)
        if any(
            item.get("assistant_description", {}).get(k) != description[k]
            for k in ("text", "fields")
        ):
            raise ValueError("Cached Assistant description differs")
        features = feature_view(item)
        budget_index(features, self.args.height, self.args.width, self.args.decode)
        if (
            features["input_ids"].ndim != 1
            or features["input_keep"].shape != features["input_ids"].shape
            or features["input_keep"].dtype != torch.bool
            or not torch.equal(
                features["text_ids"], features["input_ids"][features["input_keep"]]
            )
            or len(features["text_ids"]) != len(features["h50"])
            or not torch.isfinite(features["h50"]).all()
        ):
            raise ValueError("Invalid cached token/h50 row mapping")
        mapping = features["input_keep"].nonzero().flatten().tolist()
        for group in features["text_groups"]:
            spans = [
                s
                for s in features["selected_spans"]
                if s["block"] == group["block"] and s["section"] == group["section"]
            ]
            if len(spans) != 1 or mapping[group["start"] : group["end"]] != list(
                range(spans[0]["start"], spans[0]["end"])
            ):
                raise ValueError("Cached h50 group/input span mapping differs")
        for k, b in enumerate(item["blocks"]):
            expected = clock_block(k, self.args.decode)
            if any(
                abs(b[key] - expected[key]) > 1e-6 for key in ("start_ms", "end_ms")
            ):
                raise ValueError("Cached feature clock differs from codec")
        if len(item["blocks"]) != len(
            [g for g in features["text_groups"] if g["block"] >= 0]
        ):
            raise ValueError("Cached h50 groups do not cover all feature blocks")
        return item

    def reference(self, record, item):
        shape = (24, 1, self.args.height // 16, self.args.width // 16)
        if item is not None and tuple(item["reference"].shape) == shape:
            return item["reference"], dict(kind="cached_reference", shape=list(shape))
        import av
        from data.dataset_utils import digest

        with av.open(record["canonical"]["source"]["assistant_video"]) as container:
            first = next(container.decode(video=0))
            if first.pts is None or abs(float(first.pts * first.time_base)) > 0.001:
                raise ValueError("Reference is not Assistant frame zero")
            rgb = first.reformat(
                width=self.args.width, height=self.args.height, format="rgb24"
            ).to_ndarray()
        video = self.load_video(encode=True)
        self.residency.enter(
            "reference", ["video"], dict(intent=1, realization=2, video=3, audio=3)
        )
        seed = int(
            digest([self.args.seed, record["sample_id"], "reference_v1"])[:8], 16
        )
        value = video.encode_reference(rgb, seed)
        if tuple(value.shape) != shape:
            raise ValueError("Re-encoded reference geometry differs")
        return value, dict(
            kind="assistant_first_frame_reencoded", shape=list(shape), seed=seed
        )

    def observations(self, record, item):
        reference, source = self.reference(record, item)
        count = (
            len([g for g in item["text_groups"] if g["block"] >= 0])
            if self.args.mode == "oracle"
            else len(record["blocks"])
            + (
                self.args.realization_inference_mode == "causal"
                and getattr(self.args, "origin_intent", False)
            )
        )
        if not count:
            raise ValueError("No inference conditions")
        return dict(
            sample_id=record["sample_id"],
            codec=(
                item["codec"] if item is not None else spec(self.args.decode).identity()
            ),
            intent=self.tokenizer_identity(),
            reference=reference,
            reference_source=source,
            blocks=[clock_block(k, self.args.decode) for k in range(count)],
            available_blocks=count,
            valid_frames=valid_frames(record),
        )

    def ensure_planner(self):
        if self.planner is None:
            from ..models.duo_realization.pipelines.intent_planner import Planner

            self.args.device = str(self.device)
            self.args.intent_load_device = "cpu"
            self.planner = Planner(self.args, intent_dtype=torch.bfloat16)
            self.residency.register("intent", self.planner.model)
        return self.planner

    def prepare(self, record, cache, item, local, saved, count, emit):
        if len(saved) > count:
            raise ValueError("Condition progress exceeds the selected prefix")
        if self.args.mode == "oracle":
            features = feature_view(item)
            plans = []
            for k in range(count):
                group = next(g for g in features["text_groups"] if g["block"] == k)
                ids = features["text_ids"][group["start"] : group["end"]].tolist()
                from .inference_config_realization import conditioning_metadata

                plans.append(
                    dict(
                        mode="oracle",
                        **conditioning_metadata(
                            "oracle", self.args.realization_inference_mode
                        ),
                        body_ids=ids,
                        raw_body_ids=ids,
                        text=self.tokenizer.decode(
                            ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        ),
                        controls=[],
                        normalization=dict(changed=False, reason="cached_tokens"),
                    )
                )
                if k >= len(saved):
                    state.publish_record(local, "conditions", k, dict(plan=plans[-1]))
                    emit(
                        phase="conditions",
                        completed_conditions=k + 1,
                        total_conditions=cache["available_blocks"],
                    )
            return dict(features=features, plans=plans)
        planner = self.ensure_planner()
        if not getattr(self.args, "origin_intent", False):
            from data.inference_export import ExportCache

            planner.capture_h50 = (
                ExportCache(local).capture if self.args.save_latents else None
            )
            self.residency.enter(
                "conditions",
                ["intent"],
                dict(realization=1, video=2, audio=2, intent=3),
                workspace_bytes=12_000_000_000,
            )
            last = time.monotonic()

            def publish(k, value):
                nonlocal last
                now = time.monotonic()
                from .checkpoint_realization import rng_state

                value["rng"] = rng_state()
                value["intent_seconds"] = now - last
                state.publish_record(local, "conditions", k, value)
                last = now

            return planner.prepare_closed_loop(
                record, count, saved=saved, publish=publish, progress=emit
            )
        from data.inference_export import ExportCache

        from ..models.duo_realization.modules.media_features import (
            pad_video_tail,
            read_video,
        )

        if self.args.save_latents:
            planner.capture_h50 = ExportCache(local).capture
        prepared = record.get("preparation", {})
        duration = (
            prepared["original_frames"] * 1000 / 24
            if prepared.get("tail_policy") == "pad"
            else record["canonical"]["duration_ms"]
        )
        user = read_video(
            record["canonical"]["source"]["user_video"], duration, None, None
        )
        pad_video_tail(user, record["blocks"][-1]["frame_stop"])
        visuals = {}
        for k, row in enumerate(
            record["blocks"][: min(count - 1, len(record["blocks"]))]
        ):
            visuals["user", k] = conditioning.visual_patch(
                planner.processor,
                user["frames"][row["frame_start"] : row["frame_stop"]],
            )
        augmented = dict(
            record,
            blocks=[*record["blocks"], *cache["blocks"][len(record["blocks"]) :]],
        )
        self.residency.enter(
            "conditions",
            ["intent"],
            dict(realization=1, video=2, audio=2, intent=3),
            workspace_bytes=12_000_000_000,
        )
        prefixes = [p["features"] for p in saved]
        plans = [p["plan"] for p in saved]
        bodies = {k: p["body_ids"] for k, p in enumerate(plans)}
        if saved and saved[-1].get("rng"):
            from .checkpoint_realization import restore_rng

            restore_rng(saved[-1]["rng"])
        for k in range(len(saved), count):
            started = time.monotonic()
            features, plan = planner.plan(
                augmented, visuals, bodies, k, "closed_loop", self.args.max_new_tokens
            )
            from .inference_config_realization import conditioning_metadata

            plan.update(
                conditioning_metadata(
                    "closed_loop", self.args.realization_inference_mode
                )
            )
            plan.update(
                conditioning_policy="ntp_exact_token_prefixes_v1",
                h50_strategy="ntp_block_prefixes_v1",
                conditioning_layout=None,
                assistant_text_source="intent_ntp_exact_tokens",
            )
            bodies[k] = list(plan["body_ids"])
            prefixes.append(features)
            plans.append(plan)
            from .checkpoint_realization import rng_state

            seconds = time.monotonic() - started
            state.publish_record(
                local,
                "conditions",
                k,
                dict(
                    features=features,
                    plan=plan,
                    rng=rng_state(),
                    intent_seconds=seconds,
                ),
            )
            emit(
                phase="conditions",
                completed_conditions=k + 1,
                total_conditions=cache["available_blocks"],
                input_tokens=plan["input_tokens"],
                intent_seconds=seconds,
            )
        return dict(features=conditioning.combine_block_features(prefixes), plans=plans)


def media_io():
    path = (
        Path(__file__).resolve().parents[2]
        / "projects/RAVEN-main/scripts/inference/utils/media.py"
    )
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location("_realization_media_io", path)
    )
    module.__spec__.loader.exec_module(module)
    return module


def decoder_identity(args, codec):
    """Decoder execution identity, separate from immutable training codec metadata."""
    backend = getattr(args, "decode_backend", "standard")
    if backend not in ("standard", "stream") or (
        backend == "stream" and args.decode != "duo_5_17"
    ):
        raise ValueError("Invalid decoder backend/codec combination")
    if codec.get("mode", "old") != args.decode:
        raise ValueError("Decoder codec differs from selected input codec")
    root = Path(__file__).resolve().parents[1] / "models/duo_realization/modules"
    sources = [root / "video_vae.py", root / "vae_loading.py"]
    sources += sorted(
        p for p in (root / "native_vae").rglob("*") if p.suffix in (".py", ".yaml")
    )
    settings = dict(
        autocast_dtype=codec.get("video_autocast_dtype", "torch.bfloat16"),
        decoder_weights_dtype=codec.get("vae_dtype", "torch.float32"),
        encoder_weights_dtype=codec.get("vae_dtype", "torch.float32"),
        latent_denormalization_dtype="torch.float32",
        reference="raw_window_reference" if args.decode == "old" else "geometry_only",
        publication=(
            "delayed_next_two_eof_last_two"
            if args.decode == "duo_5_17"
            else "immediate"
        ),
    )
    if args.decode == "sana_1_4":
        sources += [
            root / "sana/causal_vit_decoder.py",
            root / "sana/decoder_config.json",
        ]
        settings.update(
            chunk=1,
            cache_latents=4,
            frames_to_trim=0,
            reference="kv_warmup_discard_rgb",
        )
    if backend == "stream":
        sources += [root / "realization_streaming_decoder.py"]
        sources += sorted(
            p
            for p in (root / "realization_streaming").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        )
        settings.update(
            decoder_weights_dtype="torch.bfloat16",
            autocast_dtype=None,
            publication="immediate_1_4_4_4_4",
            first_refresh=True,
            compensate=True,
            compile=False,
        )
    return dict(
        format="realization-decoder-runtime-v1",
        backend=backend,
        codec=deepcopy(codec),
        settings=settings,
        implementation={str(p.relative_to(root)): state.digest(p) for p in sources},
    )
