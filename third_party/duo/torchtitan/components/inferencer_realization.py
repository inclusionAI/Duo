"""Staged Realization inference; every case group owns its models, KV and recovery."""

import json
import os
import random
import signal
import subprocess
import time
import uuid
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from data.inference_budget import (
    budget_index,
    causal_start,
    cost,
    kv_workspace,
    prefix_blocks,
)
from data.inference_session import make_joint_session, make_session
from torchtitan.components import inference_state_realization as state
from torchtitan.components import storage_realization
from torchtitan.components.checkpoint_realization import (
    load_effective,
    on_rank_zero,
    restore_rng,
    rng_state,
)
from torchtitan.components.inference_runtime_realization import (
    FrozenRuntime,
    Residency,
    media_io,
)
from torchtitan.components.metrics_realization import RealizationLog
from torchtitan.models.duo_realization.modules.args import RealizationModelArgs
from torchtitan.models.duo_realization.modules.codec_spec import (
    ceil_fraction,
    spec,
    validate_codec,
)
from torchtitan.models.duo_realization.modules.model import RealizationTransformer
from torchtitan.models.duo_realization.modules.rope import checkpoint_rope, resolve_rope
from torchtitan.models.duo_realization.modules.weights import (
    configure_trainables,
    initialize_added,
)
from torchtitan.models.duo_realization.pipelines.inference import (
    generate_joint_latents,
    generate_latents,
)
from torchtitan.models.duo_realization.pipelines.streaming import (
    BudgetMediaSession,
    condition,
    shared_condition,
    timed,
)


def load_model(args, dims, device):
    checkpoint = storage_realization.current().resolve(args.checkpoint)
    manifest = checkpoint.manifest
    args.rope = resolve_rope(
        getattr(args, "rope", None), checkpoint_rope(manifest["identity"])
    )
    from torchtitan.models.duo_realization.modules.adaln_timestep import (
        checkpoint_adaln_timestep,
        resolve_adaln_timestep,
    )

    args.checkpoint_adaln_timestep = checkpoint_adaln_timestep(manifest["identity"])
    args.adaln_timestep = resolve_adaln_timestep(
        getattr(args, "adaln_timestep", None), manifest["identity"]
    )
    args.checkpoint_intent = manifest.get("identity", {}).get("qwen", {})
    if (args.checkpoint_intent.get("mode") == "origin") != getattr(
        args, "origin_intent", False
    ):
        raise ValueError("Checkpoint Intent mode differs")
    args.checkpoint_codec = validate_codec(
        manifest.get("identity", {}), getattr(args, "decode", "old")
    )
    if (
        manifest["stage"] != args.stage
        or args.checkpoint_family not in manifest["models"]
    ):
        raise ValueError("Checkpoint stage/family differs")
    entry = manifest["models"][args.checkpoint_family]
    model_args = RealizationModelArgs(
        values={k: v for k, v in entry["config"].items() if not k.startswith("_")}
    )
    cp = dims.cp
    if model_args.values["num_attention_heads"] % cp:
        raise ValueError("Checkpoint head count does not divide CP")
    with torch.device("meta"):
        model = RealizationTransformer(model_args)
    configure_trainables(
        model,
        entry["mode"],
        **(entry["lora"] or dict(rank=128, alpha=128)),
        trainable=False,
        dtype=getattr(torch, args.dtype),
        role_embedding=entry.get("role_embedding", True),
    )
    selected = None
    saved = None
    if getattr(args, "use_saved_adaln_embedding", False):
        from torchtitan.models.duo_realization.modules.adaln_embedding import (
            SavedAdaLN,
            binding,
            remove_adaln_weights,
        )

        saved = SavedAdaLN(
            args.adaln_embedding_dir,
            expected_binding=binding(checkpoint, args.checkpoint_family, args.stage),
            dtype=args.dtype,
            device=device,
            steps=args.steps,
            shift=args.flow_shift,
        )
        selected = remove_adaln_weights(model)
    # Inference has no sharding/mixed-precision wrappers: configure_trainables
    # already preserves BF16 bases and FP32 projections/adapters/roles.
    model.attention_backend = args.attention
    model.attention_deterministic = False
    model.activation_checkpointing = False
    if cp > 1:
        if not dist.is_initialized() or dist.get_world_size() != cp:
            raise ValueError("CP communication must be exactly one case group")
        model.cp_group = dist.group.WORLD
    model.to_empty(device=device)
    model.init_weights(buffer_device=device)
    initialize_added(model, args.seed)
    source = load_effective(
        model,
        args.checkpoint,
        args.checkpoint_family,
        args.stage,
        parameter_names=selected,
    )
    if saved is not None:
        model.saved_adaln_embedding = saved
        args.adaln_embedding_identity = saved.identity
    if cp > 1:
        from .inference_numerics_realization import install_cp_linear_shapes

        install_cp_linear_shapes(model, cp)
    return model.eval().requires_grad_(False), source, entry["mode"]


class FatalInferenceError(RuntimeError):
    pass


class RealizationInferencer:
    def __init__(self, args, device=None):
        self.args = args
        args.managed_residency = True
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = args.context_parallel_degree
        self.device = torch.device(
            device or ("cuda:" + os.environ.get("LOCAL_RANK", "0"))
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group(
                "cpu:gloo,cuda:nccl" if self.device.type == "cuda" else "gloo"
            )
        if self.world == 1 and dist.is_initialized():
            raise ValueError("CP=1 must run without distributed initialization")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        storage_realization.configure(
            cache=args.checkpoint_cache,
            reserve_bytes=int(args.checkpoint_reserve_gb * 1e9),
            local_world=self.world,
        )
        checkpoint = storage_realization.current().resolve(args.checkpoint)
        manifest = checkpoint.manifest
        args.rope = resolve_rope(
            getattr(args, "rope", None), checkpoint_rope(manifest["identity"])
        )
        from torchtitan.models.duo_realization.modules.adaln_timestep import (
            checkpoint_adaln_timestep,
            resolve_adaln_timestep,
        )

        args.checkpoint_adaln_timestep = checkpoint_adaln_timestep(manifest["identity"])
        args.adaln_timestep = resolve_adaln_timestep(
            getattr(args, "adaln_timestep", None), manifest["identity"]
        )
        if (
            manifest["stage"] != args.stage
            or args.checkpoint_family not in manifest["models"]
        ):
            raise ValueError("Checkpoint stage/family differs")
        args.checkpoint_intent = manifest["identity"].get("qwen", {})
        args.checkpoint_codec = validate_codec(manifest["identity"], args.decode)
        if (args.checkpoint_intent.get("mode") == "origin") != args.origin_intent:
            raise ValueError("Checkpoint Intent mode differs")
        self.source = checkpoint.source(args.checkpoint_family, args.stage)
        self.weight_mode = manifest["models"][args.checkpoint_family]["mode"]
        self.training_budget = manifest["identity"]["h3"].get("max_seq_len")
        if args.use_saved_adaln_embedding:
            from torchtitan.models.duo_realization.modules.adaln_embedding import (
                SavedAdaLN,
                binding,
            )

            # Validate the table on CPU even for a decode-only resume.
            table = SavedAdaLN(
                args.adaln_embedding_dir,
                expected_binding=binding(
                    checkpoint, args.checkpoint_family, args.stage
                ),
                dtype=args.dtype,
                steps=args.steps,
                shift=args.flow_shift,
            )
            args.adaln_embedding_identity = table.identity
            del table
        self.model = None
        self.emit = lambda **values: None
        self.residency = Residency(
            self.device, lambda **values: self.emit(**values), args.offload_pin_memory
        )
        self.runtime = (
            FrozenRuntime(args, self.device, self.residency) if self.rank == 0 else None
        )

    def gather_rng(self):
        value = rng_state()
        if not dist.is_initialized():
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def move_realization(self, device):
        self.model.to(device)
        saved = getattr(self.model, "saved_adaln_embedding", None)
        if saved is not None:
            for name in ("timesteps", "blocks", "output"):
                setattr(saved, name, getattr(saved, name).to(device))

    def ensure_realization(self, index=None, wanted=None):
        if self.model is None:
            self.model, _, _ = load_model(
                self.args, SimpleNamespace(cp=self.world), torch.device("cpu")
            )

            def modules():
                saved = getattr(self.model, "saved_adaln_embedding", None)
                return (
                    [self.model]
                    if saved is None
                    else [
                        self.model,
                        SimpleNamespace(
                            parameters=lambda: (
                                saved.timesteps,
                                saved.blocks,
                                saved.output,
                            ),
                            buffers=lambda: (),
                        ),
                    ]
                )

            self.residency.register(
                "realization", self.model, move=self.move_realization, modules=modules
            )
        workspace = 8_000_000_000
        if (
            index is not None
            and self.args.cache_mode == "block"
            and self.args.realization_inference_mode == "causal"
        ):
            c = self.model.config
            workspace = kv_workspace(
                index,
                wanted,
                self.args.max_seq_len,
                c.num_layers,
                c.hidden_size // self.world,
                2 if self.args.dtype == "bfloat16" else 4,
            )
        if index is not None and getattr(self.args, "first_gen_ref", 0):
            from torchtitan.models.duo_realization.pipelines.first_gen_ref import (
                extra_workspace_bytes,
            )

            c = self.model.config
            workspace += extra_workspace_bytes(
                index,
                wanted,
                self.args.first_gen_ref,
                c.num_layers,
                c.hidden_size // self.world,
                2 if self.args.dtype == "bfloat16" else 4,
            )
        self.residency.enter(
            "latents",
            ["realization"],
            dict(video=1, audio=1, intent=2, realization=2),
            workspace_bytes=workspace,
        )
        if self.device.type == "cuda" and not getattr(
            self.model, "_inference_rope_initialized", False
        ):
            c = self.model.config
            self.model.rope.inv_freq = 1 / (
                c.rope_theta
                ** (
                    torch.arange(
                        0,
                        2 * c.rope_freq_dim,
                        2,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    / (2 * c.rope_freq_dim)
                )
            )
            self.model._inference_rope_initialized = True

    def run_sample(self, record, root, local, resume):
        args = self.args
        started = time.monotonic()
        self.cache_restore_seconds = 0.0
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        identity = on_rank_zero(
            lambda: state.signature(args, self.source, record, self.world)
        )
        leader_state = {}

        def open_case():
            value = state.open_staged(local, root, identity, resume)
            state.verify_decoder_runtime(
                local,
                args,
                required=bool(value["decode"])
                or (Path(local) / "decode-complete").exists(),
            )
            leader_state["saved"] = value
            # Only the leader owns waveform recovery. Peers need latent/RNG
            # state and progress counts, never RGB or decoded audio.
            return dict(
                conditions=[{} for _ in value["conditions"]],
                latents=value["latents"],
                decode=[{} for _ in value["decode"]],
            )

        shared_state = on_rank_zero(open_case)
        saved = leader_state.get("saved", shared_state)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        config = dict(
            vars(args),
            framework="torchtitan",
            kind="inference",
            finetune_mode=self.weight_mode,
            world_size=self.world,
            nodes=1,
            gpus_per_node=self.world,
            sharding_mode="replicated",
            sample_id=record["sample_id"],
            split=record["split"],
            media=record.get("media", {}),
            checkpoint_source=self.source,
            conditioning="staged_frame_v5",
            training_max_seq_len=self.training_budget,
            inference_max_seq_len=args.max_seq_len,
            preparation=record.get("preparation", {}),
            audio_provenance="turn_gated_mixture_approximation",
        )
        log = RealizationLog(root, config, self.rank, resume=resume)
        self.log = log
        from torchtitan.components.telemetry_realization import resources

        monitor = (
            resources(root, args.local_dir, args.resource_interval)
            if self.rank == 0
            else None
        )

        def emit(**values):
            phase = values.get("phase", "initializing")
            if monitor:
                monitor.set_phase(phase)
            log.event("phase_progress", **values)
            if self.rank == 0:
                print(
                    json.dumps(dict(sample=record["sample_id"], **values)), flush=True
                )

        self.emit = emit
        try:
            complete = on_rank_zero(lambda: state.staged_phase(local, "decode"))
            if complete is not None:
                on_rank_zero(lambda: self.verify_completion(root, complete))
                on_rank_zero(
                    lambda: state.atomic_json(
                        Path(root) / "result.json", complete["result"]
                    )
                )
                return complete["result"]
            observations = on_rank_zero(
                lambda: state.staged_phase(local, "observations")
            )
            text = on_rank_zero(lambda: state.staged_phase(local, "conditions"))
            if text is None:
                emit(phase="conditions", completed_conditions=len(saved["conditions"]))

                def prepare():
                    prepared_at = time.monotonic()
                    item = self.runtime.read_features(record)
                    cache = observations or self.runtime.observations(record, item)
                    if observations is None:
                        state.publish_staged_phase(local, "observations", cache)
                    count = min(
                        args.max_blocks or cache["available_blocks"],
                        cache["available_blocks"],
                    )
                    if args.realization_inference_mode == "bidirectional":
                        need = (
                            cache["valid_frames"] + spec(args.decode).frames - 1
                        ) // spec(args.decode).frames
                        if cache["available_blocks"] < need:
                            raise ValueError("Cached conditions do not cover GT tail")
                        count = min(count, need)
                    text = self.runtime.prepare(
                        record, cache, item, local, saved["conditions"], count, emit
                    )
                    text.update(cache=cache, requested_blocks=count)
                    text["condition_prepare_seconds"] = time.monotonic() - prepared_at
                    state.publish_staged_phase(local, "conditions", text)
                    return text

                text = on_rank_zero(prepare)
            cache = text["cache"]
            features = text["features"]
            index = budget_index(features, args.height, args.width, args.decode)
            wanted = text["requested_blocks"]
            if args.realization_inference_mode == "bidirectional":
                wanted = prefix_blocks(index, args.max_seq_len, wanted)
            else:
                for k in range(wanted):
                    first = causal_start(index, k, args.max_seq_len)
                    if getattr(args, "kv_reset", 0):
                        from torchtitan.models.duo_realization.pipelines.kv_reset import (
                            retained_window_start,
                        )

                        retained_window_start(k, args.kv_reset, first)
            coverage = dict(
                available_conditions=cache["available_blocks"],
                selected_conditions=wanted,
                condition_exhausted=wanted == cache["available_blocks"],
                available_condition_range=dict(
                    start_block=0,
                    stop_block=cache["available_blocks"],
                    start_ms=0.0,
                    end_ms=cache["blocks"][-1]["end_ms"],
                ),
                selected_condition_range=dict(
                    start_block=0,
                    stop_block=wanted,
                    start_ms=0.0,
                    end_ms=cache["blocks"][wanted - 1]["end_ms"],
                ),
                selection_reason=(
                    "max_blocks"
                    if args.max_blocks and wanted < cache["available_blocks"]
                    else (
                        "budget_prefix"
                        if wanted < text["requested_blocks"]
                        else "all_conditions"
                    )
                ),
            )
            latent_done = on_rank_zero(lambda: state.staged_phase(local, "latents"))
            latents = [p["latent"] for p in saved["latents"]]
            if len(latents) > wanted or len(saved["decode"]) > len(latents):
                raise ValueError("Staged progress is inconsistent")
            if (
                latent_done is None
                or "latents" in latent_done
                and len(latents) < wanted
            ):
                emit(
                    phase="latents",
                    completed_latents=len(latents),
                    selected_conditions=wanted,
                )
                if latent_done is None:
                    self.ensure_realization(index, wanted)
                if args.realization_inference_mode == "bidirectional":
                    if latent_done is None:
                        session = make_joint_session(cache, features, wanted)
                        metrics = {}
                        with timed(metrics, "realization_seconds", self.device):
                            joint, detail = generate_joint_latents(
                                self.model, session, args, self.device
                            )
                        metrics.update(
                            detail,
                            training_equivalent_rows=cost(index, 0, wanted)[
                                "padded_rows"
                            ],
                        )
                        latent_done = dict(
                            latents=joint, metrics=metrics, rng=self.gather_rng()
                        )
                        on_rank_zero(
                            lambda: state.publish_staged_phase(
                                local, "latents", latent_done
                            )
                        )
                    for k in range(len(latents), wanted):
                        payload = dict(
                            latent=latent_done["latents"][k],
                            rng=latent_done["rng"],
                            metrics=latent_done["metrics"],
                            media_window=dict(start_block=0, stop_block=wanted),
                        )
                        on_rank_zero(
                            lambda k=k, payload=payload: state.publish_record(
                                local, "latents", k, payload
                            )
                        )
                        saved["latents"].append(payload)
                        latents.append(payload["latent"])
                else:
                    session = None
                    if args.cache_mode == "block":
                        session = BudgetMediaSession(
                            self.model, cache, args, self.device
                        )
                        restored = session.replay(
                            saved["latents"],
                            shared_condition(features),
                            features=features,
                        )
                        self.cache_restore_seconds = restored["cache_restore_seconds"]
                        if saved["latents"]:
                            emit(
                                phase="latents",
                                completed_latents=len(latents),
                                selected_conditions=wanted,
                                **restored,
                            )
                    if saved["latents"]:
                        restore_rng(saved["latents"][-1]["rng"][self.rank])
                    for k in range(len(latents), wanted):
                        metrics = {}
                        if session:
                            latent, metrics, part = session.generate(features, k)
                            window = metrics["media_window"]
                        else:
                            first = causal_start(index, k, args.max_seq_len)
                            view = make_session(
                                cache, features, latents, k, window_start_block=first
                            )
                            with timed(metrics, "realization_seconds", self.device):
                                latent, detail = generate_latents(
                                    self.model, view, k, args, self.device
                                )
                            metrics.update(
                                detail,
                                training_equivalent_rows=cost(index, first, k + 1)[
                                    "padded_rows"
                                ],
                            )
                            part = condition(features, k)
                            window = view["media_window"]
                        payload = dict(
                            latent=latent,
                            media_cache_condition=part,
                            rng=self.gather_rng(),
                            metrics=metrics,
                            media_window=window,
                        )
                        on_rank_zero(
                            lambda k=k, payload=payload: state.publish_record(
                                local, "latents", k, payload
                            )
                        )
                        saved["latents"].append(payload)
                        latents.append(latent)
                        emit(
                            phase="latents",
                            completed_latents=k + 1,
                            selected_conditions=wanted,
                            **metrics,
                        )
                    if session:
                        session.kv = None
                        del session
                    latent_done = dict(blocks=wanted)
                    on_rank_zero(
                        lambda: state.publish_staged_phase(
                            local, "latents", latent_done
                        )
                    )
            if len(latents) != wanted:
                raise ValueError("Completed latent phase has missing block records")
            coverage["consumed_conditions"] = wanted
            local_memory = (
                dict(
                    rank=self.rank,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                )
                if self.device.type == "cuda"
                else dict(rank=self.rank)
            )
            group_memory = [local_memory]
            if dist.is_initialized():
                group_memory = [None] * self.world
                dist.all_gather_object(group_memory, local_memory)
            self.latent_group_memory = group_memory
            emit(
                phase="decode",
                completed_latents=wanted,
                decoded_latents=len(saved["decode"]),
                **coverage,
            )
            if dist.is_initialized():
                dist.barrier()
            result = on_rank_zero(
                lambda: self.decode(
                    record,
                    cache,
                    text["plans"],
                    saved,
                    latents,
                    coverage,
                    root,
                    local,
                    started,
                    emit,
                )
            )
            if dist.is_initialized():
                dist.barrier()
            log.event("completed", phase="exited", **result)
            return result
        except BaseException as exc:
            log.event("failed", phase="exited", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, (torch.cuda.OutOfMemoryError, FatalInferenceError)) or (
                isinstance(exc, RuntimeError)
                and any(
                    word in str(exc).lower() for word in ("cuda", "nccl", "collective")
                )
            ):
                raise FatalInferenceError(str(exc)) from exc
            raise
        finally:
            if self.rank == 0 and hasattr(self.runtime, "close_decoder"):
                self.runtime.close_decoder()
            if monitor:
                monitor.close()
            self.emit = lambda **values: None

    @staticmethod
    def verify_completion(root, complete):
        for name, checksum in complete["files"].items():
            if state.digest(Path(root) / name) != checksum:
                raise ValueError("Completed media/export checksum differs")
        if not complete.get("eof_complete"):
            raise ValueError("Missing EOF completion")

    def publish_packet(self, root, packet, waves, cache, plans, coverage):
        k = packet["block"]
        folder = Path(root) / f"block-{k:05d}"
        folder.mkdir(exist_ok=True)
        marker = folder / "complete.json"
        if marker.exists():
            old = json.loads(marker.read_text())
            publication = old.get("publication")
            if publication is None or old.get("plan") != plans[k]:
                raise ValueError("Published block has incompatible recovery metadata")
            if any(
                state.digest(Path(root) / name) != checksum
                for name, checksum in publication["sha256"].items()
            ):
                raise ValueError("Published block media checksum differs")
            return dict(publication, detail=old)
        frames = packet["frames"]
        if self.args.realization_inference_mode == "bidirectional":
            retain = min(
                len(frames), cache["valid_frames"] - cache["blocks"][k]["frame_start"]
            )
            frames = frames[:retain]
        if not len(frames):
            raise ValueError("No valid frames in a selected block")
        io = media_io()
        started = time.monotonic()
        identifier = uuid.uuid4().hex
        wav, mp4 = folder / (identifier + ".tmp.wav"), folder / (
            identifier + ".tmp.mp4"
        )
        pcm = io.write_wav(wav, waves[k])
        io.write_video(mp4, frames, wav, self.args.ffmpeg)
        wav.replace(folder / "assistant.wav")
        mp4.replace(folder / "assistant.mp4")
        paths = dict(
            video=str(folder / "assistant.mp4"), audio=str(folder / "assistant.wav")
        )
        hashes = {
            f"{folder.name}/{n}": state.digest(folder / n)
            for n in ("assistant.wav", "assistant.mp4")
        }
        detail = dict(
            block=k,
            plan=plans[k],
            media=paths,
            video_status="complete",
            metrics=dict(packet["metrics"], **pcm),
            start_ms=cache["blocks"][k]["start_ms"],
            end_ms=(
                min(cache["blocks"][k]["frame_stop"], cache["valid_frames"]) * 1000 / 24
                if self.args.realization_inference_mode == "bidirectional"
                else cache["blocks"][k]["end_ms"]
            ),
            effective_video_frames=len(frames),
            effective_audio_samples=int(waves[k].shape[-1]),
            **coverage,
        )
        detail["metrics"]["media_write_seconds"] = time.monotonic() - started
        publication = dict(block=k, media=paths, sha256=hashes)
        detail["publication"] = publication
        state.atomic_json(folder / "complete.json", detail)
        self.log.event("block_complete", **detail)
        return dict(publication, detail=detail)

    def decode(
        self, record, cache, plans, saved, latents, coverage, root, local, started, emit
    ):
        runtime = self.runtime
        video = runtime.load_video()
        audio = runtime.load_audio()
        from data.inference_media import runtime_identity, validate_checkpoint_codec

        codec = video.identity()
        validate_checkpoint_codec(self.args.checkpoint_codec, codec)
        runtime_identity(local, codec, cache["intent"], cache)
        from .inference_runtime_realization import decoder_identity

        decoder = decoder_identity(self.args, codec)
        state.bind_decoder_runtime(local, decoder)
        workspace = 8_000_000_000
        if hasattr(video, "decode_workspace_bytes"):
            workspace += video.decode_workspace_bytes(self.args.height, self.args.width)
        self.residency.enter(
            "decode",
            ["video", "audio"],
            dict(intent=1, realization=2, video=3, audio=3),
            workspace_bytes=workspace,
        )
        decoded = saved["decode"]
        waves = [p["wave"] for p in decoded]
        restore_timing = {}
        with timed(
            restore_timing, "decoder_restore_seconds", self.device, collective=False
        ):
            video.restore_decode(
                cache["reference"], [p["video"] for p in latents[: len(decoded)]]
            )
        emit(phase="decode", **restore_timing)
        for k in range(len(decoded), len(latents)):
            timing = {}
            row = cache["blocks"][k]
            samples = row["audio_stop_sample"] - row["audio_start_sample"]
            with timed(timing, "decode_seconds", self.device, collective=False):
                packets = video.decode_block(latents[k]["video"])
                wave_value, metrics = audio.decode_block(latents[k]["audio"], samples)
            if self.args.realization_inference_mode == "bidirectional":
                from fractions import Fraction

                stop = min(row["frame_stop"], cache["valid_frames"])
                keep = (
                    ceil_fraction(Fraction(stop * 32000, 24))
                    - row["audio_start_sample"]
                )
                wave_value = wave_value[:, :keep]
            metrics.update(
                timing,
                seam_delta=(
                    float((wave_value[:, 0] - waves[-1][:, -1]).abs().max())
                    if waves
                    else None
                ),
            )
            waves.append(wave_value)
            publications = [
                self.publish_packet(root, p, waves, cache, plans, coverage)
                for p in packets
            ]
            payload = dict(wave=wave_value, publications=publications, metrics=metrics)
            state.publish_record(local, "decode", k, payload)
            decoded.append(payload)
            for p in publications:
                emit(
                    phase="decode",
                    published_media_blocks=p["block"] + 1,
                    decoded_latents=k + 1,
                    **coverage,
                )
            emit(
                phase="decode",
                decoded_latents=k + 1,
                selected_conditions=len(latents),
                **metrics,
            )
        from data.inference_media import check_flush

        already_flushed = (Path(root) / "video_flush.json").is_file()
        if already_flushed:
            check_flush(root, len(latents))
        packets = video.finish_decode()
        publications = (
            []
            if already_flushed
            else [
                self.publish_packet(root, p, waves, cache, plans, coverage)
                for p in packets
            ]
        )
        from data.inference_media import mark_flush

        if getattr(video, "decode_delayed", video.spec.delayed) and not already_flushed:
            mark_flush(root, len(latents), publications)
        for p in publications:
            emit(
                phase="decode",
                published_media_blocks=p["block"] + 1,
                decoded_latents=len(latents),
                **coverage,
            )
        media = self.concatenate(root, len(latents))
        export = None
        if self.args.save_latents:
            from data.inference_export import publish_staged_export

            text = state.staged_phase(local, "conditions")
            export = publish_staged_export(
                self.args.latent_export_dir,
                cache,
                latents,
                text["features"],
                plans,
                saved["latents"],
                record,
                self.args,
                self.source,
                coverage,
                local,
            )
        frames = (
            min(len(latents) * spec(self.args.decode).frames, cache["valid_frames"])
            if self.args.realization_inference_mode == "bidirectional"
            else len(latents) * spec(self.args.decode).frames
        )
        result = dict(
            completed_blocks=len(latents),
            video_blocks=video.emitted,
            total_blocks=cache["available_blocks"],
            duration_ms=frames * 1000 / 24,
            effective_video_frames=frames,
            internal_video_frames=len(latents) * spec(self.args.decode).frames,
            codec=codec,
            decoder=decoder,
            decode_backend=getattr(self.args, "decode_backend", "standard"),
            media=media,
            quality="not_accepted",
            stage=self.args.stage,
            split=record["split"],
            mode=self.args.mode,
            steps=self.args.steps,
            realization_inference_mode=self.args.realization_inference_mode,
            sample_seconds=time.monotonic() - started,
            training_max_seq_len=self.training_budget,
            inference_max_seq_len=self.args.max_seq_len,
            **coverage,
        )
        result["memory"] = dict(
            latent_ranks=getattr(self, "latent_group_memory", []),
            leader_peak_allocated_bytes=(
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            ),
            leader_peak_reserved_bytes=(
                torch.cuda.max_memory_reserved(self.device)
                if self.device.type == "cuda"
                else 0
            ),
        )
        result["effective_audio_samples"] = sum(int(w.shape[-1]) for w in waves)
        result["phase_seconds"] = dict(
            conditions=state.staged_phase(local, "conditions").get(
                "condition_prepare_seconds", 0.0
            ),
            realization=(
                saved["latents"][0]["metrics"].get("realization_seconds", 0.0)
                if self.args.realization_inference_mode == "bidirectional"
                else sum(
                    p["metrics"].get(
                        "realization_seconds", p["metrics"].get("denoise_seconds", 0.0)
                    )
                    + p["metrics"].get("cache_prefill_seconds", 0.0)
                    + p["metrics"].get("cache_evict_seconds", 0.0)
                    + p["metrics"].get("cache_commit_seconds", 0.0)
                    for p in saved["latents"]
                )
            ),
            cache_restore=getattr(self, "cache_restore_seconds", 0.0),
            vae=sum(p["metrics"].get("decode_seconds", 0.0) for p in decoded),
            decoder_restore=restore_timing["decoder_restore_seconds"],
        )
        if export:
            result["saved_latents"] = export
        paths = [Path(p) for p in media.values()]
        paths += [
            Path(root) / f"block-{k:05d}" / n
            for k in range(len(latents))
            for n in ("assistant.wav", "assistant.mp4", "complete.json")
        ]
        if getattr(video, "decode_delayed", video.spec.delayed):
            paths.append(Path(root) / "video_flush.json")
        if export:
            paths.extend(Path(export).iterdir())
        files = {
            (
                str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
            ): state.digest(p)
            for p in paths
            if p.is_file()
        }
        complete = dict(result=result, files=files, eof_complete=True)
        state.publish_staged_phase(local, "decode", complete)
        state.atomic_json(Path(root) / "result.json", result)
        return result

    def concatenate(self, root, count):
        root = Path(root)
        target = root / "assistant.tmp.wav"
        with wave.open(str(target), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(32000)
            for k in range(count):
                with wave.open(
                    str(root / f"block-{k:05d}" / "assistant.wav")
                ) as source:
                    if (
                        source.getnchannels(),
                        source.getsampwidth(),
                        source.getframerate(),
                    ) != (2, 2, 32000):
                        raise ValueError("Block WAV format differs")
                    out.writeframes(source.readframes(source.getnframes()))
        listing = root / ".concat.txt"
        entries = []
        for k in range(count):
            folder = root / f"block-{k:05d}"
            frames = json.loads((folder / "complete.json").read_text())[
                "effective_video_frames"
            ]
            path = str((folder / "assistant.mp4").resolve()).replace("'", "'\\''")
            entries.append(f"file '{path}'\nduration {frames/24:.12f}\n")
        listing.write_text("".join(entries))
        silent = root / "assistant.video.tmp.mp4"
        final = root / "assistant.tmp.mp4"

        def ffmpeg(values):
            subprocess.run(
                [self.args.ffmpeg, "-nostdin", "-v", "error", "-y", *values],
                check=True,
                stderr=subprocess.PIPE,
                timeout=180,
            )

        try:
            ffmpeg(
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "copy",
                    str(silent),
                ]
            )
            ffmpeg(
                [
                    "-i",
                    str(silent),
                    "-i",
                    str(target),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-video_track_timescale",
                    "24",
                    "-movflags",
                    "+faststart",
                    str(final),
                ]
            )
            target.replace(root / "assistant.wav")
            final.replace(root / "assistant.mp4")
        finally:
            listing.unlink(missing_ok=True)
            silent.unlink(missing_ok=True)
        return dict(
            video=str(root / "assistant.mp4"), audio=str(root / "assistant.wav")
        )

    def run(self):
        worker = json.loads(Path(self.args.case_worker_config).read_text())
        rows = []
        status_path = Path(worker["result"])
        previous = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        try:
            for record in worker["records"]:
                suffix = Path("samples") / record["split"] / record["sample_id"]
                root = Path(self.args.output_dir) / suffix
                local = Path(self.args.local_dir) / "inference" / suffix
                resume = self.args.resume and on_rank_zero(
                    lambda: (local / "config.json").is_file()
                )
                from data.inference_export import destination

                self.args.latent_export_dir = str(
                    destination(self.args.output_dir, record, worker["multiple"])
                )
                row = dict(
                    split=record["split"],
                    sample_id=record["sample_id"],
                    status="running",
                )
                rows.append(row)
                on_rank_zero(lambda: state.atomic_json(status_path, dict(samples=rows)))
                try:
                    result = self.run_sample(record, root, local, resume)
                    row.update(status="completed", result=result)
                except FatalInferenceError as exc:
                    row.update(status="failed", error=str(exc), fatal=True)
                    on_rank_zero(
                        lambda: state.atomic_json(status_path, dict(samples=rows))
                    )
                    raise
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                on_rank_zero(lambda: state.atomic_json(status_path, dict(samples=rows)))
            if any(r["status"] != "completed" for r in rows):
                raise RuntimeError("One or more case(s) failed")
        finally:
            signal.signal(signal.SIGTERM, previous)
            if dist.is_initialized():
                dist.destroy_process_group()
