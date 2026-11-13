"""Reference DCP backend for the Duo 5-to-17 pipeline."""

import json
import logging
import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import msgspec
import numpy as np
import torch
from data.dataloader_realization import RealizationSequence
from data.dataloader_realization_sequence import block_seed
from data.inference_budget import budget_index, cost
from data.inference_options import ROLLING_MEDIA_CACHE_POLICY
from sglang_omni.models.duo.kernels import use_realization_kernels
from sglang_omni.models.duo.live import LiveSessionInput
from sglang_omni.models.duo.live_intent import LiveIntentPlanner, LiveIntentState
from sglang_omni.models.duo.model import RealizationSegment
from sglang_omni.models.duo.modelopt import native_dcp_from_modelopt
from sglang_omni.models.duo.native_dcp import CachedTimestepModel, NativeDCPRealization
from sglang_omni.models.duo.native_execution import NativeDCPOptions
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks
from torchtitan.components.inferencer_realization import (
    RealizationInferencer,
    load_model,
)
from torchtitan.models.duo_realization.modules import intent_conditioning
from torchtitan.models.duo_realization.pipelines.inference import sample_sequence
from torchtitan.models.duo_realization.pipelines.streaming import (
    BudgetMediaSession,
    condition,
    shared_condition,
)
from torchtitan.models.duo_realization.schedulers.dmd import rollout_grid
from torchtitan.models.duo_realization.schedulers.flow import add_noise, to_x0

logger = logging.getLogger(__name__)


class FrameIntentPlanner(LiveIntentPlanner):
    """Use training's exact ChatML and selected H50 rows with incremental KV."""

    def visual_patch(self, frames: np.ndarray) -> dict[str, object]:
        return intent_conditioning.visual_patch(self.processor, frames)

    def system_prompt_fragments(self) -> tuple[str, str]:
        return (
            intent_conditioning.INTRO.replace(
                "<chunk_160>", self.tokenizer.duplex_chunk_token, 1
            ),
            intent_conditioning.INSTRUCTION + intent_conditioning.SPEECH_CUE,
        )

    def serialize(
        self, state: LiveIntentState, block_index: int, block_duration_ms: float
    ) -> dict[str, object]:
        prefix = intent_conditioning.serialize(
            state.record,
            self.tokenizer,
            state.visuals,
            stop_block=block_index,
            assistant_bodies=state.bodies,
            generation=True,
        )
        state.description_tokens = sum(
            group["end"] - group["start"]
            for group in prefix["text_groups"]
            if group["section"] == "assistant_description"
        )
        return prefix


class NativeBudgetMediaSession(BudgetMediaSession):
    def __init__(
        self,
        model: torch.nn.Module,
        features: dict[str, object],
        args: Namespace,
        device: torch.device,
        *,
        reference_window_chunks: int = 0,
    ) -> None:
        self.reference_window_chunks = reference_window_chunks
        self.defer_finite_checks = (
            isinstance(model, NativeDCPRealization)
            and model.options.defer_finite_checks
            and args.stage == "dmd"
        )
        if (
            isinstance(model, NativeDCPRealization)
            and model.options.cache_timestep_projections
            and args.stage == "dmd"
        ):
            schedule = rollout_grid(args.steps, shift=args.flow_shift, device=device)
            model = CachedTimestepModel(model, schedule)
        else:
            pass
        super().__init__(model, features, args, device)

        self.cuda_timings: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self.pending_kv_reset = False

    def window_start(self, block: int) -> int:
        return max(
            self.start,
            super().window_start(block),
            (
                block + 1 - self.reference_window_chunks
                if self.reference_window_chunks
                else 0
            ),
        )

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            end.record(torch.cuda.current_stream(self.device))
            self.cuda_timings[name] = (start, end)

    def initialize(self, part: dict[str, object], metrics: dict[str, object]) -> None:
        if self.kv is not None:
            return
        else:
            pass
        full = self.sequence(part)
        prefix = full.select(lambda group: group["kind"] in ("reference", "shared"))
        prefix = replace(
            prefix,
            refiner_text=prefix.refiner_text[:, : len(self.shared["h50"])],
            refiner_groups=tuple(
                group for group in prefix.refiner_groups if group["kind"] == "shared"
            ),
        )
        with self.measure("cache_prefill_seconds"):
            _, self.kv = self.model(
                prefix,
                torch.full((2,), self.clean_time, device=self.device),
                commit=True,
            )

    def current(
        self, part: dict[str, object], metrics: dict[str, object]
    ) -> RealizationSequence:
        if self.failed or part["block"] != self.next_block:
            raise ValueError("Invalid/noncontiguous media cache session")
        else:
            pass
        block = part["block"]
        start = block if self.pending_kv_reset else self.window_start(block)
        self.initialize(part, metrics)
        previous = self.kv
        evicted = {
            group["block"]
            for group in previous.groups
            if group["kind"] == "clean" and group["block"] < start
        }
        started = time.monotonic()
        self.kv = previous.evict(start)
        self.start = start
        metrics.update(
            realization_kv_reset=self.pending_kv_reset,
            denoise_cache_rows=self.kv.length,
            cache_evict_seconds=time.monotonic() - started,
            cache_evicted_blocks=len(evicted),
            cache_evicted_block_ids=sorted(evicted),
            cache_evicted_rows=previous.length - self.kv.length,
        )
        self.pending_kv_reset = False
        return self.sequence(part).select(
            lambda group: group["kind"] in ("text", "noisy")
        )

    def commit(
        self,
        sequence: RealizationSequence,
        values: tuple[torch.Tensor, torch.Tensor],
        metrics: dict[str, object],
    ) -> None:
        # current() already selected and padded the complete block; no row selection is needed.
        clean = replace(
            sequence.with_media(*values),
            groups=tuple(
                dict(group, kind="clean") if group["kind"] == "noisy" else group
                for group in sequence.groups
            ),
            time_ids=torch.zeros_like(sequence.time_ids),
        )
        with self.measure("cache_commit_seconds"):
            _, self.kv = self.model(
                clean,
                torch.full((2,), self.clean_time, device=self.device),
                cache=self.kv,
                commit=True,
            )
        self.next_block += 1

    def sample_dmd(
        self, sequence: RealizationSequence, block: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, int | float | str | bool]]:
        current = sequence.select(
            lambda group: group["block"] == block and group["kind"] in ("text", "noisy")
        )
        times = torch.full(
            (len(sequence.seeds) + 1,), self.clean_time, device=sequence.device
        )
        schedule = rollout_grid(
            self.args.steps, shift=self.args.flow_shift, device=sequence.device
        )
        generator = torch.Generator(device=sequence.device).manual_seed(
            block_seed(self.args.seed, self.observations["sample_id"], block)
        )
        state = tuple(
            torch.randn(
                value.shape,
                device=value.device,
                dtype=torch.float32,
                generator=generator,
            )
            for value in (current.video, current.audio)
        )
        finite_predictions: list[torch.Tensor] = []
        for index in range(self.args.steps):
            times[1:] = schedule[index]
            prediction, _ = self.model(current.with_media(*state), times, cache=self.kv)
            finite_predictions.extend(
                torch.isfinite(value).all() for value in prediction
            )
            state = tuple(
                to_x0(value, velocity.float(), schedule[index])
                for value, velocity in zip(state, prediction)
            )
            if index + 1 < self.args.steps:
                state = tuple(
                    add_noise(
                        value,
                        torch.randn(
                            value.shape,
                            device=value.device,
                            dtype=value.dtype,
                            generator=generator,
                        ),
                        schedule[index + 1],
                    )
                    for value in state
                )
            else:
                pass
        finite = torch.stack(
            [*finite_predictions, *(torch.isfinite(value).all() for value in state)]
        )
        # note (kemiao): Reject invalid predictions before clean KV can be committed.
        if not bool(finite.all()):
            valid = finite.tolist()
            for index in range(self.args.steps):
                if not all(valid[2 * index : 2 * index + 2]):
                    raise FloatingPointError(
                        f"Nonfinite Realization prediction at sampling step {index}"
                    )
                else:
                    pass
            raise FloatingPointError("Nonfinite generated Realization latent")
        else:
            pass
        return state, dict(
            denoising_steps=self.args.steps,
            nfe=self.args.steps,
            flow_shift=self.args.flow_shift,
            solver="consistency_renoise",
            cache_mode="rebuild_per_block",
            sequence_rows=sequence.length,
            current_rows=current.length,
            h50_rows=sequence.refiner_text.shape[1],
            generated_history_blocks=block,
            cfg=False,
        )

    @torch.no_grad()
    def generate(
        self, features: dict[str, object], block: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, object], dict[str, object]]:
        if self.index is None:
            self.index = budget_index(
                features, self.args.height, self.args.width, self.geometry.name
            )
        else:
            pass
        if self.shared is None:
            self.shared = shared_condition(features)
        else:
            pass
        part = condition(features, block)
        part["position"] = self.position(block)
        metrics: dict[str, object] = {}
        self.cuda_timings.clear()
        try:
            current = self.current(part, metrics)
            with self.measure("denoise_seconds"):
                if self.sampler:
                    values, denoise = self.sampler(current, self.kv, block)
                elif self.defer_finite_checks:
                    values, denoise = self.sample_dmd(current, block)
                else:
                    values, denoise = sample_sequence(
                        self.model,
                        current,
                        stage=self.args.stage,
                        steps=self.args.steps,
                        shift=self.args.flow_shift,
                        seed=block_seed(
                            self.args.seed, self.observations["sample_id"], block
                        ),
                        history=self.kv,
                        clean_time=self.clean_time,
                        cm_sampling_mode=(
                            self.args.cm_sampling_mode
                            if self.args.stage == "cm"
                            else None
                        ),
                    )
            self.commit(current, values, metrics)
            height, width = self.observations["reference"].shape[-2:]
            video = (
                values[0]
                .reshape(self.geometry.latents, height // 2, width // 2, 24, 2, 2)
                .permute(3, 0, 1, 4, 2, 5)
                .reshape(24, self.geometry.latents, height, width)
            )
            audio = (
                values[1].reshape(2, self.geometry.audio_latents, 32).permute(0, 2, 1)
            )
            # The blocking output copies also complete preceding clean KV writes on this stream.
            latent = dict(
                video=video.cpu().contiguous(), audio=audio.cpu().contiguous()
            )
            metrics.update(
                {
                    name: start.elapsed_time(end) / 1000
                    for name, (start, end) in self.cuda_timings.items()
                }
            )
        except BaseException:
            self.failed = True
            self.kv = None
            raise
        metrics.update(
            denoise,
            cache_mode="block",
            media_cache_policy=ROLLING_MEDIA_CACHE_POLICY,
            cache_rows=self.kv.length,
            cache_bytes=self.kv.nbytes,
            current_rows=current.length,
            h50_rows=current.refiner_text.shape[1],
            cache_committed_block=block,
            generated_history_blocks=block,
            media_window=dict(
                start_block=self.start,
                stop_block=block + 1,
                frames=(block + 1 - self.start) * self.geometry.frames,
            ),
            training_equivalent_rows=cost(self.index, self.start, block + 1)[
                "padded_rows"
            ],
            timing_mode="cuda_event_rank_local",
        )
        self.states.append(dict(latent=latent, media_cache_condition=part))
        return latent, metrics, part


class TokenWindowMediaSession(NativeBudgetMediaSession):
    def __init__(
        self,
        model: torch.nn.Module,
        features: dict[str, object],
        args: Namespace,
        device: torch.device,
        window_tokens: int,
        *,
        reference_window_chunks: int = 0,
    ) -> None:
        super().__init__(
            model,
            features,
            args,
            device,
            reference_window_chunks=reference_window_chunks,
        )
        if window_tokens <= 0:
            raise ValueError("Realization KV window must be positive")
        else:
            self.window_tokens = window_tokens

    def window_start(self, block: int) -> int:
        current = self.index["blocks"][block]
        media_tokens = current["video_rows"] + current["stereo_audio_rows"]
        plan_tokens = current["packed_rows"] - 2 * media_tokens
        available_tokens = (
            self.window_tokens - self.index["shared_rows"] - plan_tokens - media_tokens
        )
        if available_tokens < 0:
            raise ValueError(
                "Realization window cannot fit reference, shared, plan and one media block"
            )
        else:
            history_blocks = available_tokens // media_tokens
            return max(
                self.start,
                block - history_blocks,
                (
                    block + 1 - self.reference_window_chunks
                    if self.reference_window_chunks
                    else 0
                ),
                0,
            )


@dataclass(kw_only=True)
class DCPState:
    settings: LiveSessionInput
    features: dict[str, object]
    session: NativeBudgetMediaSession | None = None


class DCPRealizationSessionHooks(SessionHooks):
    def __init__(
        self,
        args: Namespace,
        *,
        device: torch.device,
        use_sglang_kernels: bool = True,
        native_backbone: bool = False,
        rendezvous: str = "env://",
        sp_mode: Literal["ulysses", "ring"] = "ulysses",
        tp_size: int = 1,
        kv_window_tokens: int | None = None,
        reference_window_chunks: int = 0,
        native_options: NativeDCPOptions | None = None,
    ) -> None:
        # This inferencer runs Realization only. Its Intent identity must describe
        # the checkpoint, independently of the separately loaded live planner.
        self.kv_window_tokens = kv_window_tokens
        self.reference_window_chunks = reference_window_chunks
        self.args = copy(args)
        manifest = json.loads((Path(args.checkpoint) / "manifest.json").read_text())
        self.args.origin_intent = manifest["identity"]["qwen"].get("mode") == "origin"
        self.device = device
        self.engine = RealizationInferencer(self.args, device=device)
        if native_backbone:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                maybe_init_distributed_environment_and_model_parallel,
            )
            from sglang_omni.models.duo.native_dcp import native_dcp_from_reference

            maybe_init_distributed_environment_and_model_parallel(
                tp_size=tp_size,
                sp_size=args.context_parallel_degree // tp_size,
                ulysses_degree=(
                    args.context_parallel_degree // tp_size
                    if sp_mode == "ulysses"
                    else 1
                ),
                ring_degree=(
                    args.context_parallel_degree // tp_size if sp_mode == "ring" else 1
                ),
                distributed_init_method=rendezvous,
            )
            if (
                native_options is not None
                and native_options.modelopt_checkpoint is not None
            ):
                self.engine.model = native_dcp_from_modelopt(
                    native_options.modelopt_checkpoint,
                    device=device,
                    direct_kernels=native_options.fp8_direct_kernels,
                )
            elif tp_size > 1:
                # Restore on CPU before sharding so startup does not require a full
                # Realization replica on each GPU.
                self.engine.model, _, _ = load_model(
                    self.args,
                    SimpleNamespace(cp=args.context_parallel_degree),
                    torch.device("cpu"),
                )
                self.engine.model = native_dcp_from_reference(
                    self.engine.model, device=device
                )
            else:
                self.engine.ensure_realization()
                self.engine.model = native_dcp_from_reference(
                    self.engine.model, device=device
                )
            if native_options is not None:
                self.engine.model.configure_execution(native_options)
                if native_options.merge_fixed_lora or native_options.fp8_block_linears:
                    self.engine.model.prepare_fixed_weights(native_options)
                else:
                    pass
                if native_options.fp8_attention_calibration is not None:
                    self.engine.model.configure_fp8_attention(
                        native_options.fp8_attention_calibration.format(
                            rank=torch.distributed.get_rank()
                        )
                    )
                else:
                    pass
            else:
                pass
            self.engine.residency.register("realization", self.engine.model)
            self.engine.residency.move("realization", device, "native")
            torch.cuda.empty_cache()
        else:
            self.engine.ensure_realization()
            if use_sglang_kernels:
                use_realization_kernels(self.engine.model)
            else:
                pass
        self.states: dict[SessionIdentity, DCPState] = {}

    @property
    def graph_counts(self) -> tuple[int, int]:
        if isinstance(self.engine.model, NativeDCPRealization):
            return self.engine.model.graph_captures, self.engine.model.graph_replays
        else:
            return 0, 0

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        settings = msgspec.convert(request.inputs, type=LiveSessionInput, strict=True)
        if (settings.height, settings.width) != (self.args.height, self.args.width):
            raise ValueError(
                "DCP session dimensions must match the configured Realization geometry"
            )
        else:
            pass
        self.states[session_identity] = DCPState(
            settings=settings,
            features={
                "sample_id": settings.sample_id,
                "codec": {"mode": "duo_5_17"},
                "protocol": intent_conditioning.PROTOCOL,
                "blocks": [],
                "text_groups": [],
            },
        )

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if context.cancelled.is_set():
            raise RuntimeError("Realization session cancelled")
        else:
            pass
        state = self.states[context.session_identity]
        features = state.features
        plan: RealizationSegment = payload.data["plan"]
        shared: RealizationSegment = payload.data["shared"]
        video: RealizationSegment = payload.data["video"]
        reference: RealizationSegment = payload.data["reference"]
        height, width = state.settings.height // 16, state.settings.width // 16
        if video.tokens.shape[0] != 5 * height * width // 4:
            raise ValueError("This checkpoint requires duo_5_17 blocks")
        else:
            pass
        block = chunk.seq
        if state.session is None:
            features["reference"] = (
                reference.tokens.reshape(1, height // 2, width // 2, 24, 2, 2)
                .permute(3, 0, 1, 4, 2, 5)
                .reshape(24, 1, height, width)
                .cpu()
            )
            features["h50"] = shared.tokens.cpu()
            description_tokens = payload.data["description_tokens"]
            features["text_groups"] = [
                dict(
                    start=0,
                    end=description_tokens,
                    role="system",
                    block=-1,
                    phase=-1,
                    section="assistant_description",
                ),
                dict(
                    start=description_tokens,
                    end=shared.tokens.shape[0],
                    role="system",
                    block=-1,
                    phase=-1,
                    section="speech_cue",
                ),
            ]
            args = copy(self.args)
            args.seed = state.settings.seed
            if self.kv_window_tokens is None:
                state.session = NativeBudgetMediaSession(
                    self.engine.model,
                    features,
                    args,
                    self.device,
                    reference_window_chunks=self.reference_window_chunks,
                )
            else:
                state.session = TokenWindowMediaSession(
                    self.engine.model,
                    features,
                    args,
                    self.device,
                    self.kv_window_tokens,
                    reference_window_chunks=self.reference_window_chunks,
                )
            state.session.shared = shared_condition(features)
        else:
            pass
        start = features["h50"].shape[0]
        features["h50"] = torch.cat((features["h50"], plan.tokens.cpu()))
        features["text_groups"].append(
            dict(
                start=start,
                end=start + plan.tokens.shape[0],
                role="assistant",
                block=block,
                phase=0,
                section="assistant_plan",
            )
        )
        features["blocks"].append(
            dict(
                block_id=block,
                start_ms=chunk.t_start_ms,
                end_ms=chunk.t_start_ms + chunk.duration_ms,
                frame_start=17 * block,
                frame_stop=17 * (block + 1),
                audio_start_sample=round(block * 17 / 24 * 32000),
                audio_stop_sample=round((block + 1) * 17 / 24 * 32000),
            )
        )
        state.session.index = budget_index(
            features, self.args.height, self.args.width, "duo_5_17"
        )
        state.session.pending_kv_reset = payload.data.get("reset_realization_kv", False)
        latent, metrics, _ = state.session.generate(features, block)
        logger.info(
            "Realization DCP block: %s",
            dict(session=str(context.session_identity), block=block, **metrics),
        )
        video_latent = latent["video"].reshape(24, 5, height // 2, 2, width // 2, 2)
        payload.data = {
            "video_latent": video_latent.permute(1, 2, 4, 0, 3, 5).reshape(-1, 96),
            "audio_latent": latent["audio"].permute(0, 2, 1).reshape(-1, 32),
            "metrics": metrics,
        }
        return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)
        if not self.states and isinstance(self.engine.model, NativeDCPRealization):
            self.engine.model.clear_execution_cache()
        else:
            pass

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        state = self.states.get(session_identity)
        if state is None or state.session is None or state.session.kv is None:
            return ResourceUsage()
        else:
            return ResourceUsage(
                kv_tokens=state.session.kv.length, bytes=state.session.kv.nbytes
            )
