"""Causal media KV lifetime and reproducible clean-block replay."""

import time
from contextlib import contextmanager
from dataclasses import replace

import torch
from data.dataloader_realization import materialize_realization
from data.dataloader_realization_sequence import block_seed, prepare_sequence
from data.inference_options import (
    MEDIA_CACHE_POLICY,
    ROLLING_MEDIA_CACHE_POLICY,
    window_start,
)

from ..modules.codec_spec import spec
from ..modules.rope import make_position_builder


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(metrics, name, device, *, collective=True):
    synchronize(device)
    start = time.perf_counter()
    yield
    synchronize(device)
    elapsed = time.perf_counter() - start
    if collective and torch.distributed.is_initialized():
        value = torch.tensor(elapsed, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        elapsed = value.item()
    metrics[name] = elapsed


def condition(features, block):
    """Only selected Realization h50, never a cumulative copy of the Intent prefix."""
    pieces, groups, cursor = [], [], 0
    for g in features["text_groups"]:
        if g["block"] == block:
            piece = features["h50"][g["start"] : g["end"]].detach().cpu().clone()
            pieces.append(piece)
            groups.append(dict(g, start=cursor, end=cursor + len(piece)))
            cursor += len(piece)
    if not pieces:
        raise ValueError("Missing commit conditioning")
    return dict(
        h50=torch.cat(pieces),
        text_groups=groups,
        protocol=features["protocol"],
        block=block,
        **(
            {"conditioning_layout": features["conditioning_layout"]}
            if "conditioning_layout" in features
            else {}
        )
    )


def shared_condition(features):
    return condition(features, -1)


class MediaSession:
    """Owned by one conversation/rank. A failed forward invalidates this object."""

    def __init__(self, model, cache, args, device, *, clean_time=1.0, sampler=None):
        self.model, self.observations, self.args = model, cache, args
        self.device, self.clean_time = torch.device(device), clean_time
        self.sampler = sampler
        self.kv = None
        self.shared = None
        self.next_block = 0
        self.failed = False
        self.geometry = spec(cache.get("codec", {}).get("mode", "old"))

    def position(self, block):
        row = self.observations["blocks"][block]
        return dict(
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            video_spans=self.geometry.spans(round(row["start_ms"] * 24 / 1000)),
            spatial_shape=list(self.observations["reference"].shape[-2:]),
            codec=self.geometry.identity(),
        )

    def window_start(self, block):
        return window_start(block, self.args.sliding_window_size, self.geometry.name)

    def sequence(self, part):
        k = part["block"]
        row = self.observations["blocks"][k]
        if "position" in part and part["position"] != self.position(k):
            raise ValueError("Commit position metadata differs from the saved clock")
        shared = self.shared
        groups = [dict(g) for g in shared["text_groups"]]
        offset = len(shared["h50"])
        groups += [
            dict(g, block=0, start=g["start"] + offset, end=g["end"] + offset)
            for g in part["text_groups"]
        ]
        reference = self.observations["reference"]
        media = dict(
            video=reference.new_zeros(24, self.geometry.latents, *reference.shape[-2:]),
            audio=torch.zeros(2, 32, self.geometry.audio_latents),
            video_weight=torch.ones(self.geometry.latents),
            audio_weight=torch.ones(self.geometry.audio_latents),
            video_spans=self.geometry.spans(round(row["start_ms"] * 24 / 1000)),
        )
        item = dict(
            codec=self.geometry.identity(),
            sample_id=self.observations["sample_id"],
            reference=reference,
            h50=torch.cat((shared["h50"], part["h50"])),
            text_groups=groups,
            protocol=part["protocol"],
            clean_blocks=0,
            blocks=[dict(row, source_block_id=k, assistant=media)],
            **(
                {"conditioning_layout": part["conditioning_layout"]}
                if "conditioning_layout" in part
                else {}
            )
        )
        prepared = prepare_sequence(
            [(item, True)],
            seed=self.args.seed,
            max_seq_len=0,
            target_blocks=[{0}],
            position_builder=make_position_builder(getattr(self.args, "rope", "old")),
        )
        seq = materialize_realization(prepared, self.device)[0]["sequence"]

        def absolute(g):
            return dict(g, block=k, phase=k) if g["block"] >= 0 else g

        return replace(
            seq,
            groups=tuple(map(absolute, seq.groups)),
            text_groups=tuple(map(absolute, seq.text_groups)),
            refiner_groups=tuple(map(absolute, seq.refiner_groups)),
        )

    def conditioning_view(self, features, block):
        part = condition(features, block)
        part["position"] = self.position(block)
        return dict(
            h50=torch.cat((self.shared["h50"], part["h50"])),
            conditioning_blocks=[block],
            media_cache_policy=MEDIA_CACHE_POLICY,
            media_cache_condition=part,
            media_cache_shared=self.shared,
            media_window=dict(
                start_block=self.window_start(block), stop_block=block + 1
            ),
        )

    def initialize(self, part, metrics):
        if self.kv is not None:
            return
        full = self.sequence(part)
        prefix = full.select(lambda g: g["kind"] in ("reference", "shared"))
        # Prefix refiner must contain only shared rows, just as all later calls
        # contain shared + the current body. Do not compute future text.
        n = len(self.shared["h50"])
        prefix = replace(
            prefix,
            refiner_text=prefix.refiner_text[:, :n],
            refiner_groups=tuple(
                g for g in prefix.refiner_groups if g["kind"] == "shared"
            ),
        )
        with timed(metrics, "cache_prefill_seconds", self.device):
            _, self.kv = self.model(
                prefix,
                torch.full((2,), self.clean_time, device=self.device),
                commit=True,
            )

    def current(self, part, metrics):
        if self.failed or part["block"] != self.next_block:
            raise ValueError("Invalid/noncontiguous media cache session")
        self.initialize(part, metrics)
        k = part["block"]
        with timed(metrics, "cache_evict_seconds", self.device):
            self.kv = self.kv.evict(self.window_start(k))
        sequence = self.sequence(part).select(lambda g: g["kind"] in ("text", "noisy"))
        budget = getattr(self.args, "max_seq_len", 0)
        if budget and sequence.length + self.kv.length > budget:
            raise ValueError("Cached inference context exceeds max_seq_len")
        return sequence

    def commit(self, sequence, values, metrics):
        clean = sequence.with_media(*values).select(
            lambda g: True, rename={"noisy": "clean"}
        )
        clean = replace(clean, time_ids=torch.zeros_like(clean.time_ids))
        with timed(metrics, "cache_commit_seconds", self.device):
            _, self.kv = self.model(
                clean,
                torch.full((2,), self.clean_time, device=self.device),
                cache=self.kv,
                commit=True,
            )
        self.next_block += 1

    @torch.no_grad()
    def generate(self, features, block):
        from .inference import sample_sequence

        if self.shared is None:
            self.shared = shared_condition(features)
        part = condition(features, block)
        part["position"] = self.position(block)
        metrics = {}
        try:
            current = self.current(part, metrics)
            with timed(metrics, "denoise_seconds", self.device):
                if self.sampler:
                    values, denoise = self.sampler(current, self.kv, block)
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
                        cm_sampling_mode=getattr(self.args, "cm_sampling_mode", None),
                    )
            self.commit(current, values, metrics)
        except BaseException:
            self.failed = True
            self.kv = None
            raise
        t, h, w = self.geometry.latents, *self.observations["reference"].shape[-2:]
        video = (
            values[0]
            .reshape(t, h // 2, w // 2, 24, 2, 2)
            .permute(3, 0, 1, 4, 2, 5)
            .reshape(24, t, h, w)
        )
        audio = values[1].reshape(2, self.geometry.audio_latents, 32).permute(0, 2, 1)
        metrics.update(
            denoise,
            cache_mode="block",
            media_cache_policy=MEDIA_CACHE_POLICY,
            cache_rows=self.kv.length,
            cache_bytes=self.kv.nbytes,
            current_rows=current.length,
            h50_rows=current.refiner_text.shape[1],
            cache_committed_block=block,
            generated_history_blocks=block,
            media_window=dict(
                start_block=self.window_start(block),
                stop_block=block + 1,
                frames=(block + 1 - self.window_start(block)) * self.geometry.frames,
            ),
        )
        return (
            dict(video=video.cpu().contiguous(), audio=audio.cpu().contiguous()),
            metrics,
            part,
        )

    @torch.no_grad()
    def replay(self, states, shared):
        if not states:
            return {}
        self.shared = shared
        metrics = {}
        try:
            with timed(metrics, "cache_restore_seconds", self.device):
                for state in states:
                    part = state["media_cache_condition"]
                    current = self.current(part, {})
                    v, a = state["latent"]["video"], state["latent"]["audio"]
                    t, h, w = v.shape[1:]
                    video = (
                        v.reshape(24, t, h // 2, 2, w // 2, 2)
                        .permute(1, 2, 4, 0, 3, 5)
                        .reshape(1, -1, 96)
                    )
                    audio = a.permute(0, 2, 1).reshape(1, -1, 32)
                    self.commit(
                        current,
                        (
                            video.to(self.device, dtype=torch.float32),
                            audio.to(self.device, dtype=torch.float32),
                        ),
                        {},
                    )
        except BaseException:
            self.failed = True
            self.kv = None
            raise
        return metrics


class BudgetMediaSession(MediaSession):
    """Training-budget rolling KV; normal sliding only evicts complete blocks.

    Retained KV keeps its original historical context. Only interrupted runs
    replay saved conditions/latents, chronologically from the first block.
    """

    def __init__(self, model, cache, args, device):
        super().__init__(model, cache, args, device)
        self.states = []
        self.start = 0
        self.index = None
        self.reset_interval = getattr(args, "kv_reset", 0)
        anchor_blocks = getattr(args, "first_gen_ref", 0)
        if (anchor_blocks or self.reset_interval) and (
            getattr(args, "realization_inference_mode", "causal") != "causal"
            or getattr(args, "cache_mode", "block") != "block"
        ):
            raise ValueError(
                "Long-video history policies require causal Realization and block cache"
            )
        self.first_reference = None
        if anchor_blocks:
            from .first_gen_ref import FirstGenReference

            self.first_reference = FirstGenReference(anchor_blocks)

    def window_start(self, block):
        from data.inference_budget import causal_start

        # A shorter current body can free space, but evicted blocks stay gone.
        start = max(self.start, causal_start(self.index, block, self.args.max_seq_len))
        if self.reset_interval:
            from .kv_reset import retained_window_start

            start = retained_window_start(block, self.reset_interval, start)
        return start

    def current(self, part, metrics):
        if self.failed or part["block"] != self.next_block:
            raise ValueError("Invalid/noncontiguous media cache session")
        k = part["block"]
        start = self.window_start(k)
        self.initialize(part, metrics)
        previous = self.kv
        if self.first_reference is not None:
            from .first_gen_ref import REGION

            previous = previous.select_groups(lambda g: not g.get(REGION, False))
        evicted = {
            g["block"]
            for g in previous.groups
            if g["kind"] == "clean" and g["block"] < start
        }
        with timed(metrics, "cache_evict_seconds", self.device):
            ordinary = previous.evict(start)
            self.kv = (
                ordinary
                if self.first_reference is None
                else self.first_reference.attach(ordinary)
            )
        self.start = start
        metrics.update(
            cache_evicted_blocks=len(evicted),
            cache_evicted_block_ids=sorted(evicted),
            cache_evicted_rows=previous.length - ordinary.length,
        )
        current = self.sequence(part).select(lambda g: g["kind"] in ("text", "noisy"))
        if self.first_reference is not None or self.reset_interval:
            from .kv_reset import reset_due

            metrics.update(
                kv_reset_applied=reset_due(k, self.reset_interval),
                kv_reset_count=k // self.reset_interval if self.reset_interval else 0,
                kv_reset_kept_block=(
                    k - 1 if reset_due(k, self.reset_interval) else None
                ),
                denoise_cache_rows=self.kv.length,
                realization_input_rows=self.kv.length + current.length,
            )
        return current

    def commit(self, sequence, values, metrics):
        super().commit(sequence, values, metrics)
        if self.first_reference is not None:
            if self.first_reference.capture(self.next_block - 1, self.kv):
                from .first_gen_ref import REGION

                ordinary = self.kv.select_groups(lambda g: not g.get(REGION, False))
                self.kv = self.first_reference.attach(ordinary)

    def generate(self, features, block):
        from data.inference_budget import budget_index, cost

        if self.index is None:
            self.index = budget_index(
                features, self.args.height, self.args.width, self.geometry.name
            )
        latent, metrics, part = super().generate(features, block)
        self.states.append(dict(latent=latent, media_cache_condition=part))
        metrics.update(
            training_equivalent_rows=cost(self.index, self.start, block + 1)[
                "padded_rows"
            ],
            media_cache_policy=ROLLING_MEDIA_CACHE_POLICY,
        )
        if self.first_reference is not None or self.reset_interval:
            from .first_gen_ref import POLICY as anchor_policy
            from .kv_reset import POLICY as reset_policy

            metrics.update(
                first_gen_ref=self.first_reference.limit if self.first_reference else 0,
                first_gen_ref_blocks=(
                    self.first_reference.blocks if self.first_reference else 0
                ),
                first_gen_ref_rows=(
                    self.first_reference.rows if self.first_reference else 0
                ),
                first_gen_ref_policy=anchor_policy if self.first_reference else "off",
                kv_reset=self.reset_interval,
                kv_reset_policy=reset_policy if self.reset_interval else "off",
            )
        return latent, metrics, part

    @torch.no_grad()
    def replay(self, states, shared, *, features):
        from data.inference_budget import budget_index

        self.index = budget_index(
            features, self.args.height, self.args.width, self.geometry.name
        )
        if len(states) > len(self.index["blocks"]):
            raise ValueError("Restored latent prefix exceeds conditioning")
        self.states = []
        if self.first_reference is not None:
            from .first_gen_ref import FirstGenReference

            self.first_reference = FirstGenReference(self.first_reference.limit)
        self.shared = shared
        self.next_block = 0
        self.kv = None
        self.start = 0
        self.failed = False
        metrics = dict(restored_latent_blocks=len(states), cache_restore_seconds=0.0)
        if not states:
            return metrics
        try:
            with timed(metrics, "cache_restore_seconds", self.device):
                for saved in states:
                    part = saved["media_cache_condition"]
                    current = self.current(part, {})
                    v, a = saved["latent"]["video"], saved["latent"]["audio"]
                    t, h, w = v.shape[1:]
                    video = (
                        v.reshape(24, t, h // 2, 2, w // 2, 2)
                        .permute(1, 2, 4, 0, 3, 5)
                        .reshape(1, -1, 96)
                    )
                    audio = a.permute(0, 2, 1).reshape(1, -1, 32)
                    self.commit(
                        current,
                        (
                            video.to(self.device, dtype=torch.float32),
                            audio.to(self.device, dtype=torch.float32),
                        ),
                        {},
                    )
                    self.states.append(
                        dict(latent=saved["latent"], media_cache_condition=part)
                    )
        except BaseException:
            self.failed = True
            self.kv = None
            raise
        return metrics

    def conditioning_view(self, features, block):
        value = super().conditioning_view(features, block)
        value["media_cache_policy"] = ROLLING_MEDIA_CACHE_POLICY
        return value
