"""Causal media KV lifetime and reproducible clean-block replay."""

from dataclasses import replace
from contextlib import contextmanager
import time
import torch
from duo.data.sequence import prepare_sequence, block_seed
from duo.data.sequence_materializer import materialize_media
from duo.data.inference_options import MEDIA_CACHE_POLICY as POLICY, window_start
from ..modules.codec_spec import spec


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def timed(metrics, name, device):
    synchronize(device)
    start = time.perf_counter()
    yield
    synchronize(device)
    elapsed = time.perf_counter() - start
    if torch.distributed.is_initialized():
        value = torch.tensor(elapsed, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        elapsed = value.item()
    metrics[name] = elapsed


def condition(features, block):
    """Only selected DiT intent_states, never a cumulative copy of the Intent prefix."""
    pieces, groups, cursor = [], [], 0
    for g in features["text_groups"]:
        if g["block"] == block:
            piece = features["intent_states"][g["start"] : g["end"]].detach().cpu().clone()
            pieces.append(piece)
            groups.append(dict(g, start=cursor, end=cursor + len(piece)))
            cursor += len(piece)
    if not pieces:
        raise ValueError("Missing commit conditioning")
    return dict(
        intent_states=torch.cat(pieces),
        text_groups=groups,
        protocol=features["protocol"],
        block=block,
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
        self.block_rows = []
        self.start_block = 0
        self.training_rows = None

    def position(self, block):
        row = self.observations["blocks"][block]
        return dict(
            start_ms=row["start_ms"],
            end_ms=row["end_ms"],
            video_spans=self.geometry.spans(round(row["start_ms"] * 24 / 1000)),
            spatial_shape=list(self.observations["reference"].shape[-2:]),
            codec=self.geometry.identity(),
        )

    def sequence(self, part):
        k = part["block"]
        row = self.observations["blocks"][k]
        if "position" in part and part["position"] != self.position(k):
            raise ValueError("Commit position metadata differs from the saved clock")
        shared = self.shared
        groups = [dict(g) for g in shared["text_groups"]]
        offset = len(shared["intent_states"])
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
            intent_states=torch.cat((shared["intent_states"], part["intent_states"])),
            text_groups=groups,
            protocol=part["protocol"],
            clean_blocks=0,
            blocks=[dict(row, source_block_id=k, assistant=media)],
        )
        prepared = prepare_sequence(
            [(item, True)],
            seed=self.args.seed,
            max_seq_len=0,
            target_blocks=[{0}],
            rope=getattr(self.args, "rope", "old"),
        )
        seq = materialize_media(prepared, self.device)[0]["sequence"]

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
            intent_states=torch.cat((self.shared["intent_states"], part["intent_states"])),
            conditioning_blocks=[block],
            media_cache_policy=POLICY,
            media_cache_condition=part,
            media_cache_shared=self.shared,
            media_window=dict(start_block=self.start_block, stop_block=block + 1),
        )

    def initialize(self, part, metrics):
        if self.kv is not None:
            return
        full = self.sequence(part)
        prefix = full.select(lambda g: g["kind"] in ("reference", "shared"))
        # Prefix refiner must contain only shared rows, just as all later calls
        # contain shared + the current body. Do not compute future text.
        n = len(self.shared["intent_states"])
        prefix = replace(
            prefix,
            refiner_text=prefix.refiner_text[:, :n],
            refiner_groups=tuple(g for g in prefix.refiner_groups if g["kind"] == "shared"),
        )
        with timed(metrics, "cache_prefill_seconds", self.device):
            _, self.kv = self.model(
                prefix, torch.full((2,), self.clean_time, device=self.device), commit=True
            )

    def current(self, part, metrics):
        if self.failed or part["block"] != self.next_block:
            raise ValueError("Invalid/noncontiguous media cache session")
        k = part["block"]
        budget = getattr(self.args, "max_seq_len", 0)
        if budget:
            # Use exactly the training packing cost: reference/shared once,
            # then each complete block's intent_states plus BOTH clean and noisy media.
            spatial = (
                self.observations["reference"].shape[-2]
                * self.observations["reference"].shape[-1]
                // 4
            )
            self.block_rows.append(
                len(part["intent_states"])
                + 2 * (spatial * self.geometry.latents + 2 * self.geometry.audio_latents)
            )
            rows = spatial + len(self.shared["intent_states"])
            start = k + 1
            for previous in range(k, -1, -1):
                if rows + self.block_rows[previous] > budget // 128 * 128:
                    break
                rows += self.block_rows[previous]
                start = previous
            if start > k:
                raise ValueError(
                    "Training token budget cannot fit reference, shared text and one complete block"
                )
            self.start_block, self.training_rows = start, rows
        else:
            self.start_block = window_start(k, self.args.sliding_window_size, self.geometry.name)
        self.initialize(part, metrics)
        with timed(metrics, "cache_evict_seconds", self.device):
            self.kv = self.kv.evict(self.start_block)
        sequence = self.sequence(part).select(lambda g: g["kind"] in ("text", "noisy"))
        return sequence

    def commit(self, sequence, values, metrics):
        clean = sequence.with_media(*values).select(lambda g: True, rename={"noisy": "clean"})
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
                        seed=block_seed(self.args.seed, self.observations["sample_id"], block),
                        history=self.kv,
                        clean_time=self.clean_time,
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
            media_cache_policy=POLICY,
            cache_rows=self.kv.length,
            cache_bytes=self.kv.nbytes,
            current_rows=current.length,
            intent_state_rows=current.refiner_text.shape[1],
            cache_committed_block=block,
            generated_history_blocks=block,
            training_window_rows=self.training_rows,
            window_token_budget=getattr(self.args, "max_seq_len", 0),
            media_window=dict(
                start_block=self.start_block,
                stop_block=block + 1,
                frames=(block + 1 - self.start_block) * self.geometry.frames,
            ),
        )
        return dict(video=video.cpu().contiguous(), audio=audio.cpu().contiguous()), metrics, part

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
