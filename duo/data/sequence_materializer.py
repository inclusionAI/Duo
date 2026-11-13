"""Finalize prepared joint-media sequences on the inference device.

Only device finalization differs: retain clean targets and explicit metadata so
noise, teacher forcing, bidirectional scoring and generated history are chosen
by the training stage. No encoder or model is imported in a data worker.
"""

from dataclasses import dataclass, replace
import math
import torch


@dataclass(frozen=True)
class MediaSequence:
    video: torch.Tensor
    audio: torch.Tensor
    text: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor
    positions: torch.Tensor
    tags: torch.Tensor
    roles: torch.Tensor
    time_ids: torch.Tensor
    video_weights: torch.Tensor
    audio_weights: torch.Tensor
    groups: tuple
    text_groups: tuple
    samples: tuple
    seeds: tuple
    refiner_text: torch.Tensor
    refiner_groups: tuple
    text_selection: torch.Tensor

    @property
    def device(self):
        return self.video.device

    @property
    def length(self):
        return len(self.tags)

    @property
    def blocks(self):
        return max((g["block"] + 1 for g in self.groups if g["kind"] == "noisy"), default=0)

    def with_media(self, video, audio):
        if video.shape != self.video.shape or audio.shape != self.audio.shape:
            raise ValueError("Media row shape differs")
        return replace(self, video=video, audio=audio)

    def drop_condition(self, sample_ids):
        """Drop intent_states only. Reference and legal media history stay unchanged."""
        text = self.refiner_text.clone()
        for g in self.refiner_groups:
            if g["sample"] in sample_ids:
                text[:, g["start"] : g["end"]] = 0
        return replace(self, refiner_text=text, text=text[:, self.text_selection])

    def select(self, predicate, *, rename=None, multiple=128):
        """Select complete semantic groups, preserving patch/channel ordering."""
        selected = [g for g in self.groups if g["kind"] != "padding" and predicate(g)]
        positions, groups, cursor = [], [], 0
        for g in selected:
            ids = torch.arange(g["start"], g["end"], device=self.device)
            positions.append(ids)
            groups.append(
                dict(
                    g,
                    start=cursor,
                    end=cursor + len(ids),
                    kind=(rename or {}).get(g["kind"], g["kind"]),
                )
            )
            cursor += len(ids)
        ids = (
            torch.cat(positions)
            if positions
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        length = max(multiple, math.ceil(cursor / multiple) * multiple)
        remap = torch.full((self.length,), -1, device=self.device, dtype=torch.long)
        remap[ids] = torch.arange(cursor, device=self.device)
        outputs = {}
        for name in ("video", "audio", "text"):
            old = getattr(self, name + "_indices")
            keep = remap[old] >= 0
            outputs[name] = getattr(self, name)[:, keep]
            outputs[name + "_indices"] = remap[old[keep]]
            if name == "text":
                outputs["text_selection"] = self.text_selection[keep]
            if name != "text":
                outputs[name + "_weights"] = getattr(self, name + "_weights")[keep]
        pos = torch.zeros(length, 3, device=self.device, dtype=self.positions.dtype)
        tags = torch.zeros(length, device=self.device, dtype=torch.long)
        roles = torch.full_like(tags, -1)
        times = torch.zeros_like(tags)
        pos[:cursor], tags[:cursor], roles[:cursor], times[:cursor] = (
            self.positions[ids],
            self.tags[ids],
            self.roles[ids],
            self.time_ids[ids],
        )
        if length > cursor:
            groups.append(dict(sample=-1, block=-1, kind="padding", start=cursor, end=length))
        tg = []
        textcursor = 0
        for g in groups:
            if g["kind"] in ("shared", "text"):
                n = g["end"] - g["start"]
                tg.append(dict(g, start=textcursor, end=textcursor + n))
                textcursor += n
        return replace(
            self,
            **outputs,
            positions=pos,
            tags=tags,
            roles=roles,
            time_ids=times,
            groups=tuple(groups),
            text_groups=tuple(tg),
        )

    def bidirectional(self):
        # Physically remove clean answer copies, not merely their loss weights.
        return self.select(lambda g: g["kind"] != "clean")

    def row_mask(self, modality, *, kind="noisy", block=None):
        indices = getattr(self, modality + "_indices")
        mask = torch.zeros(self.length, device=self.device, dtype=torch.bool)
        for g in self.groups:
            if g["kind"] == kind and (block is None or g["block"] == block):
                mask[g["start"] : g["end"]] = True
        return mask[indices]

    def update_history(self, video, audio):
        """Replace ALL clean target copies using generated noisy-row values."""
        outputs = []
        for name, generated in (("video", video), ("audio", audio)):
            indices = getattr(self, name + "_indices")
            inverse = torch.full((self.length,), -1, device=self.device, dtype=torch.long)
            inverse[indices] = torch.arange(len(indices), device=self.device)
            value = getattr(self, name).clone()
            sources = {
                (g["sample"], g["block"]): g
                for g in self.groups
                if g["kind"] == "noisy" and g.get("modality") == name
            }
            for g in self.groups:
                if g["kind"] == "clean" and g.get("modality") == name:
                    src = sources[(g["sample"], g["block"])]
                    dstids = inverse[g["start"] : g["end"]]
                    srcids = inverse[src["start"] : src["end"]]
                    value[:, dstids] = generated[:, srcids].detach()
            outputs.append(value)
        return self.with_media(*outputs)


def materialize_media(prepared, device):
    x = prepared["inputs"]
    device = torch.device(device)

    def to(t):
        return t.to(device, non_blocking=t.is_pinned())

    vi = to(x["img_pos_info"]["position_ids"])
    ai = to(x["audio_pos_info"]["position_ids"])
    ti = to(x["text_pos_info"]["position_ids"])
    sequence = MediaSequence(
        video=to(x["x"])[:, vi].clone(),
        audio=to(x["audio_x"])[:, ai].clone(),
        text=to(x["prompt_embeds"])[None],
        video_indices=vi,
        audio_indices=ai,
        text_indices=ti,
        positions=to(x["img_position_ids"])[0],
        tags=to(x["token_tags"]).clamp_min(0),
        roles=to(x["role_ids"]),
        time_ids=to(x["inverse_indices"]),
        video_weights=to(prepared["weights"][0]),
        audio_weights=to(prepared["weights"][1]),
        groups=tuple(dict(g) for g in prepared["layout"]["groups"]),
        text_groups=tuple(dict(g) for g in prepared["layout"]["refiner_groups"]),
        samples=tuple(dict(g) for g in prepared["layout"]["samples"]),
        seeds=tuple(n["seed"] for n in prepared["noise"]),
        refiner_text=to(x["prompt_embeds"])[None],
        refiner_groups=tuple(dict(g) for g in prepared["layout"]["refiner_groups"]),
        text_selection=torch.arange(len(ti), device=device),
    )
    return {"sequence": sequence}, (), prepared["layout"]


def build_media_dataloader(dataset, **kwargs):
    from .dataloader import build_dataloader

    return build_dataloader(dataset, materialize_fn=materialize_media, **kwargs)


def transfer_media(source, values, target, *, source_kind="noisy", target_kind="noisy"):
    """Match rows by conversation/block/modality, never by packed offsets."""
    outputs = []
    for name, source_value in zip(("video", "audio"), values):
        si = getattr(source, name + "_indices")
        ti = getattr(target, name + "_indices")
        sm = torch.full((source.length,), -1, device=source.device, dtype=torch.long)
        sm[si] = torch.arange(len(si), device=source.device)
        tm = torch.full((target.length,), -1, device=target.device, dtype=torch.long)
        tm[ti] = torch.arange(len(ti), device=target.device)
        groups = {
            (g["sample"], g["block"]): g
            for g in source.groups
            if g["kind"] == source_kind and g.get("modality") == name
        }
        value = getattr(target, name).clone()
        for dst in target.groups:
            if dst["kind"] != target_kind or dst.get("modality") != name:
                continue
            src = groups.get((dst["sample"], dst["block"]))
            if src is None:
                continue
            value[:, tm[dst["start"] : dst["end"]]] = source_value[:, sm[src["start"] : src["end"]]]
        outputs.append(value)
    return tuple(outputs)
