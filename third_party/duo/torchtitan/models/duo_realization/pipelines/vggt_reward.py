"""Frozen, video-only DMD quality weights; no generator or score-model logic.

VGGT and native VAE sources remain external. Disabled training never constructs
this service. Pose comparisons use rotations only, in degrees and real seconds.
"""

import hashlib
import importlib
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
DEFAULT_MODEL = ""
DEFAULT_SOURCE = ""


def policy(strength):
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("vggt-loss-weight must be finite and in [0,1]")
    return dict(
        version="vggt_rotation_duration_path_v2",
        strength=strength,
        video_only=True,
        frozen=True,
        frame_stride=6,
        fps=24,
        group_frames=16,
        anchor="first_valid_generated_frame",
        padding="exclude_frames_at_or_after_original_frame_end",
        angle_deadzone_deg=1.0,
        angle_scale_deg=5.0,
        duration_scale_s=5.0,
        path_gate_scale_deg=25.0,
        path_scale_deg=180.0,
        duration_integration="trapezoidal_squared_excess",
        rgb_preprocess="official_pad_518_bicubic",
        quality_min=0.1,
        quality_max=2.0,
    )


def frame_groups(count):
    """First anchor + boundary overlap + new frames; each adjacent edge once."""
    if count < 2:
        raise ValueError("VGGT needs at least two valid frames")
    groups = [tuple(range(min(count, 16)))]
    cursor = len(groups[0])
    while cursor < count:
        stop = min(count, cursor + 14)
        groups.append((0, cursor - 1, *range(cursor, stop)))
        cursor = stop
    return groups


def _angle(left, right):
    import torch

    relative = left.transpose(-1, -2) @ right
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        -1,
    )
    sine = skew.norm(dim=-1) * 0.5
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return torch.atan2(sine, cosine) * (180 / math.pi)


def rotation_metrics(chunks, timestamps, strength):
    """Each chunk is (global frame indices, camera-to-world rotation matrices).

    Relative angles are always evaluated within a single VGGT forward, so
    independent world gauges cannot create artificial inter-group rotation.
    """
    import torch

    settings = policy(strength)
    times = torch.as_tensor(timestamps, dtype=torch.float64, device="cpu")
    if times.ndim != 1 or len(times) < 2 or not torch.isfinite(times).all():
        raise ValueError("Invalid VGGT timestamps")
    dt = times[1:] - times[:-1]
    if not (dt > 0).all():
        raise ValueError("VGGT timestamps must increase")
    theta = torch.full_like(times, float("nan"))
    delta = torch.full_like(dt, float("nan"))
    for indices, matrices in chunks:
        r = torch.as_tensor(matrices).detach().to(device="cpu", dtype=torch.float64)
        if not indices or indices[0] != 0 or r.shape != (len(indices), 3, 3):
            raise ValueError("Invalid VGGT rotation group")
        if not torch.isfinite(r).all():
            raise FloatingPointError("Nonfinite VGGT rotation")
        eye = torch.eye(3, dtype=r.dtype)
        if not torch.allclose(
            r.transpose(-1, -2) @ r, eye.expand_as(r), atol=2e-3, rtol=0
        ) or not torch.allclose(
            torch.linalg.det(r), torch.ones(len(r), dtype=r.dtype), atol=2e-3, rtol=0
        ):
            raise ValueError("VGGT output is not a proper rotation")
        angles = _angle(r[:1], r)
        steps = _angle(r[:-1], r[1:])
        for j, index in enumerate(indices):
            if not 0 <= index < len(times):
                raise ValueError("VGGT frame index outside trajectory")
            if torch.isnan(theta[index]):
                theta[index] = angles[j]
            if j and index == indices[j - 1] + 1:
                if not torch.isnan(delta[index - 1]):
                    raise ValueError("Duplicate VGGT trajectory edge")
                delta[index - 1] = steps[j - 1]
    if not torch.isfinite(theta).all() or not torch.isfinite(delta).all():
        raise ValueError("VGGT groups did not cover the complete trajectory")
    deadzone = settings["angle_deadzone_deg"]
    excursion = ((theta - deadzone).clamp_min(0) / settings["angle_scale_deg"]).square()
    duration = float(((excursion[:-1] + excursion[1:]) * 0.5 * dt).sum() / 5)
    gate = ((theta - deadzone) / 25).clamp(0, 1)
    effective = float((delta * (gate[:-1] + gate[1:]) * 0.5).sum())
    path = (effective / 180) ** 2
    exponent = duration + path
    quality = min(2.0, max(0.1, 2 * math.exp(-exponent)))
    return dict(
        weight=1 + strength * (quality - 1),
        rotation_index=exponent,
        duration_index=duration,
        path_index=path,
        cumulative_rotation_deg=float(delta.sum()),
        effective_rotation_deg=effective,
        max_rotation_deg=float(theta.max()),
        max_adjacent_rotation_deg=float(delta.max()),
        duration_s=float(times[-1] - times[0]),
        frames=len(times),
        max_sample_interval_s=float(dt.max()),
    )


def preprocess_rgb(frames):
    """In-memory equivalent of official load_and_preprocess_images(mode='pad')."""
    import numpy as np
    import torch
    from PIL import Image

    images = []
    for frame in frames:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        width, height = image.size
        scale = 518 / max(width, height)
        w, h = round(width * scale / 14) * 14, round(height * scale / 14) * 14
        if min(w, h) < 14:
            raise ValueError("Unsupported VGGT input aspect ratio")
        image = image.resize((w, h), Image.Resampling.BICUBIC)
        tensor = (
            torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255
        )
        dw, dh = 518 - w, 518 - h
        images.append(
            torch.nn.functional.pad(
                tensor, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2), value=1.0
            )
        )
    return torch.stack(images)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_resources(model_path, source_path):
    """Verify the staged resources, without importing VGGT or loading tensors."""
    model, source = Path(model_path), Path(source_path)
    manifest = json.loads((model.parent / "weights_manifest.json").read_text())
    entry = manifest["files"][model.name]
    actual = _sha(model)
    if model.stat().st_size != entry["size"] or actual != entry["sha256"]:
        raise ValueError("VGGT weight content differs from the verified manifest")
    upstream = json.loads((source / "source_manifest.json").read_text())
    if upstream["revision"] != REVISION:
        raise ValueError("VGGT source revision differs from the pinned revision")
    for relative, expected in upstream["files"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or _sha(path) != expected:
            raise ValueError("VGGT source content differs: " + relative)
    actual_files = {str(p.relative_to(source)) for p in (source / "vggt").rglob("*.py")}
    expected_files = {
        p for p in upstream["files"] if p.startswith("vggt/") and p.endswith(".py")
    }
    if actual_files != expected_files:
        raise ValueError("VGGT runtime source file set differs")
    code_sha = hashlib.sha256(
        json.dumps(upstream["files"], sort_keys=True).encode()
    ).hexdigest()
    return dict(weights_sha256=actual, source_revision=REVISION, source_sha256=code_sha)


@contextmanager
def preserve_rng(device):
    import numpy as np
    import torch

    py, array = random.getstate(), np.random.get_state()
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(array)


def video_latent(sequence, group):
    """Inverse the existing T,H/2,W/2,C,2,2 patch order, without a copy of history."""
    import torch

    t, h, w = group["video_shape"]
    inverse = torch.full(
        (sequence.length,), -1, device=sequence.device, dtype=torch.long
    )
    inverse[sequence.video_indices] = torch.arange(
        len(sequence.video_indices), device=sequence.device
    )
    rows = inverse[group["start"] : group["end"]]
    if (rows < 0).any() or len(rows) != t * h * w // 4:
        raise ValueError("VGGT video metadata and packed rows differ")
    return (
        sequence.video[0, rows]
        .detach()
        .reshape(t, h // 2, w // 2, 24, 2, 2)
        .permute(3, 0, 1, 4, 2, 5)
        .reshape(24, t, h, w)
    )


def decode_frames(sequence, sample, decoder):
    """Decode only this window, including native delayed EOF; retain 4fps RGB."""
    groups = [
        g
        for g in sequence.groups
        if g["sample"] == sample and g.get("modality") == "video"
    ]
    references = [g for g in groups if g["kind"] == "reference"]
    blocks = sorted(
        (g for g in groups if g["kind"] == "noisy"), key=lambda g: g["block"]
    )
    if len(references) != 1 or not blocks:
        raise ValueError("VGGT example needs its own reference and generated blocks")
    decoder.start_decode(video_latent(sequence, references[0]))
    frames, timestamps, seen, last = [], [], 0, None

    def consume(packets):
        nonlocal seen, last
        for packet in packets:
            group = blocks[packet["block"]]
            for index, rgb in enumerate(packet["frames"]):
                stamp = (group["start_ms"] + index * 1000 / 24) / 1000
                if stamp >= group.get("valid_end_ms", group["end_ms"]) / 1000 - 1e-7:
                    continue
                if seen % 6 == 0:
                    frames.append(rgb.copy())
                    timestamps.append(stamp)
                last = (seen, rgb.copy(), stamp)
                seen += 1

    for group in blocks:
        consume(decoder.decode_block(video_latent(sequence, group)))
    consume(decoder.finish_decode())
    if last is not None and (last[0] % 6):
        frames.append(last[1])
        timestamps.append(last[2])
    if len(frames) < 2:
        raise ValueError("VGGT example has fewer than two valid decoded frames")
    return frames, timestamps


class VGGTVideoReward:
    """One frozen evaluator per CP group; no FSDP collectives or optimizer state."""

    def __init__(self, config, device, cp_group=None):
        import torch
        import torch.distributed as dist

        self.config, self.device, self.cp_group = config, torch.device(device), cp_group
        self.strength = config.vggt_loss_weight
        if self.strength <= 0:
            raise ValueError("Do not construct VGGT when the feature is disabled")
        self.leader = cp_group is None or dist.get_rank(cp_group) == 0
        self.identity = self._call(self._initialize)

    def _call(self, function):
        import torch.distributed as dist

        payload = [None]
        if self.leader:
            try:
                payload[0] = dict(value=function())
            except Exception as exc:
                payload[0] = dict(error=f"{type(exc).__name__}: {exc}")
        if self.cp_group is not None:
            dist.broadcast_object_list(
                payload, src=dist.get_global_rank(self.cp_group, 0), group=self.cp_group
            )
        if "error" in payload[0]:
            raise RuntimeError("VGGT video weighting failed: " + payload[0]["error"])
        return payload[0]["value"]

    def _initialize(self):
        import torch
        from accelerate import init_empty_weights

        from ..modules.video_vae import VideoVAE

        c = self.config
        identity = inspect_resources(c.vggt_path, c.vggt_source)
        source = str(Path(c.vggt_source).resolve())
        with preserve_rng(self.device):
            sys.path.insert(0, source)
            try:
                upstream = importlib.import_module("vggt.models.vggt")
                pose = importlib.import_module("vggt.utils.pose_enc")
            finally:
                sys.path.remove(source)
            if not Path(upstream.__file__).resolve().is_relative_to(Path(source)):
                raise ValueError("A different VGGT package is already imported")
            with init_empty_weights():
                self.model = upstream.VGGT()
            if Path(c.vggt_path).suffix == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(c.vggt_path, device="cpu")
            else:
                state = torch.load(
                    c.vggt_path, map_location="cpu", weights_only=True, mmap=True
                )
            self.model.load_state_dict(state, strict=True, assign=True)
            del state
            self.model.point_head = self.model.depth_head = self.model.track_head = None
            self.model.eval().requires_grad_(False).to(self.device)
            self.pose_decoder = pose.pose_encoding_to_extri_intri
            base = Path(c.pretrained)
            if base.name != "transformer" or base.parent.name not in (
                "FL2VA",
                "Ref2VA",
            ):
                raise ValueError(
                    "VGGT decoding requires the standard Realization family/transformer source layout"
                )
            root = base.parent.parent
            if c.encode not in ("old", "duo_2_5", "duo_5_17"):
                raise ValueError(
                    "VGGT weighting currently supports the native Realization video codecs"
                )
            args = SimpleNamespace(
                realization=str(root),
                decode=c.encode,
                decode_backend="standard",
                video_decode_latent=3,
            )
            self.decoder = VideoVAE(
                args,
                vae_dtype=torch.float32,
                video_autocast_dtype=torch.bfloat16,
                feature_dtype=torch.float32,
            ).to(self.device)
            # Materialize before restoring RNG, rather than during the first rollout.
            self.decoder._native()
            identity["video_vae_sha256"] = _sha(
                root / "FL2VA/video_vae/source/model.safetensors"
            )
            identity["video_vae_config_sha256"] = _sha(
                Path(__file__).resolve().parents[1]
                / "modules/native_vae/video_vae.yaml"
            )
            from importlib.metadata import version

            identity["runtime"] = {
                name: version(name) for name in ("Pillow", "accelerate")
            }
        identity["policy"] = policy(self.strength)
        identity["codec"] = self.decoder.spec.identity()
        return identity

    def score(self, sequence):
        def evaluate():
            import torch

            weights, records = {}, {}
            with preserve_rng(self.device), torch.no_grad():
                for sample, info in enumerate(sequence.samples):
                    if info.get("padding", False):
                        continue
                    start = time.monotonic()
                    frames, stamps = decode_frames(sequence, sample, self.decoder)
                    decoded = time.monotonic()
                    chunks = []
                    for indices in frame_groups(len(frames)):
                        images = preprocess_rgb([frames[i] for i in indices]).to(
                            self.device
                        )
                        with torch.autocast(
                            self.device.type,
                            dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda",
                        ):
                            result = self.model(images)
                        extrinsics, _ = self.pose_decoder(
                            result["pose_enc"].float(), build_intrinsics=False
                        )
                        rotations = extrinsics[0, :, :3, :3].transpose(-1, -2).cpu()
                        chunks.append((indices, rotations))
                        del images, result, extrinsics
                    metrics = rotation_metrics(chunks, stamps, self.strength)
                    metrics.update(
                        decode_s=decoded - start,
                        score_s=time.monotonic() - decoded,
                        groups=len(chunks),
                        sample_id=info["sample_id"],
                    )
                    records[sample] = metrics
                    weights[sample] = metrics["weight"]
            return weights, records

        return self._call(evaluate)
