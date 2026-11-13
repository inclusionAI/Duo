"""Checkpoint-bound Realization timestep/modality tables, independent of media and masks."""

import hashlib
import json
import uuid
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ..schedulers.flow import grid

FORMAT = "realization-adaln-v1"
PAYLOAD = "adaln_embeddings.safetensors"
ORDERS = ("shift_attn", "scale_attn", "gate_attn", "shift_ff", "scale_ff", "gate_ff")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(checkpoint, family, stage):
    if (
        checkpoint.manifest["stage"] != stage
        or family not in checkpoint.manifest["models"]
    ):
        raise ValueError("AdaLN checkpoint stage/family differs")
    return dict(
        checkpoint=dict(complete_sha256=checkpoint.sha256, stage=stage, family=family),
        model=checkpoint.manifest["models"][family],
    )


def snapshot_blueprint(model):
    model._realization_full_parameter_shapes = {
        n: tuple(v.shape) for n, v in model.state_dict().items()
    }


def select_adaln(model, layers, *, output=True):
    """Keep the original parameter names without allocating a full Realization."""
    snapshot_blueprint(model)
    layers = set(layers)
    blocks = []
    for i, block in enumerate(model.transformer_blocks):
        selected = nn.Module()
        if i in layers:
            selected.adaln_proj = block.adaln_proj
        blocks.append(selected)
    model.transformer_blocks = nn.ModuleList(blocks)
    final = nn.Module()
    if output:
        final.linear = model.norm_out.linear
    model.norm_out = final
    for name in list(model._modules):
        if name not in ("time_proj", "time_embedder", "transformer_blocks", "norm_out"):
            delattr(model, name)
    return tuple(model.state_dict())


def remove_adaln_weights(model):
    """Call before allocation/FSDP; preserve the output RMSNorm and every other weight."""
    snapshot_blueprint(model)
    model.time_proj = None
    model.time_embedder = None
    for block in model.transformer_blocks:
        block.adaln_proj = None
    model.norm_out.linear = None
    return tuple(model.state_dict())


@torch.no_grad()
def extract_tables(model, timesteps, layers, *, output=True):
    from .diffusers_model import get_parameter_dtype

    temb = model.time_embedder(
        model.time_proj(timesteps).to(get_parameter_dtype(model.time_embedder))
    )
    result = {"timesteps": timesteps.cpu().contiguous()}
    for layer in layers:
        rows = model.transformer_blocks[layer].adaln_proj(temb)
        result[f"block.{layer:03d}"] = torch.stack(rows).cpu().contiguous()
    if output:
        projection = model.norm_out.linear
        shift, scale = projection(
            torch.nn.functional.silu(temb).to(get_parameter_dtype(projection))
        ).chunk(2, -1)
        result["output_modulations"] = torch.stack((shift, scale)).cpu().contiguous()
        result["time_embedding"] = temb.cpu().contiguous()
    return result


@torch.no_grad()
def extract_sampling_tables(model, timesteps, layers, *, output=True):
    """Use the sampler's two-row [clean, noisy] GEMM shape at every noise level.

    A single large timestep batch changes native BF16/FP32 GEMM rounding and
    can amplify the difference during Euler sampling. Keep the released
    module arithmetic, including separate LoRA additions, unchanged.
    """
    layers = tuple(layers)
    pieces = [
        extract_tables(model, torch.stack((timesteps[-1], t)), layers, output=output)
        for t in timesteps
    ]
    result = {"timesteps": timesteps.cpu().contiguous()}
    for layer in layers:
        key = f"block.{layer:03d}"
        result[key] = torch.cat(
            [piece[key][:, 3:6] for piece in pieces], dim=1
        ).contiguous()
    if output:
        result["output_modulations"] = torch.cat(
            [p["output_modulations"][:, 1:2] for p in pieces], dim=1
        ).contiguous()
        result["time_embedding"] = torch.cat(
            [p["time_embedding"][1:2] for p in pieces]
        ).contiguous()
    return result


def sampling_times(steps, shift, device="cpu", *, stage="causal"):
    # Exactly the sampler's float32 arithmetic, including the clean endpoint.
    if stage == "dmd":
        from ..schedulers.dmd import rollout_grid

        return torch.cat(
            (
                rollout_grid(steps, shift=shift, device=device),
                torch.ones(1, device=device),
            )
        )
    return grid(steps + 1, shift=shift, device=device)


def publish(folder, tensors, metadata, validation):
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError("AdaLN destination already exists: " + str(folder))
    temporary = folder.with_name("." + folder.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.mkdir(parents=True)
    try:
        save_file(tensors, str(temporary / PAYLOAD))
        manifest = dict(
            metadata,
            format=FORMAT,
            payload=PAYLOAD,
            payload_sha256=sha256(temporary / PAYLOAD),
            modalities=["video", "text", "audio"],
            modulation_order=list(ORDERS),
            time_direction="noise_0_clean_1",
            tensor_specs={
                n: dict(shape=list(t.shape), dtype=str(t.dtype))
                for n, t in tensors.items()
            },
        )
        (temporary / "validation.json").write_text(
            json.dumps(validation, indent=2) + "\n"
        )
        (temporary / "complete.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(folder)
        return manifest
    except BaseException:
        # Keep failed publications as evidence; never expose them as completed tables.
        raise


class SavedAdaLN:
    def __init__(
        self,
        folder,
        *,
        expected_binding=None,
        dtype=None,
        device="cpu",
        steps=None,
        shift=1.0,
    ):
        folder = Path(folder)
        metadata = json.loads((folder / "complete.json").read_text())
        if metadata.get("format") != FORMAT or metadata.get("payload") != PAYLOAD:
            raise ValueError("Unsupported AdaLN cache format")
        if metadata.get("modalities") != ["video", "text", "audio"] or metadata.get(
            "modulation_order"
        ) != list(ORDERS):
            raise ValueError("AdaLN modality/modulation order differs")
        if expected_binding is not None and metadata["binding"] != expected_binding:
            raise ValueError("AdaLN cache checkpoint/base/LoRA identity differs")
        if dtype is not None and metadata["dtype"] != dtype:
            raise ValueError("AdaLN cache execution precision differs")
        if sha256(folder / PAYLOAD) != metadata["payload_sha256"]:
            raise ValueError("AdaLN cache checksum differs")
        values = load_file(str(folder / PAYLOAD))
        specs = {
            n: dict(shape=list(t.shape), dtype=str(t.dtype)) for n, t in values.items()
        }
        if specs != metadata["tensor_specs"]:
            raise ValueError("AdaLN cache tensor inventory differs")
        cfg = metadata["binding"]["model"]["config"]
        times = values.get("timesteps")
        if (
            times is None
            or times.dtype != torch.float32
            or times.ndim != 1
            or not len(times)
        ):
            raise ValueError("AdaLN timesteps must be a nonempty float32 vector")
        if not bool(torch.isfinite(times).all()) or not bool(
            (times[1:] > times[:-1]).all()
        ):
            raise ValueError("AdaLN timesteps must be finite and strictly increasing")
        if (
            times[0].item() != 0.0
            or times[-1].item() != 1.0
            or metadata.get("time_direction") != "noise_0_clean_1"
        ):
            raise ValueError("AdaLN cache must include native noise/clean endpoints")
        count, hidden, layers = len(times), cfg["hidden_size"], cfg["num_layers"]
        expected = {
            "timesteps",
            "block_modulations",
            "output_modulations",
            "time_embedding",
        }
        if set(values) != expected or values["block_modulations"].shape != (
            layers,
            6,
            count * 3,
            hidden,
        ):
            raise ValueError("AdaLN block table shape differs")
        if values["output_modulations"].shape != (2, count, hidden) or values[
            "time_embedding"
        ].shape != (count, cfg["time_embed_dim"]):
            raise ValueError("AdaLN output/time table shape differs")
        projection_dtype = getattr(torch, metadata["dtype"])
        if (
            values["block_modulations"].dtype != projection_dtype
            or values["output_modulations"].dtype != projection_dtype
            or values["time_embedding"].dtype != torch.float32
        ):
            raise ValueError("AdaLN native projection/time precision differs")
        if not all(bool(torch.isfinite(v).all()) for v in values.values()):
            raise ValueError("Nonfinite AdaLN cache")
        if steps is not None:
            required = sampling_times(
                steps,
                shift,
                device=device,
                stage=metadata["binding"]["checkpoint"]["stage"],
            ).cpu()
            indices = torch.searchsorted(times, required).clamp_max(count - 1)
            if not torch.equal(times[indices], required):
                raise ValueError(
                    "AdaLN cache does not cover the exact inference time grid"
                )
        self.metadata = metadata
        self.identity = dict(
            format=FORMAT,
            binding=metadata["binding"],
            dtype=metadata["dtype"],
            payload_sha256=metadata["payload_sha256"],
        )
        self.timesteps = times.to(device)
        self.blocks = values["block_modulations"].to(device)
        self.output = values["output_modulations"].to(device)

    def time_indices(self, times):
        if times.dtype != torch.float32 or times.device != self.timesteps.device:
            raise ValueError(
                "AdaLN lookup requires native float32 times on the cache device"
            )
        indices = torch.searchsorted(self.timesteps, times)
        bounded = indices.clamp_max(len(self.timesteps) - 1)
        valid = (indices < len(self.timesteps)) & (self.timesteps[bounded] == times)
        if times.device.type == "cuda":
            torch._assert_async(valid.all(), "AdaLN cache has no exact timestep entry")
        elif not bool(valid.all()):
            raise ValueError("AdaLN cache has no exact timestep entry")
        return bounded
