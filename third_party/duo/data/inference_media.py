"""Publish codec packets with their own audio block, independently of generation."""

import json
from pathlib import Path

from .inference_export import atomic_json, digest


def packet_state(packets):
    import torch

    return [dict(p, frames=torch.from_numpy(p["frames"])) for p in packets]


def restore_packets(payload, generation_block):
    if "video_packets" in payload:
        return [dict(p, frames=p["frames"].numpy()) for p in payload["video_packets"]]
    return [dict(block=generation_block, frames=payload["frames"].numpy(), metrics={})]


def publish_packets(output, packets, waves, ffmpeg):
    from scripts.inference.utils import media

    output = Path(output)
    publications = []
    for packet in packets:
        block = packet["block"]
        if not 0 <= block < len(waves):
            raise ValueError(
                "Decoded video does not have a matching generated audio block"
            )
        folder = output / f"block-{block:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        wav, mp4 = folder / "assistant.tmp.wav", folder / "assistant.tmp.mp4"
        pcm = media.write_wav(wav, waves[block])
        media.write_video(mp4, packet["frames"], wav, ffmpeg)
        wav.replace(folder / "assistant.wav")
        mp4.replace(folder / "assistant.mp4")
        paths = dict(
            video=str(folder / "assistant.mp4"), audio=str(folder / "assistant.wav")
        )
        publication = dict(
            block=block,
            media=paths,
            pcm=pcm,
            sha256={
                f"{folder.name}/{name}": digest(folder / name)
                for name in ("assistant.wav", "assistant.mp4")
            },
        )
        publications.append(publication)
        # Earlier generation records remain owned by their generation block.
        # Only their previously pending media publication is completed here.
        marker = folder / "complete.json"
        if marker.exists():
            detail = json.loads(marker.read_text())
            detail.update(media=paths, video_status="complete")
            detail.setdefault("metrics", {}).update(packet["metrics"])
            atomic_json(detail, marker)
    return publications


def current_media(publications, block):
    return next((p["media"] for p in publications if p["block"] == block), {})


def check_flush(output, wanted):
    marker = Path(output) / "video_flush.json"
    if not marker.exists():
        return
    value = json.loads(marker.read_text())
    if value["generated_blocks"] != wanted:
        raise ValueError(
            "A flushed duo_5_17 prefix cannot be extended; use a new output to preserve published frames"
        )
    for name, checksum in value["media_sha256"].items():
        if digest(Path(output) / name) != checksum:
            raise ValueError("Final video publication checksum differs")


def mark_flush(output, generated, publications):
    atomic_json(
        dict(
            generated_blocks=generated,
            eof_context="last_two_latents",
            media_sha256={k: v for p in publications for k, v in p["sha256"].items()},
        ),
        Path(output) / "video_flush.json",
    )


def runtime_identity(local, codec, intent, cache):
    from torchtitan.models.duo_realization.modules.codec_spec import codec_identity

    expected = codec_identity(cache)
    if expected["mode"] != codec["mode"]:
        raise ValueError("Reference/features codec differs from decoder")
    if "weights" in expected and expected != codec:
        raise ValueError("Reference/features VAE weights or numerical settings differ")
    if cache.get("intent") and cache["intent"] != intent:
        raise ValueError("Feature Intent identity differs from selected adapter")
    value = dict(codec=codec, intent=intent)
    path = Path(local) / "codec_runtime.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("VAE/Intent runtime identity differs on resume")
    if not path.exists():
        atomic_json(value, path)


def validate_checkpoint_codec(checkpoint, runtime):
    if checkpoint is None:
        return
    if checkpoint.get("mode", "old") != runtime["mode"]:
        raise ValueError("Checkpoint encoding scheme differs from decoder")
    if "weights" in checkpoint and checkpoint != runtime:
        raise ValueError(
            "Checkpoint VAE weights or numerical configuration differ from decoder"
        )


def verify_runtime_files(local, args):
    """Validate frozen dependencies even when an already completed run is skipped."""
    path = Path(local) / "codec_runtime.json"
    if not path.exists():
        return  # Legacy old runs did not write this identity.
    saved = json.loads(path.read_text())
    weights = saved["codec"].get("weights", {})
    if weights:
        sana = Path(getattr(args, "sana_weights", ""))
        files = {
            "model.safetensors": Path(args.realization)
            / "FL2VA/video_vae/source/model.safetensors",
            "diffusion_pytorch_model.safetensors": Path(args.realization)
            / "audio_vae/diffusion_pytorch_model.safetensors",
            "taeh3_official.pth": sana / "tae/taeh3_official.pth",
            "causal_vit24_step6000.safetensors": sana
            / "causal_decoder/causal_vit24_step6000.safetensors",
        }
        for name, checksum in weights.items():
            if digest(files[name]) != checksum:
                raise ValueError("Frozen VAE weights changed on resume: " + name)
    intent = saved.get("intent", {})
    if not intent:
        return
    if intent.get("mode") == "origin":
        if not getattr(args, "origin_intent", False):
            raise ValueError("Original Intent resume requires --origin-intent")
        from torchtitan.models.duo_realization.modules.intent import prepare_tokenizer

        prepare_tokenizer(args)
        if args.intent_identity != intent:
            raise ValueError("Original Intent identity changed on resume")
        return
    if getattr(args, "origin_intent", False):
        raise ValueError("Adapter Intent state cannot resume with original Intent")
    adapter = Path(args.intent_adapter)
    for name, checksum in intent.get("adapter_weights", {}).items():
        if digest(adapter / name) != checksum:
            raise ValueError("Intent adapter changed on resume")
    manifest = intent.get("protocol_manifest_sha256")
    protocol = Path(
        getattr(args, "intent_protocol_manifest", "")
        or adapter / "protocol_manifest.json"
    )
    if manifest and digest(protocol) != manifest:
        raise ValueError("Intent protocol manifest changed on resume")
