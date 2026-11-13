"""CPU-only geometry and identity shared by Duo data, training and inference."""

from dataclasses import dataclass
from fractions import Fraction

VIDEO_MODES = ("old", "sana_1_4", "duo_2_5", "duo_5_17")
FPS = 24
SAMPLE_RATE = 32000
AUDIO_HOP = 800


@dataclass(frozen=True)
class VideoSpec:
    name: str
    frame_spans: tuple[int, ...]
    delayed: bool = False

    @property
    def frames(self):
        return sum(self.frame_spans)

    @property
    def latents(self):
        return len(self.frame_spans)

    @property
    def audio_latents(self):
        return ceil_fraction(Fraction(self.frames * SAMPLE_RATE, FPS * AUDIO_HOP))

    def spans(self, frame_start):
        result = []
        for frames in self.frame_spans:
            stop = frame_start + frames
            result.append([frame_start * 1000 / FPS, stop * 1000 / FPS])
            frame_start = stop
        return result

    def identity(self):
        return dict(
            version=1,
            mode=self.name,
            fps=FPS,
            frames_per_block=self.frames,
            video_latents_per_block=self.latents,
            frame_spans=list(self.frame_spans),
            audio_sample_rate=SAMPLE_RATE,
            audio_hop=AUDIO_HOP,
            audio_latents_per_block=self.audio_latents,
            posterior="sample"
            if self.name == "old"
            else "deterministic_tae"
            if self.name == "sana_1_4"
            else "mean",
            normalization="tae_normalized" if self.name == "sana_1_4" else "duo_channel_mean_std",
            decode_policy={
                "old": "raw_window_last4",
                "sana_1_4": "anchor_continuous_kv4",
                "duo_2_5": "typed_history4_crop17_22",
                "duo_5_17": "exact_delay_eof_last_two",
            }[self.name],
        )


def spec(mode="old"):
    if mode not in VIDEO_MODES:
        raise ValueError(f"Unknown video codec {mode!r}; choose {VIDEO_MODES}")
    return VideoSpec(
        mode,
        {"old": (4,), "sana_1_4": (4,), "duo_2_5": (1, 4), "duo_5_17": (1, 4, 4, 4, 4)}[mode],
        mode == "duo_5_17",
    )


def ceil_fraction(value):
    return -(-value.numerator // value.denominator)


def mode_from_args(args):
    return getattr(args, "decode", None) or getattr(args, "encode", None) or "old"


def codec_identity(value):
    """Missing codec fields are the legacy old protocol, never an inferred shape."""
    identity = value.get("codec")
    if identity is None:
        return spec("old").identity()
    expected = spec(identity["mode"]).identity()
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Invalid codec identity")
    return identity


def validate_codec(value, mode):
    identity = codec_identity(value)
    if identity["mode"] != mode:
        raise ValueError(
            f"Codec mismatch: features/checkpoint={identity['mode']}, requested={mode}"
        )
    return identity


def validate_media(item, mode=None):
    geometry = spec(codec_identity(item)["mode"])
    if mode is not None:
        validate_codec(item, mode)
    h, w = item["reference"].shape[-2:]
    if tuple(item["reference"].shape) != (24, 1, h, w):
        raise ValueError("Reference must contain one 24-channel latent")
    for block in item["blocks"]:
        media = block["assistant"]
        if tuple(media["video"].shape) != (24, geometry.latents, h, w):
            raise ValueError("Video feature geometry differs from codec")
        if tuple(media["audio"].shape) != (2, 32, geometry.audio_latents):
            raise ValueError("Audio feature geometry differs from codec")
        if (
            len(media["video_weight"]) != geometry.latents
            or len(media["audio_weight"]) != geometry.audio_latents
        ):
            raise ValueError("Media weights differ from codec")
    return geometry


def encode_cli_args(args):
    """Top-level public alias shared by training and launcher preflight."""
    aliases = {
        "--encode": "--duo.encode",
        "--base-model": "--duo.base-model",
        "--only-bidirectional-loss": "--duo.only-bidirectional-loss",
        "--no-only-bidirectional-loss": "--duo.no-only-bidirectional-loss",
    }
    return [
        aliases.get(x.split("=", 1)[0], x.split("=", 1)[0])
        + ("=" + x.split("=", 1)[1] if "=" in x else "")
        for x in args
    ]
