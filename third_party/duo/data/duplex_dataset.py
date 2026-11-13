"""Canonical duplex records with a stable train/validation partition."""

import json
from pathlib import Path

from data.dataset_info import DUPLEX_MODALITIES
from data.dataset_utils import activity_intervals, aligned_words, digest


class DuplexDataset:
    def __init__(self, config, split=None):
        self.root = Path(config["root"]).expanduser().resolve()
        self.config = config
        self.records, self.rejections = [], []
        tail_policy = config.get("tail_policy", "trim")
        if tail_policy not in ("trim", "pad"):
            raise ValueError("tail_policy must be trim or pad")
        requested = config.get("sample_ids")
        if isinstance(requested, str):
            requested = requested.split(",")
        if requested is not None and (
            not requested or len(set(requested)) != len(requested)
        ):
            raise ValueError("sample_ids must be nonempty and unique")
        manifest = Path(
            config.get("manifest", self.root.parent / "dataset_manifest.json")
        )
        for item in json.loads(manifest.read_text())["samples"]:
            sid = item["sample_id"]
            if requested is not None and sid not in requested:
                continue
            if Path(sid).name != sid:
                raise ValueError("sample_id must be a single directory name")
            if item["status"] != "accept" or item.get("overlap_turns", 0):
                self.rejections.append(
                    dict(sample_id=sid, reason="review_or_script_overlap")
                )
                continue
            path = self.root / sid / "canonical/canonical.json"
            canonical = json.loads(path.read_text())
            if canonical["sample_id"] != sid:
                raise ValueError(f"Sample ID differs in {path}")
            preparation = dict(
                resolution=config.get("resolution", "fixed"),
                audio_source=config.get("role_audio_source", {}).get(sid, "mixed"),
            )
            if preparation["resolution"] not in ("fixed", "native"):
                raise ValueError("resolution must be fixed or native")
            if preparation["audio_source"] not in ("mixed", "role"):
                raise ValueError("role_audio_source values must be mixed or role")
            from torchtitan.models.duo_realization.modules.codec_spec import spec

            geometry = spec(config.get("video_codec", "old"))
            if tail_policy == "pad":
                frames = int(canonical["video"]["source_frames"])
                if frames < 1 or canonical["video"]["source_fps"] != 24:
                    raise ValueError("Padding requires a nonempty 24 fps source")
                padded = (
                    (frames + geometry.frames - 1) // geometry.frames * geometry.frames
                )
                duration = padded * 1000 / 24
                if any(w["end_ms"] > duration for w in aligned_words(canonical)):
                    raise ValueError(
                        "Aligned speech exceeds padded video timeline: " + sid
                    )
                samples = (frames * 32000 + 23) // 24
                padded_samples = (padded * 32000 + 23) // 24
                preparation.update(
                    tail_policy="pad",
                    padding_version="repeat_frame_silence_v1",
                    original_frames=frames,
                    kept_frames=frames,
                    padded_frames=padded - frames,
                    output_frames=padded,
                    padded_frame_range=[frames, padded],
                    padded_time_ms=[frames * 1000 / 24, duration],
                    dropped_frames=0,
                    original_duration_ms=canonical["duration_ms"],
                    original_audio_samples=samples,
                    output_audio_samples=padded_samples,
                    audio_tail_pad_samples=padded_samples - samples,
                    audio_tail_pad_range=[samples, padded_samples],
                )
                canonical["video"]["source_frames"] = padded
                canonical["duration_ms"] = duration
            elif config.get("trim_incomplete_block", False) or "video_codec" in config:
                frames = canonical["video"]["source_frames"]
                kept = frames // geometry.frames * geometry.frames
                if kept < geometry.frames or canonical["video"]["source_fps"] != 24:
                    raise ValueError(
                        "Trimming requires at least one complete codec block at 24 fps"
                    )
                duration = kept * 1000 / 24
                if any(w["end_ms"] > duration for w in aligned_words(canonical)):
                    raise ValueError(
                        "Tail trimming would remove aligned speech: " + sid
                    )
                preparation.update(
                    original_frames=frames,
                    kept_frames=kept,
                    dropped_frames=frames - kept,
                    dropped_frame_range=[kept, frames],
                    dropped_time_ms=[duration, canonical["duration_ms"]],
                    original_duration_ms=canonical["duration_ms"],
                )
                canonical["video"]["source_frames"] = kept
                canonical["duration_ms"] = duration
                canonical["text_tokens"] = [
                    t for t in canonical["text_tokens"] if t["commit_ms"] < duration
                ]
            try:
                activity = activity_intervals(canonical)
            except ValueError as exc:
                self.rejections.append(dict(sample_id=sid, reason=str(exc)))
                continue
            if config.get("media_root"):
                layout = dict(DUPLEX_MODALITIES, **config.get("modalities", {}))
                canonical["source"].update(
                    {
                        name: str(
                            Path(config["media_root"]) / pattern.format(sample_id=sid)
                        )
                        for name, pattern in layout.items()
                    }
                )
            self.records.append(
                dict(
                    sample_id=sid,
                    canonical=canonical,
                    canonical_path=str(path),
                    media=dict(canonical["source"]),
                    text=canonical["text_tokens"],
                    activity=activity,
                    audio_provenance="turn_gated_mixture_approximation",
                    preparation=preparation,
                    assistant_description_override=config.get(
                        "assistant_descriptions", {}
                    ).get(sid),
                )
            )
        seed = config.get("seed", 42)
        ranked = sorted(self.records, key=lambda r: digest([seed, r["sample_id"]]))
        if config.get("split_file"):
            partition = json.loads(Path(config["split_file"]).read_text())
            train, validation = set(partition["train"]), set(partition["validation"])
            if train & validation or {r["sample_id"] for r in ranked} - (
                train | validation
            ):
                raise ValueError("Invalid dataset split")
        else:
            fraction = config.get("validation_fraction", 0.1)
            if not 0 <= fraction < 1:
                raise ValueError("validation_fraction must be in [0, 1)")
            n = (
                min(max(1, round(len(ranked) * fraction)), max(0, len(ranked) - 1))
                if fraction
                else 0
            )
            validation = {r["sample_id"] for r in ranked[:n]}
        if config.get("max_samples", 0):
            val = [r for r in ranked if r["sample_id"] in validation]
            train = [r for r in ranked if r["sample_id"] not in validation]
            self.records = (val[:1] + train[:1] + val[1:] + train[1:])[
                : config["max_samples"]
            ]
        for record in self.records:
            record["split"] = (
                "validation" if record["sample_id"] in validation else "train"
            )
        if split:
            self.records = [r for r in self.records if r["split"] == split]
        if requested is not None:
            by_id = {r["sample_id"]: r for r in self.records}
            if set(requested) - by_id.keys():
                raise ValueError(
                    "Requested samples unavailable in selected split: "
                    + str(sorted(set(requested) - by_id.keys()))
                )
            self.records = [by_id[sid] for sid in requested]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def summary(self):
        return dict(
            conversations=len(self),
            rejections=self.rejections,
            **{
                s: sum(r["split"] == s for r in self.records)
                for s in ("train", "validation")
            },
        )
