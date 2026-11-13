"""Intent rollout inputs independent of Realization feature extraction."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

LAYOUT = "ntp_description_body_v1"
POLICY = "intent_reference_rollout_v2"
H50_STRATEGY = "ntp_forward_layer50_v1"


def build_system_prompt(chunk_token, description, explicit_speak_token=False):
    description = description.strip()
    if not description or any(
        t in description for t in ("<|", "<wait>", "<blank>", "<chunk_")
    ):
        raise ValueError("Missing/invalid Assistant description")
    instruction = (
        "Begin every speaking chunk with <speak>. Output <wait> if the assistant is "
        "waiting or continuing an unfinished utterance."
        if explicit_speak_token
        else "Predict what the assistant should say in the next chunk. Output <wait> if the "
        "assistant is waiting or continuing an unfinished utterance."
    )
    return (
        f"{chunk_token} You are the assistant in a real-time full-duplex video call.\n"
        "The assistant's visual appearance, voice and scene are described as follows:\n"
        f'"{description}"\n{instruction}'
    )


def parse_generation(ids, wait_id, speak_id=None):
    """Reference action parsing; never normalize ordinary wait/content mixtures."""
    if not ids:
        return [wait_id], [], "wait", True
    if speak_id is None:
        waiting = ids[0] == wait_id
        return (
            list(ids),
            [] if waiting else list(ids),
            "wait" if waiting else "speak",
            True,
        )
    if ids[0] == wait_id:
        return [wait_id], [], "wait", True
    if ids[0] == speak_id:
        return list(ids), list(ids[1:]), "speak", True
    return [wait_id], [], "wait", False


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _protocol_failure(tokenizer, field, expected, actual):
    source = getattr(tokenizer, "duplex_rollout_protocol_source", {})
    raise AssertionError(
        "Intent rollout protocol mismatch "
        f"[model={source.get('model', '<unbound>')} "
        f"adapter={source.get('adapter', '<none>')} "
        f"protocol={source.get('protocol', '<unbound>')}]: "
        f"{field} expected={expected!r}, actual={actual!r}"
    )


def _require_protocol(tokenizer, field, expected, actual):
    if actual != expected:
        _protocol_failure(tokenizer, field, expected, actual)


def _encoded_control(tokenizer, token, *, required=True):
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    if required:
        _protocol_failure(tokenizer, token + " encoding", "one token", ids)
    return tokenizer.convert_tokens_to_ids(token)


def _check_manifest_clock(tokenizer, protocol, manifest):
    for field in ("frames_per_block", "fps"):
        if field in manifest:
            expected = protocol["frames" if field == "frames_per_block" else "fps"]
            _require_protocol(tokenizer, field, expected, manifest[field])
    clock = manifest.get("chunk_clock", {})
    if clock:
        if "fps" in clock:
            _require_protocol(
                tokenizer, "chunk_clock.fps", protocol["fps"], clock["fps"]
            )
        if "frames" in clock:
            _require_protocol(
                tokenizer, "chunk_clock.frames", [protocol["frames"]], clock["frames"]
            )
    if manifest.get("causal_context") is not None:
        _require_protocol(
            tokenizer,
            "causal_context",
            "A_<i text + U_<i text/user_video",
            manifest["causal_context"],
        )


def bind_rollout_protocol(tokenizer, args, adapter_config=None):
    """Bind execution semantics separately from the existing cached weight identity."""
    from .codec_spec import mode_from_args, spec

    geometry = spec(mode_from_args(args))
    origin = getattr(tokenizer, "duplex_origin_intent", False) is True
    adapter = str(getattr(args, "intent_adapter", "") or "")
    external = str(getattr(args, "intent_protocol_manifest", "") or "")
    tokenizer.duplex_rollout_protocol_source = dict(
        model=str(getattr(args, "intent", "<unknown>")),
        adapter=adapter or "<none>",
        protocol=external
        or (str(Path(adapter) / "protocol_manifest.json") if adapter else "<origin>"),
    )
    manifest = getattr(tokenizer, "duplex_protocol_manifest", None) or {}
    options = manifest.get("training_options", {})
    if not isinstance(options, dict):
        _protocol_failure(tokenizer, "training_options", "mapping or missing", options)
    explicit = options.get("explicit_speak_token", False)
    if type(explicit) is not bool:
        _protocol_failure(tokenizer, "explicit_speak_token", "bool", explicit)
    if origin and explicit:
        _protocol_failure(tokenizer, "origin explicit_speak_token", False, explicit)
    chunk = getattr(tokenizer, "duplex_chunk_token", None)
    if not isinstance(chunk, str) or not chunk:
        _protocol_failure(tokenizer, "chunk_token", "resolved nonempty token", chunk)
    wait = _encoded_control(tokenizer, "<wait>", required=not origin)
    trained = (adapter_config or {}).get("trainable_token_indices")
    if isinstance(trained, dict):
        trained = [i for ids in trained.values() for i in ids]
    if trained is None:
        trained = manifest.get("trainable_token_ids")
    tokenizer._duplex_rollout_trained_ids = (
        None if trained is None else frozenset(trained)
    )
    speak = None
    if explicit:
        declared = manifest.get("special_tokens", {}).get("speak")
        option = options.get("speak_token")
        if not isinstance(declared, dict) or option != declared:
            _protocol_failure(tokenizer, "speak declaration", declared, option)
        _require_protocol(tokenizer, "speak text", "<speak>", declared.get("text"))
        speak = _encoded_control(tokenizer, "<speak>")
        _require_protocol(tokenizer, "speak token ID", declared.get("id"), speak)
        for label, ids in (
            ("adapter trainable tokens", tokenizer._duplex_rollout_trained_ids),
            ("manifest trainable tokens", manifest.get("trainable_token_ids")),
        ):
            if ids is None or speak not in ids:
                _protocol_failure(tokenizer, label + " speak membership", speak, ids)
    if not origin and not explicit and "<speak>" in tokenizer.get_vocab():
        token = _encoded_control(tokenizer, "<speak>", required=False)
        if (
            tokenizer._duplex_rollout_trained_ids
            and token in tokenizer._duplex_rollout_trained_ids
        ):
            _protocol_failure(tokenizer, "trained speak action", True, explicit)
    identity = getattr(args, "intent_identity", {})
    protocol = dict(
        mode="origin" if origin else "explicit_speak" if explicit else "legacy",
        explicit_speak=explicit,
        wait_id=wait,
        speak_id=speak,
        chunk_token=chunk,
        frames=geometry.frames,
        fps=24,
        context_shift=1,
        protocol_sha256=identity.get("protocol_manifest_sha256"),
    )
    _check_manifest_clock(tokenizer, protocol, manifest)
    tokenizer.duplex_rollout_protocol = protocol
    return assert_rollout_protocol(tokenizer)


def assert_rollout_protocol(tokenizer, inputs=None):
    """Explicit raises survive Python -O; reject weights/input-policy mismatches."""
    protocol = getattr(tokenizer, "duplex_rollout_protocol", None)
    fields = (
        "mode",
        "explicit_speak",
        "wait_id",
        "speak_id",
        "chunk_token",
        "frames",
        "fps",
        "context_shift",
        "protocol_sha256",
    )
    if not isinstance(protocol, dict) or any(k not in protocol for k in fields):
        _protocol_failure(tokenizer, "bound rollout protocol", fields, protocol)
    manifest = getattr(tokenizer, "duplex_protocol_manifest", None) or {}
    options = manifest.get("training_options", {})
    explicit = (
        options.get("explicit_speak_token", False)
        if isinstance(options, dict)
        else None
    )
    _require_protocol(
        tokenizer, "explicit_speak_token", protocol["explicit_speak"], explicit
    )
    _require_protocol(
        tokenizer,
        "chunk token",
        protocol["chunk_token"],
        getattr(tokenizer, "duplex_chunk_token", None),
    )
    _require_protocol(tokenizer, "context_shift", 1, protocol["context_shift"])
    _require_protocol(tokenizer, "fps", 24, protocol["fps"])
    origin = getattr(tokenizer, "duplex_origin_intent", False) is True
    _require_protocol(
        tokenizer,
        "mode",
        "origin" if origin else "explicit_speak" if explicit else "legacy",
        protocol["mode"],
    )
    _require_protocol(
        tokenizer,
        "wait token ID",
        protocol["wait_id"],
        _encoded_control(tokenizer, "<wait>", required=not origin),
    )
    _check_manifest_clock(tokenizer, protocol, manifest)
    if explicit:
        declared = manifest.get("special_tokens", {}).get("speak")
        _require_protocol(
            tokenizer, "speak declaration", declared, options.get("speak_token")
        )
        if not isinstance(declared, dict):
            _protocol_failure(tokenizer, "speak declaration", "mapping", declared)
        actual = _encoded_control(tokenizer, "<speak>")
        _require_protocol(
            tokenizer, "declared speak token ID", declared.get("id"), actual
        )
        _require_protocol(
            tokenizer, "bound speak token ID", protocol["speak_id"], actual
        )
        trained = getattr(tokenizer, "_duplex_rollout_trained_ids", None)
        if trained is None or actual not in trained:
            _protocol_failure(tokenizer, "trained speak membership", actual, trained)
    if inputs is not None:
        _require_protocol(
            tokenizer,
            "inputs.rollout_protocol",
            protocol,
            getattr(inputs, "rollout_protocol", None),
        )
        _require_protocol(
            tokenizer,
            "inputs.explicit_speak",
            protocol["explicit_speak"],
            getattr(inputs, "explicit_speak", None),
        )
        description = getattr(inputs, "description", None)
        if not isinstance(description, str) or not description.strip():
            _protocol_failure(
                tokenizer, "inputs.description", "nonempty string", description
            )
        expected = build_system_prompt(
            protocol["chunk_token"], description, protocol["explicit_speak"]
        )
        _require_protocol(
            tokenizer, "inputs.system", expected, getattr(inputs, "system", None)
        )
        geometry = getattr(inputs, "geometry", None)
        if geometry is not None:
            _require_protocol(
                tokenizer, "inputs block frames", protocol["frames"], geometry.frames
            )
    return dict(protocol)


def encode_assistant_body(tokenizer, ids, *, speaking):
    """One GT action formatter shared by DMD and reference-prefix construction."""
    protocol = assert_rollout_protocol(tokenizer)
    body = list(ids)
    if not speaking or not body:
        return [protocol["wait_id"]]
    if body[0] == protocol["wait_id"]:
        _protocol_failure(tokenizer, "speaking GT body", "speech tokens", body)
    if protocol["explicit_speak"] and body[0] != protocol["speak_id"]:
        body.insert(0, protocol["speak_id"])
    return body


class RolloutInputs:
    def __init__(self, record, tokenizer, processor, args):
        import torch

        canonical, raw_path = self._initialize(record, tokenizer, args)
        video = canonical.get("videos", {}).get("user", canonical.get("video", {}))
        cache = video.get("cache")
        if cache:
            cache = Path(cache)
            if not cache.is_absolute():
                base = raw_path.parent if raw_path else Path(".")
                if canonical.get("path_base") == "dataset_root":
                    base = next(
                        (
                            p
                            for p in (raw_path.parents if raw_path else ())
                            if (p / "dataset_manifest.json").is_file()
                        ),
                        base,
                    )
                cache = base / cache
        pattern = video.get("frame_pattern")
        selected_source = record["canonical"]["source"]["user_video"]
        source_matches = (
            Path(canonical["source"]["user_video"]).resolve()
            == Path(selected_source).resolve()
        )
        compatible = (
            source_matches
            and (
                pattern == [self.geometry.frames]
                or (pattern is None and self.geometry.frames % 2 == 0)
            )
            and video.get("source_fps", 24) == 24
            and video.get("crop", "full") == "full"
            and video.get("min_pixels", 1024) == 1024
            and video.get("max_pixels", args.intent_max_pixels)
            == args.intent_max_pixels
        )
        if cache and cache.is_file() and compatible:
            from safetensors.torch import load_file

            patches = load_file(str(cache))["patches"]
            self.video, self.patches = video, patches
            self.identity.update(
                video_patches_sha256=_sha(cache), video_source="canonical_patches"
            )
        else:
            self.video, self.patches = self._video(record, processor, args)
            self.identity.update(
                video_source="reference_chunk_patches",
                user_video_sha256=_sha(record["canonical"]["source"]["user_video"]),
            )
        if len(self.patches) != len(self.video["timeline"]):
            raise ValueError("Intent video timeline/patch count differs")
        self.patches = self.patches.to(dtype=torch.float16, device="cpu")
        self.grid_hw = self.video["grid_hw"]
        self.merge = self.video["spatial_merge_size"]

    def _initialize(self, record, tokenizer, args):
        from .codec_spec import mode_from_args, spec
        from .intent_conditioning import describe

        self.tokenizer = tokenizer
        self.geometry = spec(mode_from_args(args))
        self.rollout_protocol = assert_rollout_protocol(tokenizer)
        raw_path = (
            Path(record["canonical_path"]) if record.get("canonical_path") else None
        )
        canonical = (
            json.loads(raw_path.read_text())
            if raw_path
            else deepcopy(record["canonical"])
        )
        self.canonical = canonical
        description = canonical.get("assistant_description", {}).get("text")
        if record.get("assistant_description_override") is not None or not description:
            description = describe(record)["text"]
        self.description = description.strip()
        self.explicit_speak = self.rollout_protocol["explicit_speak"]
        self.system = build_system_prompt(
            tokenizer.duplex_chunk_token, self.description, self.explicit_speak
        )
        self.units = []
        view_path = (
            raw_path.parent.parent
            / "views"
            / (tokenizer.duplex_chunk_token.strip("<>") + ".json")
            if raw_path
            else None
        )
        view = (
            json.loads(view_path.read_text())
            if view_path and view_path.is_file()
            else None
        )
        view_compatible = (
            view is not None
            and view.get("chunk_token", {}).get("text") == tokenizer.duplex_chunk_token
            and len(view["units"]) >= len(record["blocks"])
            and all(
                abs(u["start_ms"] - round(b["start_ms"])) <= 1
                and abs(u["end_ms"] - round(b["end_ms"])) <= 1
                for u, b in zip(view["units"], record["blocks"])
            )
        )
        if not view_compatible:
            view = None
        for k, block in enumerate(record["blocks"]):
            lo, hi = round(block["start_ms"]), round(block["end_ms"])
            if view is not None:
                unit = view["units"][k]
                lo, hi = unit["start_ms"], unit["end_ms"]
                content = unit["user"]["text"]
                if view.get("emotion", {}).get("default_mode") == "duplex":
                    content += "".join(
                        e["text"] for e in unit.get("emotion", {}).get("user_input", [])
                    )
            else:
                ids = [
                    t["token_id"]
                    for t in canonical["text_tokens"]
                    if t["role"] == "user" and lo <= t["commit_ms"] < hi
                ]
                content = (
                    tokenizer.decode(ids, skip_special_tokens=False)
                    if ids
                    else "<blank>"
                )
                emotions = canonical.get("emotions", {})
                if emotions.get("default_mode") == "duplex":
                    content += "".join(
                        e["user_token"]
                        for e in emotions.get("attachments", [])
                        if e.get("role") == "user"
                        and e.get("usable_for_training")
                        and e.get("user_token")
                        and e.get("commit_ms") is not None
                        and lo <= e["commit_ms"] < hi
                    )
            self.units.append(
                dict(block, unit_id=k, text=content, start_ms=lo, end_ms=hi)
            )
        self.identity = dict(
            canonical_sha256=(
                _sha(raw_path)
                if raw_path
                else hashlib.sha256(
                    json.dumps(canonical, sort_keys=True).encode()
                ).hexdigest()
            ),
            system_prompt=self.system,
            view_sha256=_sha(view_path) if view is not None else None,
            user_text_source=(
                "canonical_view" if view is not None else "canonical_commit_clock"
            ),
            rollout_policy=POLICY,
            rollout_protocol=dict(self.rollout_protocol),
        )
        self._predecoded_patches = None
        assert_rollout_protocol(tokenizer, self)
        return canonical, raw_path

    @classmethod
    def from_visuals(cls, record, tokenizer, visuals, args):
        """Use already decoded per-block User patches without reading media again."""
        import torch

        self = cls.__new__(cls)
        self._initialize(record, tokenizer, args)
        self._predecoded_patches, self._predecoded_visuals = [], []
        checksum = hashlib.sha256()
        for k, unit in enumerate(self.units):
            spec = visuals[("user", k)]
            grid = list(spec["grid"])
            if len(grid) != 3 or any(type(v) is not int or v < 1 for v in grid):
                raise ValueError("Invalid predecoded User grid at block " + str(k))
            t, h, w = grid
            merge = spec["merge"]
            if type(merge) is not int or merge < 1 or h % merge or w % merge:
                raise ValueError("Invalid predecoded User merge at block " + str(k))
            frames = unit["frame_stop"] - unit["frame_start"]
            if t != (frames + 1) // 2:
                raise ValueError(
                    "Predecoded User grid clock differs at block " + str(k)
                )
            patches = torch.as_tensor(spec["patches"]).detach()
            if patches.device.type != "cpu":
                raise ValueError("Predecoded User patches must be CPU tensors")
            if patches.ndim == 2 and patches.shape[0] == t * h * w:
                temporal = patches.reshape(t, h * w, -1)
            elif patches.ndim == 4 and list(patches.shape[:3]) == grid:
                temporal = patches.reshape(t, h * w, -1)
            else:
                raise ValueError("Predecoded User patch shape differs from its grid")
            temporal = temporal.contiguous()
            first = len(self._predecoded_patches)
            indices = list(range(first, first + t))
            self._predecoded_patches.extend(temporal.unbind(0))
            count = h * w // merge**2
            content = "".join(
                f"<{round(min(unit['frame_start']+2*j+1,unit['frame_stop']-1)*1000/24)/1000:.1f} seconds>"
                + "<|vision_start|>"
                + "<|video_pad|>" * count
                + "<|vision_end|>"
                for j in range(t)
            )
            self._predecoded_visuals.append((content + unit["text"], indices, grid))
            checksum.update(
                json.dumps(
                    dict(grid=grid, merge=merge, dtype=str(temporal.dtype))
                ).encode()
            )
            checksum.update(temporal.view(torch.uint8).numpy().tobytes())
        self.identity.update(
            video_source="predecoded_block_patches",
            video_patches_sha256=checksum.hexdigest(),
        )
        assert_rollout_protocol(tokenizer, self)
        return self

    def _video(self, record, processor, args):
        import numpy as np
        import torch

        from .media_features import read_video

        prep = record.get("preparation", {})
        duration = (
            prep.get("original_frames", record["canonical"]["video"]["source_frames"])
            * 1000
            / 24
        )
        source = read_video(
            record["canonical"]["source"]["user_video"], duration, None, None
        )
        frames, frame_ids = [], []
        for start in range(0, len(source["frames"]), self.geometry.frames):
            ids = list(
                range(start, min(start + self.geometry.frames, len(source["frames"])))
            )
            if len(ids) % 2:
                ids.append(ids[-1])
            frames.extend(source["frames"][ids])
            frame_ids.extend(ids)
        kwargs = dict(
            videos=[np.stack(frames)],
            do_sample_frames=False,
            size=dict(shortest_edge=1024, longest_edge=args.intent_max_pixels),
            return_tensors="pt",
        )
        try:
            from transformers.video_utils import VideoMetadata

            kwargs["video_metadata"] = [
                VideoMetadata(
                    total_num_frames=len(source["frames"]),
                    fps=24,
                    width=frames[0].shape[1],
                    height=frames[0].shape[0],
                    duration=len(source["frames"]) / 24,
                    frames_indices=frame_ids,
                )
            ]
        except ImportError:
            pass  # Older RAVEN Transformers does not expose VideoMetadata.
        result = processor.video_processor(**kwargs)
        t, h, w = result["video_grid_thw"][0].tolist()
        patches = result["pixel_values_videos"].reshape(t, h, w, -1).to(torch.float16)
        timeline = [
            dict(
                index=k, commit_ms=round(max(frame_ids[2 * k : 2 * k + 2]) * 1000 / 24)
            )
            for k in range(t)
        ]
        return (
            dict(
                grid_hw=[h, w],
                spatial_merge_size=processor.video_processor.merge_size,
                timeline=timeline,
            ),
            patches,
        )

    def visual(self, k):
        if self._predecoded_patches is not None:
            content, indices, grid = self._predecoded_visuals[k]
            return content, list(indices), list(grid)
        unit = self.units[k]
        indices = [
            p["index"]
            for p in self.video["timeline"]
            if unit["start_ms"] <= p["commit_ms"] < unit["end_ms"]
        ]
        h, w = self.grid_hw
        text = "".join(
            f"<{self.video['timeline'][i]['commit_ms']/1000:.1f} seconds><|vision_start|>"
            + "<|video_pad|>" * (h * w // self.merge**2)
            + "<|vision_end|>"
            for i in indices
        )
        return text + unit["text"], indices, ([len(indices), h, w] if indices else None)

    def pixels(self, indices):
        if self._predecoded_patches is not None:
            if not indices:
                return None
            import torch

            return torch.cat([self._predecoded_patches[i] for i in indices])
        return (
            self.patches[indices].reshape(-1, self.patches.shape[-1])
            if indices
            else None
        )
