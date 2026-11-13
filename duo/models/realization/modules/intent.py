"""Frozen team Intent adapter, trained control tokens and causal serialization."""

from __future__ import annotations
import importlib
import json
from pathlib import Path

PROTOCOL = "streaming_special_planning_v2"
EMOTIONS = ("happy", "sad", "angry", "fearful", "disgusted", "surprised")
SPECIALS = ["<wait>", "<blank>", "<chunk_160>"] + [f"<emo_{e}>" for e in EMOTIONS]
data = importlib.import_module("duo.data.dataset_utils")


def resolve_chunk_token(args, tokenizer, adapter_config):
    from duo.models.realization.modules.codec_spec import mode_from_args, spec

    geometry = spec(mode_from_args(args))
    adapter = Path(args.intent_adapter)
    external = getattr(args, "intent_protocol_manifest", "")
    path = Path(external) if external else adapter / "protocol_manifest.json"
    if external and not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text()) if path.is_file() else None
    requested = getattr(args, "intent_chunk_token", "")
    if manifest is None:
        if geometry.name not in ("old", "sana_1_4"):
            raise ValueError("New codec modes require the adapted Intent protocol_manifest.json")
        token = requested or "<chunk_160>"
    else:
        declared = manifest.get("chunk_tokens", {})
        choices = {v["text"]: v for v in declared.values()}
        if requested:
            token = requested
        elif len(choices) == 1:
            token = next(iter(choices))
        else:
            raise ValueError(
                "Intent declares multiple chunk tokens; specify --intent-chunk-token"
            )
        ids = tokenizer.encode(token, add_special_tokens=False)
        if token not in choices or len(ids) != 1 or choices[token]["id"] != ids[0]:
            raise ValueError("Intent chunk token differs from protocol manifest/tokenizer")
        for declaration in (manifest, choices[token]):
            for key, expected in (("frames_per_block", geometry.frames), ("fps", 24)):
                if key in declaration and declaration[key] != expected:
                    raise ValueError("Intent manifest clock differs from video codec: " + key)
    tokenizer.reciprocal_chunk_token = token
    tokenizer.reciprocal_protocol_manifest = manifest
    if manifest is not None:
        for field, filename in (
            ("tokenizer_json_sha256", "tokenizer.json"),
            ("chat_template_sha256", "chat_template.jinja"),
        ):
            if (
                manifest.get(field)
                and __import__("hashlib").sha256((adapter / filename).read_bytes()).hexdigest()
                != manifest[field]
            ):
                raise ValueError("Intent protocol checksum differs: " + filename)
        key = next(key for key, value in declared.items() if value["text"] == token)
        allowed_ms = {4: {"160", "166", "167"}, 5: {"208"}, 17: {"708"}}[geometry.frames]
        if key not in allowed_ms:
            raise ValueError("Intent chunk clock differs from selected codec")
    args.intent_identity = dict(
        chunk_token=token,
        protocol_manifest_sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest()
        if manifest
        else None,
        frames_per_block=geometry.frames,
        fps=24,
        adapter_weights={
            p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(args.intent_adapter).glob("*.safetensors"))
        },
    )
    if external:
        for field, filename in (
            ("adapter_model_sha256", "adapter_model.safetensors"),
            ("adapter_config_sha256", "adapter_config.json"),
        ):
            actual = __import__("hashlib").sha256((adapter / filename).read_bytes()).hexdigest()
            if manifest.get(field) != actual:
                raise ValueError("External Intent protocol does not match adapter: " + filename)
    return token


def verify_tokens(tokenizer, adapter_config, chunk_token="<chunk_160>"):
    trained = adapter_config["trainable_token_indices"]
    trained = (
        set(i for values in trained.values() for i in values)
        if isinstance(trained, dict)
        else set(trained)
    )
    manifest = getattr(tokenizer, "reciprocal_protocol_manifest", None)
    specials = [t if t != "<chunk_160>" else chunk_token for t in SPECIALS]
    if manifest is not None:
        declared = manifest["special_tokens"]
        entries = [declared["wait"], declared["blank"], *declared.get("emotions", {}).values()]
        if declared.get("speak") is not None:
            entries.append(declared["speak"])
        specials = [e["text"] for e in entries] + [chunk_token]
        for entry in entries:
            if tokenizer.encode(entry["text"], add_special_tokens=False) != [entry["id"]]:
                raise ValueError("Intent special token differs from manifest")
        if set(manifest["trainable_token_ids"]) != trained:
            raise ValueError("Intent trained token IDs differ from adapter config")
    ids = {}
    for token in specials:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] not in trained:
            raise ValueError(f"missing trained special token: {token}")
        ids[token] = encoded[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError("special token IDs collide")
    return ids


def explicit_speak_token(tokenizer):
    """Return the declared action token for v2 checkpoints, otherwise ``None``."""
    manifest = getattr(tokenizer, "reciprocal_protocol_manifest", None)
    if not manifest or not manifest.get("training_options", {}).get("explicit_speak_token"):
        return None
    option = manifest["training_options"].get("speak_token")
    declaration = manifest.get("special_tokens", {}).get("speak")
    if not isinstance(option, dict) or option != declaration:
        raise ValueError("Intent explicit speak declaration differs from protocol manifest")
    token = option.get("text")
    token_id = option.get("id")
    if not isinstance(token, str) or tokenizer.encode(token, add_special_tokens=False) != [
        token_id
    ]:
        raise ValueError("Intent explicit speak token differs from tokenizer")
    return token


ORIGIN_FILTER = "drop_unsupported_controls_v1"


def origin_mode(tokenizer):
    return getattr(tokenizer, "reciprocal_origin_intent", False) is True


def filter_controls(tokenizer, text):
    """Remove unsupported protocol spellings before ordinary BPE tokenization."""
    if not origin_mode(tokenizer):
        return text
    import re

    vocab = tokenizer.reciprocal_vocab
    return re.sub(
        r"<(?:wait|blank|chunk_[0-9]+|emo_[A-Za-z_]+)>",
        lambda m: m[0] if m[0] in vocab else "",
        text,
    )


def encode_protocol(tokenizer, text):
    return tokenizer.encode(filter_controls(tokenizer, text), add_special_tokens=False)


def origin_identity(args, tokenizer):
    import hashlib
    from .codec_spec import mode_from_args, spec

    root = Path(args.intent).resolve()
    geometry = spec(mode_from_args(args))
    chunk = {4: "<chunk_160>", 5: "<chunk_208>", 17: "<chunk_708>"}[geometry.frames]
    tokenizer.reciprocal_origin_intent = True
    tokenizer.reciprocal_chunk_token = chunk
    tokenizer.reciprocal_protocol_manifest = None
    tokenizer.reciprocal_vocab = tokenizer.get_vocab()
    tokenizer.reciprocal_vocab_ids = frozenset(tokenizer.reciprocal_vocab.values())
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.json",
            "chat_template.jinja",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        )
        if (root / name).is_file()
    }
    args.intent_identity = dict(
        mode="origin",
        control_filter=ORIGIN_FILTER,
        base_source=str(root),
        base_files=files,
        frames_per_block=geometry.frames,
        fps=24,
        intent_state_layer=50,
        removed_controls=[
            t for t in dict.fromkeys([*SPECIALS, chunk]) if t not in tokenizer.reciprocal_vocab
        ],
    )
    return args.intent_identity


def prepare_tokenizer(args):
    """CPU identity preflight shared by extraction and online loading."""
    from transformers import AutoTokenizer

    if getattr(args, "origin_intent", False):
        args.intent_adapter = ""
        tokenizer = AutoTokenizer.from_pretrained(args.intent, local_files_only=True)
        origin_identity(args, tokenizer)
        return tokenizer, {}
    adapter = Path(args.intent_adapter)
    config = json.loads((adapter / "adapter_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    chunk = resolve_chunk_token(args, tokenizer, config)
    return tokenizer, verify_tokens(tokenizer, config, chunk)


def load(args, *, dtype):
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    tokenizer, ids = prepare_tokenizer(args)
    processor = AutoProcessor.from_pretrained(args.intent, local_files_only=True)
    processor.tokenizer = tokenizer
    processor.video_processor.size = {
        "shortest_edge": 1024,
        "longest_edge": args.intent_max_pixels,
    }
    # Load all 64 layers before the adapter: truncating at layer 50 at load time
    # would silently discard part of the saved adapter. The hook selects intent_states.
    base = AutoModelForImageTextToText.from_pretrained(
        args.intent,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation="flash_attention_2",
        device_map={"": getattr(args, "device", "cuda:0")},
    )
    if getattr(args, "origin_intent", False):
        if max(tokenizer.reciprocal_vocab_ids) >= base.get_input_embeddings().num_embeddings:
            raise ValueError("Original tokenizer exceeds base Intent embeddings")
        return (
            processor,
            base.eval().requires_grad_(False),
            dict(protocol=PROTOCOL, special_ids=ids, layers=64, **args.intent_identity),
        )
    from peft import PeftModel

    adapter = Path(args.intent_adapter)
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model = (
        PeftModel.from_pretrained(
            base,
            adapter,
            is_trainable=False,
            autocast_adapter_dtype=getattr(args, "intent_rollout", False),
        )
        .eval()
        .requires_grad_(False)
    )
    return (
        processor,
        model,
        dict(
            protocol=PROTOCOL,
            special_ids=ids,
            layers=64,
            intent_state_layer=50,
            **args.intent_identity,
        ),
    )


def language_layers(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.language_model.layers
