"""Frame-v5 Intent: User video, Assistant text, and three selected h50 spans."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import intent
from . import media_clock as clock

PROTOCOL = "duplex_frame_chatml_v5"
TEMPLATE_VERSION = "assistant_description_plan_v1"
INTRO = (
    "<chunk_160> You are the assistant in a real-time full-duplex video call.\n"
    "The assistant's visual appearance and scene are described as follows:\n\""
)
INSTRUCTION = (
    '"\nPredict what the assistant should say in the next chunk. Output <wait> '
    "if the assistant is waiting or continuing an unfinished utterance.\n"
)
SPEECH_CUE = "In this chunk, the assistant should say:"
DESCRIPTION_FIELDS = ("appearance", "environment", "initial_pose")


def describe(record):
    """Read existing static metadata, never dialogue or future action descriptions."""
    path = Path(record["canonical"]["source"]["metadata"]).resolve()
    fields = record.get("assistant_description_override")
    if fields is None:
        script = json.loads(path.read_bytes())["script"]
        if script.get("sample_id", record["sample_id"]) != record["sample_id"]:
            raise ValueError("Assistant metadata sample identity differs")
        fields = script["assistant"]
    if any(
        not isinstance(fields.get(k), str) or not fields[k].strip()
        for k in DESCRIPTION_FIELDS
    ):
        raise ValueError(
            f"Missing Assistant appearance/environment/initial_pose: {path}"
        )
    description = " ".join(fields[k].strip() for k in DESCRIPTION_FIELDS)
    if any(t in description for t in ("<|", "<wait>", "<blank>", "<chunk_")):
        raise ValueError("Assistant description contains protocol controls")
    return dict(
        text=description,
        fields={k: fields[k] for k in DESCRIPTION_FIELDS},
        source=(
            dict(kind="config_first_frame_description", fields=dict(fields))
            if record.get("assistant_description_override") is not None
            else dict(path=str(path))
        ),
    )


def serialize(
    record,
    tokenizer,
    visuals,
    mode="planning",
    *,
    stop_block=None,
    assistant_bodies=None,
    generation=False,
    context_mode="full_gt",
):
    """Serialize causal Intent input and select exactly the Realization conditioning rows.

    Selected: static Assistant description, shared speech cue, Assistant body
    for each block. User video/text remains Intent context, not a selected h50
    span. ChatML delimiters, timestamps and the virtual initial wait are omitted
    from the selected matrix. Each body stops before its closing delimiter.
    """
    if context_mode not in ("full_gt", "closed_loop"):
        raise ValueError("Unknown Intent context mode")
    blocks, canonical = record["blocks"], record["canonical"]
    stop = len(blocks) - 1 if stop_block is None else stop_block
    if not 0 <= stop < len(blocks):
        raise ValueError("Invalid current block")
    description = describe(record)
    (
        input_token_ids,
        selected_token_mask,
        text_groups,
        selected_spans,
        grids,
        patches,
        controls,
        h50_ends,
        messages,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    selected_rows = 0
    input_spans = []

    def encoded(value):
        return intent.encode_protocol(tokenizer, value)

    def append(
        part, section=None, block=-1, *, source_section=None, role=None, visual=None
    ):
        nonlocal selected_rows
        part = list(part)
        start = len(input_token_ids)
        input_token_ids.extend(part)
        if part:
            span = dict(
                start=start,
                end=len(input_token_ids),
                block=block,
                role=role or ("assistant" if block >= 0 else "system"),
                section=source_section or section,
            )
            if span["section"] is None:
                raise ValueError("Missing Intent input provenance")
            if visual is not None:
                span.update(visual)
            input_spans.append(span)
        selected_token_mask.extend([section is not None] * len(part))
        if section is not None and (
            part or (section == "assistant_plan" and intent.origin_mode(tokenizer))
        ):
            text_groups.append(
                dict(
                    start=selected_rows,
                    end=selected_rows + len(part),
                    role="assistant" if block >= 0 else "system",
                    block=block,
                    phase=block,
                    section=section,
                )
            )
            selected_spans.append(
                dict(
                    start=start, end=len(input_token_ids), section=section, block=block
                )
            )
            selected_rows += len(part)

    # Explicit fragment boundaries make the selected token spans unambiguous.
    append(
        encoded(
            "<|im_start|>system\n"
            + INTRO.replace(
                "<chunk_160>",
                getattr(tokenizer, "duplex_chunk_token", "<chunk_160>"),
                1,
            )
        ),
        source_section="system_control",
    )
    append(encoded(description["text"]), "assistant_description")
    append(encoded(INSTRUCTION), source_section="system_instruction")
    append(encoded(SPEECH_CUE), "speech_cue")
    append(
        encoded("<|im_end|>\n<|im_start|>assistant\n<wait><|im_end|>\n"),
        source_section="virtual_initial_wait",
        role="assistant",
    )
    messages.extend(
        [
            dict(
                role="system",
                content=INTRO + description["text"] + INSTRUCTION + SPEECH_CUE,
            ),
            dict(role="assistant", content="<wait>"),
        ]
    )
    for k, b in enumerate(blocks[: stop + 1]):
        include_user = context_mode == "full_gt" or k < stop
        if include_user:
            spec = visuals["user", k]
            grids.append(spec["grid"])
            patches.append(spec["patches"])
            video = visual_payload(b, spec)
            user = body_ids(canonical, b, "user", mode, tokenizer)
            # Keep the original encoded fragments intact; annotations never retokenize.
            prefix = encoded("<|im_start|>user\n" + video)
            pad = tokenizer.convert_tokens_to_ids("<|video_pad|>")
            offset = len(input_token_ids)
            video_rows = [offset + i for i, token in enumerate(prefix) if token == pad]
            append(
                prefix,
                block=k,
                source_section="user_video_context",
                role="user",
                visual=dict(
                    frame_start=b["frame_start"],
                    frame_stop=b["frame_stop"],
                    start_ms=b["start_ms"],
                    end_ms=b["end_ms"],
                    video_grid_thw=spec["grid"],
                    video_token_rows=video_rows,
                ),
            )
            append(user, block=k, source_section="user_body", role="user")
            append(
                encoded("<|im_end|>\n<|im_start|>assistant\n"),
                block=k,
                source_section="user_to_assistant_separator",
                role="protocol",
            )
        else:
            append(
                encoded("<|im_start|>assistant\n"),
                block=k,
                source_section="assistant_generation_start",
                role="protocol",
            )
        body = (
            []
            if generation and k == stop
            else (
                list(assistant_bodies[k])
                if assistant_bodies is not None and k in assistant_bodies
                else body_ids(canonical, b, "assistant", mode, tokenizer)
            )
        )
        if (
            not body
            and not (generation and k == stop)
            and not intent.origin_mode(tokenizer)
        ):
            raise ValueError("Empty Assistant body")
        append(body, "assistant_plan", k)
        h50_ends.append(selected_rows)
        controls.append(
            tokenizer.decode(
                body, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )
        if include_user:
            messages.append(
                dict(
                    role="user",
                    content=video + tokenizer.decode(user, skip_special_tokens=False),
                )
            )
        messages.append(dict(role="assistant", content=controls[-1], open=k == stop))
        if k < stop:
            append(
                encoded("<|im_end|>\n"),
                block=k,
                source_section="assistant_message_end",
                role="protocol",
            )
    return dict(
        input_ids=input_token_ids,
        input_spans=input_spans,
        keep=selected_token_mask,
        text_groups=text_groups,
        h50_ends=h50_ends,
        controls=controls,
        messages=messages,
        video_grid_thw=grids,
        patches=patches,
        protocol=PROTOCOL,
        selected_spans=selected_spans,
        assistant_description=description,
        template_version=TEMPLATE_VERSION,
    )


def extract_h50(text, model, *, capture=None):
    """Layer index 49 output -> selected rows [T, 5120] -> Realization prompt_embeds.

    Keep all 64 Intent layers loaded for adapter compatibility. Capture the output
    of layer 50 before the final language-model norm, and preserve row order.
    data/dataloader_realization_sequence.py maps these rows into the joint sequence; the upstream Realization
    applies condition_proj, then its two masked token-refiner layers.
    """
    import torch

    layer_outputs = []
    hook = intent.language_layers(model)[49].register_forward_hook(
        lambda m, a, out: layer_outputs.append(
            (out[0] if isinstance(out, tuple) else out)[0].detach()
        )
    )
    try:
        with torch.no_grad():
            model(
                **{k: v.to(model.device) for k, v in model_batch(text, model).items()},
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        hook.remove()
    if len(layer_outputs) != 1:
        raise RuntimeError("Expected exactly one layer-50 result")
    selected_token_mask = torch.tensor(
        text["keep"], device=model.device, dtype=torch.bool
    )
    h50 = layer_outputs[0][selected_token_mask].cpu().contiguous()
    if h50.shape[1] != 5120 or not torch.isfinite(h50).all():
        raise ValueError("Invalid h50")
    result = dict(
        h50=h50,
        h50_ends=text["h50_ends"],
        controls=text["controls"],
        text_groups=text["text_groups"],
        protocol=PROTOCOL,
        input_ids=torch.tensor(text["input_ids"]),
        input_keep=selected_token_mask.cpu(),
        text_ids=torch.tensor(text["input_ids"])[selected_token_mask.cpu()],
        intent_tokens=len(text["input_ids"]),
    )

    result.update(
        selected_spans=text["selected_spans"],
        assistant_description=text["assistant_description"],
        template_version=TEMPLATE_VERSION,
    )
    if capture is not None:
        result["h50_capture"] = capture(layer_outputs[0], text)
    return result


def encode_text(record, videos, processor, model, args):
    visuals = {
        ("user", k): visual_patch(
            processor, videos["user"]["frames"][b["frame_start"] : b["frame_stop"]]
        )
        for k, b in enumerate(record["blocks"])
    }
    text = serialize(record, processor.tokenizer, visuals, args.text_time_mode)
    if len(text["input_ids"]) > args.max_intent_tokens:
        raise ValueError("Full Intent sequence exceeds token budget; no truncation")
    return extract_h50(text, model)


def speech_ids(canonical, block, role, mode):
    key = "start_ms" if role == "assistant" and mode == "planning" else "commit_ms"
    return [
        int(t["token_id"])
        for t in canonical["text_tokens"]
        if t["role"] == role and clock.contains(block, t[key])
    ]


def body_ids(canonical, block, role, mode, tokenizer):
    # Only the clock changes. Keep natural IDs and the original emotion placement.
    input_token_ids = speech_ids(canonical, block, role, mode)
    if intent.origin_mode(tokenizer):
        valid = tokenizer.duplex_vocab_ids
        if any(token not in valid for token in input_token_ids):
            raise ValueError(
                "Canonical natural token ID is absent from original Intent vocabulary"
            )
    if not input_token_ids:
        input_token_ids = intent.encode_protocol(
            tokenizer, "<blank>" if role == "user" else "<wait>"
        )
    tag = emotion(canonical, block, role, mode)
    emotion_ids = intent.encode_protocol(tokenizer, tag)
    return (
        input_token_ids + emotion_ids
        if role == "user"
        else emotion_ids + input_token_ids
    )


def visual_patch(processor, rgb):
    """One independent codec block; temporal padding repeats only its last frame."""

    import numpy as np

    frames = len(rgb)
    if frames not in (4, 5, 17):
        raise ValueError(
            "Intent visual observation must contain a complete codec block"
        )
    if frames % 2:
        rgb = np.concatenate([rgb, rgb[-1:]], axis=0)
    result = processor.video_processor(
        videos=[rgb], do_sample_frames=False, return_tensors="pt"
    )
    grid = result["video_grid_thw"][0].tolist()
    if len(grid) != 3 or grid[0] != (frames + 1) // 2:
        raise ValueError(f"Unexpected Intent temporal patch count, got {grid}")
    return dict(
        patches=result["pixel_values_videos"].cpu().contiguous(),
        grid=grid,
        merge=int(processor.video_processor.merge_size),
    )


def visual_payload(block, spec):
    t, h, w = spec["grid"]
    count = h * w // spec["merge"] ** 2
    if t != (block["frame_stop"] - block["frame_start"] + 1) // 2 or count < 1:
        raise ValueError("Invalid codec-block visual grid")
    return "".join(
        f"<{min(block['frame_start'] + 2 * j + 1, block['frame_stop'] - 1) / clock.FPS:.1f} seconds>"
        + "<|vision_start|>"
        + "<|video_pad|>" * count
        + "<|vision_end|>"
        for j in range(t)
    )


def model_batch(text, model=None):
    import torch

    input_token_ids = torch.tensor([text["input_ids"]], dtype=torch.long)
    batch = dict(
        input_ids=input_token_ids, attention_mask=torch.ones_like(input_token_ids)
    )
    if text["patches"]:
        batch.update(
            pixel_values_videos=torch.cat(text["patches"]),
            video_grid_thw=torch.tensor(text["video_grid_thw"], dtype=torch.long),
        )
    # Transformers 5 uses explicit modality labels for M-RoPE. Inspect the
    # unwrapped model API, rather than passing unsupported generation kwargs
    # into older extraction environments. No token is re-tokenized here.
    if model is not None:
        import inspect

        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        core = getattr(base, "model", base)
        rope = getattr(core, "get_rope_index", None)
        if (
            rope is not None
            and "mm_token_type_ids" in inspect.signature(rope).parameters
        ):
            modalities = torch.zeros_like(input_token_ids)
            for span in text.get("input_spans", ()):
                rows = span.get("video_token_rows", ())
                if rows:
                    modalities[0, list(rows)] = 2
            batch["mm_token_type_ids"] = modalities
    return batch


def emotion(canonical, block, role, mode):
    meta = canonical.get("emotions", {})
    events = meta.get("attachments", meta.get("turns", []))
    result = []
    for item in events:
        if item.get("role") != role or not item.get("usable_for_training"):
            continue
        token = item.get("user_token" if role == "user" else "assistant_token")
        if not token:
            continue
        timestamp = item.get(
            "commit_ms" if role == "user" else "assistant_target_commit_ms"
        )
        if role == "assistant" and mode == "planning":
            # Move the existing emotion attachment with the token(s) that the
            # original serializer put immediately after it, not to every block.
            candidates = [
                t["start_ms"]
                for t in canonical["text_tokens"]
                if t["role"] == role
                and t.get("turn_id") == item.get("turn_id")
                and t["commit_ms"] == timestamp
            ]
            timestamp = min(candidates) if candidates else timestamp
        if timestamp is not None and block["start_ms"] <= timestamp < block["end_ms"]:
            result.append(token)
    return "".join(result)


def combine_block_features(prefixes):
    """Assemble joint Realization text from each block's own closed-loop Intent prefix."""
    import torch

    result = dict(prefixes[-1])
    result.update(
        h50=[], text_ids=[], text_groups=[], h50_ends=[], controls=[], selected_spans=[]
    )
    input_ids, input_keep, cursor, offset, captures, capture_rows = [], [], 0, 0, [], []
    for block, item in enumerate(prefixes):
        keep = torch.zeros_like(item["input_keep"])
        input_ids.append(item["input_ids"])
        selected = (item["input_keep"].nonzero().flatten()).tolist()
        for group in item["text_groups"]:
            if group["block"] != block and not (block == 0 and group["block"] == -1):
                continue
            lo, hi = group["start"], group["end"]
            result["h50"].append(item["h50"][lo:hi])
            result["text_ids"].append(item["text_ids"][lo:hi])
            result["text_groups"].append(
                dict(group, start=cursor, end=cursor + hi - lo)
            )
            span = next(
                s
                for s in item["selected_spans"]
                if s["block"] == group["block"] and s["section"] == group["section"]
            )
            result["selected_spans"].append(
                dict(span, start=offset + span["start"], end=offset + span["end"])
            )
            rows = selected[lo:hi]
            keep[rows] = True
            capture_rows.extend(offset + r for r in rows)
            cursor += hi - lo
        result["h50_ends"].append(cursor)
        result["controls"].append(item["controls"][block])
        input_keep.append(keep)
        if item.get("h50_capture"):
            captures.append(item["h50_capture"])
        offset += len(item["input_ids"])
    result.update(
        h50=torch.cat(result["h50"]),
        text_ids=torch.cat(result["text_ids"]),
        input_ids=torch.cat(input_ids),
        input_keep=torch.cat(input_keep),
        intent_tokens=offset,
    )
    if captures:
        if len(captures) != len(prefixes):
            raise ValueError("Incomplete block-prefix h50 captures")
        identity = hashlib.sha256(
            json.dumps(captures, sort_keys=True).encode()
        ).hexdigest()
        result["h50_capture"] = dict(
            id="capture-composite-" + identity,
            sha256=identity,
            parts=captures,
            rows=offset,
            selected_rows=capture_rows,
        )
    return result
