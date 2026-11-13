"""Reference incremental Intent rollout; h50 comes only from executed NTP forwards."""

import hashlib
import inspect
import json
from copy import deepcopy

from ..modules.intent_rollout_inputs import (
    H50_STRATEGY,
    LAYOUT,
    POLICY,
    assert_rollout_protocol,
    parse_generation,
)


def metadata():
    return dict(
        conditioning_policy=POLICY,
        h50_strategy=H50_STRATEGY,
        conditioning_layout=LAYOUT,
        assistant_text_source="intent_generated_history",
        asr_status="removed",
    )


class Rollout:
    def __init__(self, model, inputs, args, capture=None):
        import torch

        assert_rollout_protocol(inputs.tokenizer, inputs)
        self.model, self.inputs, self.args, self.capture = model, inputs, args, capture
        self.tokenizer = inputs.tokenizer
        self.device = torch.device(getattr(args, "device", str(model.device)))
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.core = base.model
        self.layers = self.core.language_model.layers
        self.wait = self.tokenizer.convert_tokens_to_ids("<wait>")
        self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.speak = (
            self.tokenizer.convert_tokens_to_ids("<speak>")
            if inputs.explicit_speak
            else None
        )
        self.messages = [dict(role="system", content=inputs.system)]
        self.next_block = 0
        self.context_exhausted = False
        self.cache = None
        self.cached_ids = []
        self.rope_position = 0
        self.logits = None
        self.ready_block = None
        self.force_rebuild = False
        self.segments = []
        self.segment_hidden = []
        self.captures = []

    def prompt(self, messages, generation=True):
        import torch

        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation,
            return_tensors="pt",
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return ids[0].tolist() if isinstance(ids, torch.Tensor) else list(ids)

    def forward(
        self,
        ids,
        indices=(),
        grids=(),
        *,
        reset=False,
        selected=(),
        role="protocol",
        replay=False
    ):
        import torch

        if not ids:
            raise ValueError("Empty incremental Intent segment")
        if reset:
            self.cache = None
            self.cached_ids = []
            self.rope_position = 0
        if len(self.cached_ids) + len(ids) > self.args.max_intent_tokens:
            raise ValueError("Intent context budget exceeded; no truncation")
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        grid = (
            torch.tensor(grids, dtype=torch.long, device=self.device) if grids else None
        )
        rope_args = dict(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            video_grid_thw=grid,
        )
        if (
            "mm_token_type_ids"
            in inspect.signature(self.core.get_rope_index).parameters
        ):
            rope_args["mm_token_type_ids"] = (
                tokens == self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            ).long() * 2
        position, _ = self.core.get_rope_index(**rope_args)
        position = position.to(self.device) + self.rope_position
        next_position = int(position.max().item()) + 1
        past = int(self.cache.get_seq_length()) if self.cache is not None else 0
        batch = dict(
            input_ids=tokens,
            attention_mask=torch.ones(
                1, past + len(ids), dtype=torch.long, device=self.device
            ),
            position_ids=position,
            cache_position=torch.arange(past, past + len(ids), device=self.device),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        pixels = self.inputs.pixels(list(indices))
        if pixels is not None:
            batch.update(
                pixel_values_videos=pixels.to(self.device), video_grid_thw=grid
            )
        hidden = []

        def hook(module, arguments, output):
            value = output[0] if isinstance(output, tuple) else output
            if not replay:
                hidden.append(value[0].detach().cpu().contiguous())

        handle = self.layers[49].register_forward_hook(hook)
        try:
            with torch.no_grad():
                output = self.model(**batch)
        finally:
            handle.remove()
        if output.past_key_values.get_seq_length() != past + len(ids):
            raise ValueError("Intent incremental KV length differs")
        self.cache, self.logits, self.rope_position = (
            output.past_key_values,
            output.logits[:, -1, :],
            next_position,
        )
        self.cached_ids.extend(ids)
        segment = dict(
            ids=list(ids),
            indices=list(indices),
            grids=list(grids),
            reset=reset,
            selected=list(selected),
            role=role,
            rope_stop=next_position,
            cache_stop=past + len(ids),
        )
        if not replay:
            if (
                len(hidden) != 1
                or hidden[0].shape != (len(ids), 5120)
                or not torch.isfinite(hidden[0]).all()
            ):
                raise ValueError("Invalid NTP layer-50 output")
            self.segments.append(segment)
            self.segment_hidden.append(hidden[0])
            if self.capture:
                keep = [i in selected for i in range(len(ids))]
                self.captures.append(
                    self.capture(
                        hidden[0],
                        dict(
                            input_ids=ids,
                            keep=keep,
                            input_spans=[
                                dict(
                                    start=0,
                                    end=len(ids),
                                    section="ntp_" + role,
                                    execution_role=role,
                                    video_patch_indices=list(indices),
                                    video_token_rows=[
                                        i
                                        for i, t in enumerate(ids)
                                        if t
                                        == self.tokenizer.convert_tokens_to_ids(
                                            "<|video_pad|>"
                                        )
                                    ],
                                )
                            ],
                            protocol="duplex_frame_chatml_v5",
                            video_grid_thw=list(grids),
                        ),
                    )
                )
        if hidden:
            self.hidden_dtype = hidden[0].dtype
        return hidden[0] if hidden else None

    def _bootstrap_rows(self, messages, generation):
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation
        )
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        if encoded["input_ids"] != self.prompt(messages, generation):
            raise ValueError("Intent template token/offset mapping differs")
        start = (
            rendered.index(self.inputs.system)
            + self.inputs.system.index('"' + self.inputs.description + '"')
            + 1
        )
        stop = start + len(self.inputs.description)
        desc = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < stop and b > start
        ]
        marker = "<|im_start|>assistant\n<wait>"
        start = rendered.index(marker) + len(marker) - len("<wait>")
        bootstrap = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < start + len("<wait>") and b > start
        ]
        if (
            not desc
            or len(bootstrap) != 1
            or encoded["input_ids"][bootstrap[0]] != self.wait
        ):
            raise ValueError("Missing exact description/bootstrap h50 rows")
        return desc, bootstrap

    def _description_rows(self, messages, generation=True):
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation
        )
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        if encoded["input_ids"] != self.prompt(messages, generation):
            raise ValueError("Intent template token/offset mapping differs")
        start = (
            rendered.index(self.inputs.system)
            + self.inputs.system.index('"' + self.inputs.description + '"')
            + 1
        )
        stop = start + len(self.inputs.description)
        rows = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < stop and b > start
        ]
        if not rows:
            raise ValueError("Missing exact Assistant description h50 rows")
        return rows

    def _visuals(self, stop):
        indices, grids = [], []
        for j in range(stop):
            _, part, grid = self.inputs.visual(j)
            indices.extend(part)
            if grid:
                grids.append(grid)
        return indices, grids

    def _prepare(self, k):
        """Append only U(k-1); a cropped continuation never invents U(-1)."""
        messages = list(self.messages)
        if k:
            content, _, _ = self.inputs.visual(k - 1)
            messages.append(dict(role="user", content=content))
        ids = self.prompt(messages)
        description = []
        if self.context_exhausted:
            return False, description
        if len(ids) > self.args.max_intent_tokens:
            # Preserve the last committed prefix instead of evicting history.
            if not self.cached_ids:
                base = [dict(role="system", content=self.inputs.system)]
                base_ids = self.prompt(base, False)
                if len(base_ids) > self.args.max_intent_tokens:
                    raise ValueError(
                        "Intent mandatory system prefix exceeds the configured context budget"
                    )
                rows = self._description_rows(base, False)
                self.forward(base_ids, selected=rows, role="prefix_only")
                description = [(0, rows)]
            self.context_exhausted = True
            return False, description
        self.messages = messages
        if self.ready_block == k and self.cached_ids == ids:
            self.ready_block = None
        else:
            reset = self.force_rebuild or ids[: len(self.cached_ids)] != self.cached_ids
            self.force_rebuild = False
            indices, grids = self._visuals(k) if reset else ([], [])
            if not reset and k:
                _, indices, grid = self.inputs.visual(k - 1)
                grids = [grid] if grid else []
            suffix = ids if reset else ids[len(self.cached_ids) :]
            rows = self._description_rows(messages) if not self.cached_ids else []
            if suffix:
                self.forward(
                    suffix, indices, grids, reset=reset, selected=rows, role="prompt"
                )
            if rows:
                description = [(len(self.segments) - 1, rows)]
        self._current_prompt = ids
        return True, description

    def _feature(self, block, selections):
        import torch

        from ..modules.intent_conditioning import PROTOCOL

        ids = [x for segment in self.segments for x in segment["ids"]]
        offset = [0]
        for segment in self.segments:
            offset.append(offset[-1] + len(segment["ids"]))
        groups, spans, pieces, text_ids, keep = [], [], [], [], [False] * len(ids)
        cursor = 0
        for section, owner, selected in selections:
            for segment, rows in selected:
                if not rows:
                    continue
                values = self.segment_hidden[segment][rows]
                pieces.append(values)
                text_ids.extend(self.segments[segment]["ids"][i] for i in rows)
                absolute = [offset[segment] + i for i in rows]
                for i in absolute:
                    keep[i] = True
                spans.append(
                    dict(
                        start=absolute[0],
                        end=absolute[-1] + 1,
                        section=section,
                        block=owner,
                    )
                )
            length = sum(len(rows) for _, rows in selected)
            groups.append(
                dict(
                    start=cursor,
                    end=cursor + length,
                    section=section,
                    block=owner,
                    phase=owner,
                    role="system" if owner < 0 else "assistant",
                )
            )
            cursor += length
        hidden = (
            torch.cat(pieces)
            if pieces
            else torch.empty(
                (0, 5120), dtype=getattr(self, "hidden_dtype", torch.float32)
            )
        )
        feature = dict(
            h50=hidden,
            text_ids=torch.tensor(text_ids, dtype=torch.long),
            input_ids=torch.tensor(ids, dtype=torch.long),
            input_keep=torch.tensor(keep, dtype=torch.bool),
            text_groups=groups,
            selected_spans=spans,
            h50_ends=[cursor],
            controls=[],
            protocol=PROTOCOL,
            conditioning_layout=LAYOUT,
            h50_strategy=H50_STRATEGY,
            template_version=POLICY,
            intent_tokens=len(ids),
            assistant_description=dict(text=self.inputs.description),
        )
        if self.captures:
            feature["h50_capture"] = composite_capture(
                self.captures, [i for i, v in enumerate(keep) if v]
            )
        self.segments, self.segment_hidden, self.captures = [], [], []
        return feature

    def step(self, k, *, body=None, prefix=(), bootstrap=False, lookahead=False):
        """One shared protocol step: known GT, a forced prefix, or greedy NTP."""
        if k != self.next_block or not 0 <= k < len(self.inputs.units):
            raise ValueError("Intent steps must cover the selected window in order")
        if body is not None and prefix:
            raise ValueError("Choose a known body or a continuation prefix")
        description, selected = [], []
        raw, committed = [], []
        ended = truncated = context_limited = prefix_only = False
        action, valid = "prefix_only", None
        generated_limit = 0
        reason = "forced_history" if body is not None else "reference_action_parser"
        forced = list(body if body is not None else prefix)
        if bootstrap:
            if (
                k
                or body is not None
                or prefix
                or self.messages != [dict(role="system", content=self.inputs.system)]
            ):
                raise ValueError("Invalid whole-conversation Intent bootstrap")
            self.messages.append(dict(role="assistant", content="<wait>"))
            messages, indices, grids = list(self.messages), [], []
            if lookahead:
                content, part, grid = self.inputs.visual(0)
                candidate = [*messages, dict(role="user", content=content)]
                if len(self.prompt(candidate)) <= self.args.max_intent_tokens:
                    messages, indices, grids = candidate, part, [grid] if grid else []
                else:
                    lookahead = False
            desc, boot = self._bootstrap_rows(messages, lookahead)
            ids = self.prompt(messages, lookahead)
            if len(ids) > self.args.max_intent_tokens:
                raise ValueError(
                    "Intent mandatory system/bootstrap prefix exceeds the configured context budget"
                )
            self.forward(ids, indices, grids, selected=desc + boot, role="bootstrap")
            description, selected = [(0, desc)], [(0, boot)]
            committed, action, valid = [self.wait], "wait", True
            self.ready_block = 1 if lookahead else None
        else:
            prepared, description = self._prepare(k)
            if not prepared:
                prefix_only = context_limited = True
            else:
                available = self.args.max_intent_tokens - len(self.cached_ids)
                known = forced[: max(0, available)]
                if known:
                    rows = list(range(len(known)))
                    self.forward(
                        known,
                        selected=rows,
                        role="assistant_gt" if body is not None else "assistant_prefix",
                    )
                    selected.append((len(self.segments) - 1, rows))
                committed = list(known)
                if len(known) != len(forced):
                    prefix_only = context_limited = self.context_exhausted = True
                elif body is not None:
                    _, _, action, valid = parse_generation(
                        committed, self.wait, self.speak
                    )
                else:
                    available = max(
                        0, self.args.max_intent_tokens - len(self.cached_ids)
                    )
                    generated_limit = max(0, min(self.args.max_new_tokens, available))
                    context_limited = generated_limit < self.args.max_new_tokens
                    if not generated_limit:
                        prefix_only = True
                    else:
                        for _ in range(generated_limit):
                            token = int(self.logits.argmax(-1).item())
                            if token == self.eos:
                                ended = True
                                break
                            raw.append(token)
                            self.forward([token], selected=[0], role="assistant")
                            selected.append((len(self.segments) - 1, [0]))
                        truncated = not ended
                        combined = committed + raw
                        normalized, _, action, valid = parse_generation(
                            combined, self.wait, self.speak
                        )
                        if normalized == combined[: len(normalized)]:
                            selected = self._take_rows(selected, len(normalized))
                        elif normalized != combined:
                            # Canonical fallback has its own executed hidden state.
                            if combined:
                                indices, grids = self._visuals(k)
                                self.forward(
                                    self._current_prompt,
                                    indices,
                                    grids,
                                    reset=True,
                                    role="fallback_prompt",
                                )
                            selected = []
                            for token in normalized:
                                self.forward(
                                    [token], selected=[0], role="assistant_fallback"
                                )
                                selected.append((len(self.segments) - 1, [0]))
                        self.force_rebuild = normalized != combined and bool(combined)
                        committed = normalized
            if prefix_only:
                reason = "context_budget_prefix_only"
                action, valid = "prefix_only", None
            text = self.tokenizer.decode(committed, skip_special_tokens=False)
            if prepared:
                self.messages.append(dict(role="assistant", content=text))
            self.ready_block = None
        text = self.tokenizer.decode(committed, skip_special_tokens=False)
        selections = []
        if self.next_block == 0:
            selections.append(("assistant_description", -1, description))
        selections.append(("assistant_plan", k, selected))
        segments = deepcopy(self.segments)
        feature = self._feature(k, selections)
        plan = dict(
            mode="closed_loop",
            **metadata(),
            block=k,
            body_ids=committed,
            raw_body_ids=forced + raw,
            generated_ids=raw,
            text=text,
            raw_text=self.tokenizer.decode(forced + raw, skip_special_tokens=False),
            action=action,
            action_valid=valid,
            controls=[t for t in committed if t in (self.wait, self.speak)],
            bootstrap=bootstrap,
            input_tokens=len(self.cached_ids),
            excluded_current_assistant_video=True,
            gt_prefix_ids=list(prefix),
            forced_body=body is not None,
            ended=ended,
            truncated=truncated,
            context_limited=context_limited,
            prefix_only=prefix_only,
            generated_limit=generated_limit,
            h50_rows=len(feature["h50"])
            - sum(
                g["end"] - g["start"] for g in feature["text_groups"] if g["block"] < 0
            ),
            normalization=dict(
                changed=committed != forced + raw if not bootstrap else False,
                reason=reason,
            )
        )
        item = dict(
            features=feature,
            plan=plan,
            forward_segments=segments,
            input_identity=self.inputs.identity,
            ready_block=self.ready_block,
            force_rebuild=self.force_rebuild,
            messages=deepcopy(self.messages),
            context_exhausted=self.context_exhausted,
        )
        self.next_block += 1
        return item

    @staticmethod
    def _take_rows(selected, length):
        result = []
        for segment, rows in selected:
            count = min(length, len(rows))
            if count:
                result.append((segment, rows[:count]))
                length -= count
        return result

    def run(self, count, saved=(), publish=None, progress=None):
        if not 0 < count <= len(self.inputs.units) or len(saved) > count:
            raise ValueError("Invalid reference rollout coverage")
        emit = progress or (lambda **kw: None)
        records = list(saved)
        for k, item in enumerate(saved):
            if item["input_identity"] != self.inputs.identity:
                raise ValueError("Saved Intent rollout inputs differ")
            for segment in item["forward_segments"]:
                self.forward(
                    segment["ids"],
                    segment["indices"],
                    segment["grids"],
                    reset=segment["reset"],
                    replay=True,
                )
                if (
                    self.rope_position != segment["rope_stop"]
                    or len(self.cached_ids) != segment["cache_stop"]
                ):
                    raise ValueError("Saved Intent cache/position replay differs")
            self.messages = deepcopy(item["messages"])
            self.context_exhausted = item["context_exhausted"]
            self.ready_block = item["ready_block"]
            self.force_rebuild = item.get("force_rebuild", False)
            self.hidden_dtype = item["features"]["h50"].dtype
            self.next_block = k + 1
        if saved and saved[-1].get("rng"):
            from ....components.checkpoint_realization import restore_rng

            restore_rng(saved[-1]["rng"])
        for k in range(len(saved), count):
            item = self.step(k, bootstrap=k == 0, lookahead=count > 1)
            if publish:
                publish(k, item)
            records.append(item)
            emit(
                phase="conditions",
                completed_conditions=k + 1,
                total_conditions=count,
                input_tokens=len(self.cached_ids),
                prefix_only=item["plan"]["prefix_only"],
            )
        return dict(features=assemble(records), plans=[p["plan"] for p in records])

    def close(self):
        self.cache = None
        self.logits = None
        self.segments = []
        self.segment_hidden = []


def composite_capture(parts, rows):
    identity = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return dict(
        id="capture-composite-" + identity,
        sha256=identity,
        parts=parts,
        rows=sum(p["rows"] for p in parts),
        selected_rows=rows,
        strategy=H50_STRATEGY,
    )


def assemble(records):
    import torch

    features = [r["features"] for r in records]
    value = dict(features[0])
    groups = []
    spans = []
    cursor = offset = 0
    ends = []
    captures = []
    for f in features:
        groups.extend(
            dict(g, start=g["start"] + cursor, end=g["end"] + cursor)
            for g in f["text_groups"]
        )
        spans.extend(
            dict(s, start=s["start"] + offset, end=s["end"] + offset)
            for s in f["selected_spans"]
        )
        cursor += len(f["h50"])
        offset += len(f["input_ids"])
        ends.append(cursor)
        if f.get("h50_capture"):
            captures.append(f["h50_capture"])
    value.update(
        {
            name: torch.cat([f[name] for f in features])
            for name in ("h50", "text_ids", "input_ids", "input_keep")
        }
    )
    value.update(
        text_groups=groups,
        selected_spans=spans,
        h50_ends=ends,
        controls=[r["plan"]["text"] for r in records],
        intent_tokens=offset,
    )
    if captures:
        if len(captures) != sum(bool(len(f["input_ids"])) for f in features):
            raise ValueError("Incomplete incremental h50 captures")
        value["h50_capture"] = composite_capture(
            captures, value["input_keep"].nonzero().flatten().tolist()
        )
    return value
