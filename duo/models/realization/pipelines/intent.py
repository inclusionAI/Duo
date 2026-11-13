"""Intent NTP and intent_states extraction with a shared full-sequence oracle path."""

from contextlib import contextmanager

from ..modules import intent_conditioning
from . import intent_history as history


class Intent:
    def __init__(self, args, *, intent_dtype):
        self.args = args
        self.processor, self.model, self.loaded = intent_conditioning.intent.load(
            args, dtype=intent_dtype
        )
        self.tokenizer = self.processor.tokenizer
        self.device = getattr(args, "device", "cuda:0")
        self.model.to("cpu")
        self.emotions = {
            self.tokenizer.convert_tokens_to_ids(x)
            for x in self.loaded["special_ids"]
            if x.startswith("<emo_")
        }

    @contextmanager
    def on_gpu(self):
        import torch

        try:
            self.model.to(getattr(self, "device", "cuda:0"))
            yield
        finally:
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def extract(self, text):
        capture = getattr(self, "capture_intent_states", None)
        if capture is None:
            return intent_conditioning.extract_intent_states(text, self.model)
        return intent_conditioning.extract_intent_states(text, self.model, capture=capture)

    def generate_body(self, record, visuals, bodies, k, max_new_tokens):
        """One NTP turn. The caller owns GPU residency and subsequent intent_states extraction."""
        import torch

        history.require_history(bodies, k)
        prefix = intent_conditioning.serialize(
            record,
            self.tokenizer,
            visuals,
            stop_block=k,
            assistant_bodies=bodies,
            generation=True,
            context_mode="closed_loop",
        )
        if len(prefix["input_ids"]) + max_new_tokens > self.args.max_intent_tokens:
            raise ValueError("Intent context budget exceeded; no truncation")
        eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        allowed = self.emotions | {eos}
        if not intent_conditioning.intent.origin_mode(self.tokenizer):
            allowed.add(self.tokenizer.convert_tokens_to_ids("<wait>"))
        forbidden = sorted(set(self.tokenizer.all_special_ids) - allowed)
        inputs = {
            n: t.to(getattr(self, "device", "cuda:0"))
            for n, t in intent_conditioning.model_batch(prefix).items()
        }
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos,
                pad_token_id=self.tokenizer.pad_token_id,
                suppress_tokens=forbidden,
                use_cache=True,
            )
        generated = output[0, len(prefix["input_ids"]) :].tolist()
        if not generated or generated[-1] != eos:
            raise RuntimeError("Intent did not end its message within max_new_tokens")
        raw_body = generated[:-1]
        origin = intent_conditioning.intent.origin_mode(self.tokenizer)
        if (not raw_body and not origin) or any(x in forbidden or x == eos for x in raw_body):
            raise ValueError("Intent emitted an empty or invalid Assistant body")
        if origin:
            text = self.tokenizer.decode(
                raw_body, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            filtered = intent_conditioning.intent.filter_controls(self.tokenizer, text)
            body = (
                raw_body
                if text == filtered
                else self.tokenizer.encode(filtered, add_special_tokens=False)
            )
            return body, dict(
                generated_ids=generated,
                raw_body_ids=raw_body,
                normalization=dict(
                    version=intent_conditioning.intent.ORIGIN_FILTER,
                    changed=body != raw_body,
                    reason="original_intent_controls",
                ),
            )
        wait = self.tokenizer.convert_tokens_to_ids("<wait>")
        body, normalization = history.normalize_body(
            raw_body,
            self.emotions,
            wait,
            set(self.tokenizer.all_special_ids) - (self.emotions | {wait}),
        )
        return body, dict(
            generated_ids=generated, raw_body_ids=raw_body, normalization=normalization
        )

    def serialize(self, record, visuals, bodies, stop, context_mode="full_gt"):
        text = intent_conditioning.serialize(
            record,
            self.tokenizer,
            visuals,
            stop_block=stop,
            assistant_bodies=bodies,
            context_mode=context_mode,
        )
        if len(text["input_ids"]) > self.args.max_intent_tokens:
            raise ValueError("Intent intent_states budget exceeded; no truncation")
        return text

    def plan_record(self, mode, body, metadata, text, k):
        if metadata is None:
            metadata = dict(
                generated_ids=None,
                raw_body_ids=list(body),
                normalization=dict(
                    version=history.ORACLE_POLICY, changed=False, reason="canonical_gt_tokens"
                ),
            )
        # Every prefix ends at its Assistant body, before the closing delimiter.
        end = next(
            s["end"]
            for s in text["selected_spans"]
            if s["block"] == k and s["section"] == "assistant_plan"
        )
        return dict(
            mode=mode,
            **history.conditioning_metadata(mode),
            body_ids=body,
            **metadata,
            raw_text=self.tokenizer.decode(
                metadata["raw_body_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            text=self.tokenizer.decode(
                body, skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            controls=[x for x in body if x in self.emotions],
            intent_states_end=text["intent_state_ends"][k],
            input_tokens=end,
            excluded_current_assistant_media=True,
        )

    def prepare_sequence(self, record, visuals, blocks, mode, max_new_tokens, progress=None):
        """One full GT forward for oracle, one legal prefix per block for NTP.

        Only bidirectional closed-loop uses the NTP branch of this entry point.
        """
        if not 0 < blocks <= len(record["blocks"]):
            raise ValueError("Invalid text preparation length")
        if mode not in ("oracle", "closed_loop"):
            raise ValueError(f"Unknown inference mode: {mode}")
        emit = progress or (lambda **values: None)
        bodies = (
            history.oracle_bodies(record, blocks - 1, self.tokenizer) if mode == "oracle" else {}
        )
        metadata = [None] * blocks
        emit(phase="intent_to_gpu", completed=False, total_blocks=blocks)
        with self.on_gpu():
            emit(phase="intent_to_gpu", completed=True, total_blocks=blocks)
            if mode == "closed_loop":
                prefixes = []
                for k in range(blocks):
                    emit(
                        phase="ntp",
                        completed=False,
                        block=k,
                        completed_blocks=k,
                        total_blocks=blocks,
                    )
                    bodies[k], metadata[k] = self.generate_body(
                        record, visuals, bodies, k, max_new_tokens
                    )
                    text = self.serialize(record, visuals, bodies, k, "closed_loop")
                    prefixes.append(self.extract(text))
                    emit(
                        phase="intent_states",
                        completed=True,
                        block=k,
                        completed_blocks=k + 1,
                        total_blocks=blocks,
                    )
                # Each target keeps the hidden states from its own legal prefix.
                features = intent_conditioning.combine_block_features(prefixes)
                plans = [
                    self.plan_record(
                        mode,
                        bodies[k],
                        metadata[k],
                        self.serialize(record, visuals, bodies, k, "closed_loop"),
                        k,
                    )
                    for k in range(blocks)
                ]
                for plan in plans:
                    plan.update(
                        conditioning_policy="bidirectional_ntp_user_history_v1",
                        intent_state_strategy="ntp_block_prefixes_v1",
                        assistant_text_source="intent_stream_full_sequence",
                        asr_status="skipped",
                    )
            else:
                text = self.serialize(record, visuals, bodies, blocks - 1)
                emit(
                    phase="intent_states",
                    completed=False,
                    total_blocks=blocks,
                    input_tokens=len(text["input_ids"]),
                )
                features = self.extract(text)
                emit(
                    phase="intent_states",
                    completed=True,
                    total_blocks=blocks,
                    input_tokens=len(text["input_ids"]),
                    intent_state_rows=len(features["intent_states"]),
                )
                plans = [
                    self.plan_record(mode, bodies[k], metadata[k], text, k) for k in range(blocks)
                ]
        return dict(features=features, plans=plans, bodies=bodies)

    def plan(self, record, visuals, bodies, k, mode, max_new_tokens):
        """Causal closed-loop turn; oracle prefix path is retained for parity checks."""
        if mode not in ("oracle", "closed_loop"):
            raise ValueError(f"Unknown inference mode: {mode}")
        with self.on_gpu():
            if mode == "oracle":
                bodies = history.oracle_bodies(record, k, self.tokenizer)
                body, metadata = bodies[k], None
            else:
                body, metadata = self.generate_body(record, visuals, bodies, k, max_new_tokens)
                bodies = {**bodies, k: body}
            text = self.serialize(
                record, visuals, bodies, k, "full_gt" if mode == "oracle" else "closed_loop"
            )
            features = self.extract(text)
            return features, self.plan_record(mode, body, metadata, text, k)
