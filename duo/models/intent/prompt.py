"""Original-Intent global P, independent of the v2 prefix and chunk features."""

import copy
import time

from ...serving.common import atomic


def shared_text(text):
    """Keep the native causal input through the final global-prompt token."""
    spans = [s for s in text["selected_spans"] if s["block"] == -1]
    stop = max(s["end"] for s in spans)
    return dict(
        text,
        input_ids=text["input_ids"][:stop],
        keep=text["keep"][:stop],
        input_spans=[s for s in text["input_spans"] if s["end"] <= stop],
        text_groups=[g for g in text["text_groups"] if g["block"] == -1],
        selected_spans=spans,
        patches=[],
        video_grid_thw=[],
        controls=[],
        intent_state_ends=[],
    )


def describe_groups(text, tokenizer):
    return [
        dict(
            section=g["section"],
            rows=g["end"] - g["start"],
            text=tokenizer.decode(
                text["input_ids"][s["start"] : s["end"]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )
        for g, s in zip(text["text_groups"], text["selected_spans"])
        if g["block"] == -1
    ]


class OriginalPrompt:
    def __init__(self, stream):
        # The historical origin-intent entry and the loaded v2 base resolve to
        # the same checkpoint. Keep one resident base and disable *all* PEFT
        # adapters (including trained token embeddings) only for this forward.
        self.args = copy.copy(stream.args)
        self.args.origin_intent = True
        self.args.intent_adapter = ""
        self.args.intent_protocol_manifest = ""
        self.tokenizer, _ = stream.intent.prepare_tokenizer(self.args)
        config = stream.model.active_peft_config
        if config.bias != "none" or getattr(config, "lora_bias", False):
            raise ValueError("Original P requires a base without trained adapter biases")
        if stream.model.get_model_status().merged_adapters:
            raise ValueError("Original P requires unmerged Intent adapters")
        embedding = stream.model.get_input_embeddings()
        # PEFT's TrainableTokensWrapper exposes the unmodified embedding here,
        # but does not forward num_embeddings while its adapter is enabled.
        embedding = getattr(embedding, "original_module", embedding)
        if max(self.tokenizer.reciprocal_vocab_ids) >= embedding.num_embeddings:
            raise ValueError("Original Intent vocabulary exceeds resident base embeddings")

    def extract(self, stream, folder):
        # P is before every User/video/Assistant chunk. The native origin-mode
        # serializer supplies its own tokenizer, instruction and control filter;
        # causal extraction can stop at the speech cue without any request KV.
        text = stream.qc.serialize(
            stream.record(1),
            self.tokenizer,
            {},
            "planning",
            generation=True,
            context_mode="closed_loop",
        )
        text = shared_text(text)
        model = stream.model
        core = model.get_base_model().model
        rope_deltas = core.rope_deltas
        began = time.time()
        try:
            with model.disable_adapter(), stream.torch.inference_mode():
                result = stream.qc.extract_intent_states(text, model)
        finally:
            core.rope_deltas = rope_deltas
            model.requires_grad_(False)
        hidden = result["intent_states"]
        groups = [dict(g) for g in result["text_groups"]]
        context_groups = describe_groups(text, self.tokenizer)
        atomic(
            folder / "prompt.json",
            dict(
                source="original",
                intent=self.args.intent_identity,
                execution="resident_base_with_all_adapters_disabled",
                use_cache=False,
                intent_state_layer=50,
                before_final_norm=True,
                rows=len(hidden),
                input_ids=text["input_ids"],
                input_keep=text["keep"],
                input_spans=text["input_spans"],
                text_groups=groups,
                selected_spans=text["selected_spans"],
                wait_silence=stream.wait_silence,
                condition_version=stream.condition_version,
                started=began,
                finished=time.time(),
            ),
        )
        (folder / "full_prompt.txt").write_text(
            self.tokenizer.decode(
                text["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        )
        (folder / "selected_condition.txt").write_text(
            "\n\n".join(g["text"] for g in context_groups)
        )
        return dict(
            intent_states=hidden,
            groups=groups,
            context_groups=context_groups,
            intent=self.args.intent_identity,
        )
