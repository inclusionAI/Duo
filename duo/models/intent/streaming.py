"""One resident Intent v2: incremental greedy NTP and native layer-50 intent states.

Reference: duo/pipelines/intent_rollout.py (native_ntp, legacy layout). DiT/VAE
cache resets do not touch this producer or its intent states queue.
"""

import copy
import inspect
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

from ...config import load_service_config
from ...serving.common import RUNTIME, atomic, read, native_imports, clock_block
from ...serving.cases import Cases
from .prompt import OriginalPrompt
from ...preprocessing.alignment import speech_gate, user_mapping
from .reply_boundary import ReplyBoundary
from ...runtime.components.checkpoint_schema import intent_identity

native_imports()


class StreamingIntent:
    def __init__(
        self,
        case_id="default",
        emit=lambda event: None,
        request_root=None,
        settings=None,
        config_path=None,
    ):
        import torch
        from duo.models.realization.modules import intent, intent_conditioning as qc

        config = load_service_config(config_path or os.environ["DUO_CONFIG"])
        self.config = config
        self.torch, self.intent, self.qc = torch, intent, qc
        self.emit = emit
        self.request_root = request_root or RUNTIME / "requests"
        self.wait_silence = False
        self.condition_version = 0
        self.trace = None
        self.args = SimpleNamespace(
            intent=config.resolve(config.weights.intent),
            intent_adapter=config.resolve(config.weights.intent_adapter),
            origin_intent=False,
            intent_protocol_manifest=config.resolve(config.weights.intent_protocol),
            intent_chunk_token="",
            encode="duo_5_17",
            device="cuda:0",
            intent_attention="fa2",
            intent_max_pixels=16384,
            max_intent_tokens=131072,
            max_new_tokens=128,
            text_time_mode="planning",
            intent_rollout=True,
        )
        emit(dict(type="service", phase="loading_intent", detail="Loading intent"))
        self.processor, self.model, self.model_info = intent.load(self.args, dtype=torch.bfloat16)
        self.args.max_intent_tokens = self.model.config.text_config.max_position_embeddings
        self.tokenizer = self.processor.tokenizer
        manifest = read(Path(config.resolve(config.weights.realization)) / "complete.json")
        if self.args.intent_identity != intent_identity(manifest):
            raise ValueError("Intent identity does not match realization checkpoint")
        self.core = self.model.get_base_model().model
        self.mm_rope = "mm_token_type_ids" in inspect.signature(self.core.get_rope_index).parameters
        self.wait_id = self.tokenizer.convert_tokens_to_ids("<wait>")
        self.speak_id = self.tokenizer.convert_tokens_to_ids("<speak>")
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if intent.explicit_speak_token(self.tokenizer) != "<speak>":
            raise ValueError("NTP requires the declared Intent v2 speak token")
        self.original_prompt = None
        self.configure_case(settings or Cases().settings(case_id))

    def encode(self, text):
        return self.intent.encode_protocol(self.tokenizer, text)

    def decode(self, ids):
        return self.tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def prompt(self, messages, generation=False):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=generation
        )

    def system_prefix(self):
        description = self.case_settings["description"].strip()
        self.system = (
            self.qc.INTRO.replace("<chunk_160>", self.tokenizer.reciprocal_chunk_token)
            + description
            + self.qc.EXPLICIT_SPEAK_INSTRUCTION
            + self.qc.SPEECH_CUE
        )
        messages = [dict(role="system", content=self.system)]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids = self.prompt(messages)
        if encoded["input_ids"] != ids:
            raise ValueError("Intent template token/offset mapping differs")
        origin = rendered.index(self.system)
        description_start = origin + self.system.index('"' + description + '"') + 1
        cue_start = origin + self.system.rindex(self.qc.SPEECH_CUE)
        keep, groups, context = [False] * len(ids), [], []
        cursor = 0
        for section, start, stop, text in (
            (
                "assistant_description",
                description_start,
                description_start + len(description),
                description,
            ),
            ("speech_cue", cue_start, cue_start + len(self.qc.SPEECH_CUE), self.qc.SPEECH_CUE),
        ):
            rows = [
                i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < stop and b > start
            ]
            if not rows or any(keep[i] for i in rows):
                raise ValueError("Missing or overlapping shared P rows")
            for row in rows:
                keep[row] = True
            groups.append(
                dict(
                    start=cursor,
                    end=cursor + len(rows),
                    section=section,
                    block=-1,
                    phase=-1,
                    role="system",
                )
            )
            context.append(dict(section=section, rows=len(rows), text=text))
            cursor += len(rows)
        return dict(input_ids=ids, keep=keep, text_groups=groups), context

    def record(self, count):
        return {
            "sample_id": self.case_id,
            "assistant_description_text": self.case_settings["description"],
            "blocks": [clock_block(k) for k in range(count)],
            "wait_silence": False,
            "canonical": {
                "text_tokens": [],
                "duration_ms": count * 17 * 1000 / 24,
                "video": {"source_frames": count * 17, "source_fps": 24},
            },
        }

    def configure_case(self, settings):
        from duo.models.realization.modules import media_features

        if self.trace:
            self.trace.close()
        self.case_settings, self.case_id = copy.deepcopy(settings), settings["case_id"]
        reference_path = Cases().assistant_image(self.case_id)
        self.source = {"media": {"assistant_image": str(reference_path)}}
        self.reference_rgb = media_features.read_image(
            reference_path, self.config.inference.height, self.config.inference.width
        )
        # Use the reference NTP's complete ChatML tokenization, including
        # tokens that span the description/quote boundary. P still selects
        # description + speech cue; no canned User opening is present.
        self.prefix_text, context_groups = self.system_prefix()
        self.prefix_length = len(self.prefix_text["input_ids"])
        self.shared_groups = [dict(g) for g in self.prefix_text["text_groups"]]
        self.condition_version += 1
        self.prefix_folder = RUNTIME / "case" / self.case_id / settings["config_id"] / "native_ntp"
        self.prefix_folder.mkdir(parents=True, exist_ok=True)
        self.reset_conversation()
        selected = [i for i, keep in enumerate(self.prefix_text["keep"]) if keep]
        hidden, timing = self.forward(
            self.prefix_text["input_ids"], selected=selected, role="system"
        )
        self.prefix_intent_states, self.shared_rows = hidden, len(hidden)
        self.shared_prompts = dict(
            finetuned=dict(
                intent_states=hidden,
                groups=self.shared_groups,
                context_groups=context_groups,
                intent=self.args.intent_identity,
            )
        )
        atomic(
            self.prefix_folder / "prefix.json",
            dict(
                tokens=self.prefix_length,
                intent_state_rows=self.shared_rows,
                intent=self.args.intent_identity,
                model=self.model_info,
                settings=settings,
                condition_version=self.condition_version,
                intent_state_layer=50,
                lora_dtype="float32",
                base_dtype="bfloat16",
                intent_state_strategy="ntp_forward_layer50_v1",
                conditioning_profile="legacy",
                timing=timing,
            ),
        )
        (self.prefix_folder / "full_prefix.txt").write_text(
            self.decode(self.prefix_text["input_ids"])
        )
        self.boundaries[0] = self.checkpoint()

    def reset_conversation(self):
        self.past, self.logits, self.rope_position = None, None, 0
        self.cached_ids, self.history = [], []
        self.messages = [dict(role="system", content=self.system)]
        self.visuals, self.boundaries = {}, {}
        self.next_global = 0
        self.request = None
        self.next_local = 0
        self.trace = None

    def forward(self, ids, *, visual=None, selected=(), role="protocol", reset=False):
        torch = self.torch
        if reset:
            self.past, self.logits, self.rope_position = None, None, 0
            self.cached_ids = []
        if not ids or len(self.cached_ids) + len(ids) > self.args.max_intent_tokens:
            raise ValueError("The intent context is full. Reset the session to continue.")
        past = len(self.cached_ids)
        tokens = torch.tensor([ids], dtype=torch.long, device="cuda:0")
        visuals = visual if isinstance(visual, list) else [visual] if visual is not None else []
        specs = [self.visuals[index] for index in visuals]
        grid = (
            torch.tensor([spec["grid"] for spec in specs], dtype=torch.long, device="cuda:0")
            if specs
            else None
        )
        rope_args = dict(
            input_ids=tokens, attention_mask=torch.ones_like(tokens), video_grid_thw=grid
        )
        if self.mm_rope:
            rope_args["mm_token_type_ids"] = (
                tokens == self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
            ).long() * 2
        position, _ = self.core.get_rope_index(**rope_args)
        position = position.to("cuda:0") + self.rope_position
        kwargs = dict(
            input_ids=tokens,
            attention_mask=torch.ones((1, past + len(ids)), dtype=torch.long, device="cuda:0"),
            position_ids=position,
            cache_position=torch.arange(past, past + len(ids), device="cuda:0"),
            past_key_values=self.past,
            use_cache=True,
            logits_to_keep=1,
        )
        if specs:
            kwargs.update(
                pixel_values_videos=torch.cat([spec["patches"] for spec in specs]).to("cuda:0"),
                video_grid_thw=grid,
            )
        captured = []

        def capture(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured.append(value[0, list(selected)].detach().cpu().contiguous())

        hook = (
            self.intent.language_layers(self.model)[49].register_forward_hook(capture)
            if selected
            else None
        )
        started = time.time()
        try:
            with torch.no_grad():
                output = self.model(**kwargs)
        finally:
            if hook:
                hook.remove()
        if output.past_key_values.get_seq_length() != past + len(ids):
            raise ValueError("Intent incremental KV length differs")
        self.past, self.logits = output.past_key_values, output.logits[:, -1, :]
        self.rope_position = int(position.max().item()) + 1
        self.cached_ids.extend(ids)
        segment = dict(
            ids=list(ids),
            visual=visual,
            role=role,
            start=past,
            stop=len(self.cached_ids),
            rope_stop=self.rope_position,
            selected=list(selected),
        )
        self.history.append(segment)
        if self.trace:
            self.trace.write(
                json.dumps(dict(segment, reset=reset, started=started, finished=time.time())) + "\n"
            )
        hidden = captured[0] if captured else torch.empty((0, 5120), dtype=torch.bfloat16)
        if hidden.shape != (len(selected), 5120) or not torch.isfinite(hidden).all():
            raise ValueError("Invalid native NTP intent states")
        return hidden, dict(
            started=started,
            finished=time.time(),
            forwarded_tokens=len(ids),
            start_token=past,
            stop_token=len(self.cached_ids),
            role=role,
        )

    def checkpoint(self):
        return dict(
            tokens=len(self.cached_ids), segments=len(self.history), rope=self.rope_position
        )

    def rewind(self, checkpoint):
        if self.past is not None:
            self.past.crop(checkpoint["tokens"])
        del self.cached_ids[checkpoint["tokens"] :]
        del self.history[checkpoint["segments"] :]
        self.rope_position = checkpoint["rope"]
        self.logits = None

    def ensure_original_prompt(self):
        if "original" not in self.shared_prompts:
            if self.original_prompt is None:
                self.original_prompt = OriginalPrompt(self)
            self.shared_prompts["original"] = self.original_prompt.extract(
                self, self.prefix_folder / "original_prompt"
            )
        return self.public_condition()

    def public_condition(self):
        return dict(
            version=self.condition_version,
            wait_silence=False,
            context=self.public_context(),
            shared=self.prefix_intent_states.float().numpy(),
            shared_groups=self.shared_groups,
            shared_variants={
                name: dict(
                    shared=value["intent_states"].float().numpy(),
                    shared_groups=value["groups"],
                    intent=value["intent"],
                )
                for name, value in self.shared_prompts.items()
            },
        )

    def public_context(self):
        return dict(
            description=self.case_settings["description"],
            settings=self.case_settings,
            fixedPrefix=self.decode(self.prefix_text["input_ids"]),
            warmup=[],
            sharedSource="finetuned",
            sharedGroups=self.shared_prompts["finetuned"]["context_groups"],
            sharedVariants={
                name: dict(
                    groups=value["context_groups"],
                    rows=len(value["intent_states"]),
                    intent=value["intent"],
                )
                for name, value in self.shared_prompts.items()
            },
        )

    def chunk(self, local):
        prepared = self.request["prepared"]
        texts = prepared["body_texts"]
        source_local = local if local < len(texts) else self.last_speaking
        index = self.start_global + local
        return dict(
            global_index=index,
            local=local,
            text="",
            condition="",
            user=user_mapping(
                prepared, local, self.start_global, texts[source_local], source_local=source_local
            ),
            sequence_url=self.sequence_url,
            condition_version=self.condition_version,
            user_speech_gate=dict(
                self.gate,
                applied=self.gate["release_block"] is None or index < self.gate["release_block"],
            ),
            wait_silence=False,
            token_ids=[],
        )

    def begin(self, request, start_global):
        if self.request is not None or start_global != self.next_global:
            raise ValueError("Intent request boundary differs from the DiT stream")
        self.request = request
        self.start_global, self.next_local = start_global, 0
        prepared = request["prepared"]
        self.input_count = len(prepared["body_ids"])
        self.reply = ReplyBoundary(self.input_count)
        self.gate = speech_gate(prepared["tokens"], self.input_count, start_global)
        speaking = [k for k, ids in enumerate(prepared["body_ids"]) if ids]
        if not speaking:
            raise ValueError("The user video contains no usable speech tokens.")
        self.last_speaking = speaking[-1]
        visuals = self.torch.load(prepared["patches"], map_location="cpu", weights_only=True)
        if len(visuals) != self.input_count:
            raise ValueError("User video patches and ASR chunks differ")
        for k, visual in enumerate(visuals):
            self.visuals[start_global + k] = visual
        self.directory = self.request_root / str(request["id"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace = (self.directory / "forward.jsonl").open("a", buffering=1)
        self.sequence_url = f"./requests/{self.request_root.name}/{request['id']}/full_sequence.txt"
        self.chunks = [self.chunk(k) for k in range(self.input_count)]
        self.write_sequence()
        return self.chunks

    def write_sequence(self):
        path = self.directory / "full_sequence.txt"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(self.decode(self.cached_ids))
        temporary.replace(path)
        atomic(
            self.directory / "sequence.json",
            dict(
                input_ids=self.cached_ids,
                shared_groups=self.shared_groups,
                chunks=self.chunks,
                intent_state_strategy="ntp_forward_layer50_v1",
                conditioning_profile="legacy",
                rollout_policy="intent_reference_rollout_v4",
                intent=self.args.intent_identity,
            ),
        )

    def next_intent_states(self):
        if self.reply.complete:
            return dict(type="ntp_done", request_id=self.request["id"], reason="waiting_for_user")
        local, index = self.next_local, self.next_global
        chunk = self.chunks[local] if local < len(self.chunks) else self.chunk(local)
        source = self.start_global + (local if local < self.input_count else self.last_speaking)
        self.visuals[index] = self.visuals[source]
        spec = self.visuals[index]
        unit = clock_block(index)
        t, h, w = spec["grid"]
        count = h * w // spec["merge"] ** 2
        video = "".join(
            f"<{round(min(unit['frame_start'] + 2 * j + 1, unit['frame_stop'] - 1) * 1000 / 24) / 1000:.1f} seconds>"
            + "<|vision_start|>"
            + "<|video_pad|>" * count
            + "<|vision_end|>"
            for j in range(t)
        )
        body = (
            self.request["prepared"]["body_texts"][local] if local < self.input_count else "<blank>"
        )
        messages = list(self.messages)
        if index == 0:
            messages.append(dict(role="assistant", content="<wait>"))
        messages.append(dict(role="user", content=video + body))
        ids = self.prompt(messages, generation=True)
        if len(ids) >= self.args.max_intent_tokens:
            return dict(type="ntp_done", request_id=self.request["id"], reason="context_limit")
        if local >= len(self.chunks):
            self.chunks.append(chunk)
            self.emit(
                dict(
                    type="planned",
                    request_id=self.request["id"],
                    chunks=[chunk],
                    sequence_url=self.sequence_url,
                )
            )
        self.emit(
            dict(
                type="stage",
                request_id=self.request["id"],
                index=index,
                stage="intent_states",
                state="running",
            )
        )
        began = time.time()
        rebuild = ids[: len(self.cached_ids)] != self.cached_ids
        suffix = ids if rebuild else ids[len(self.cached_ids) :]
        self.forward(
            suffix,
            visual=list(range(index + 1)) if rebuild else index,
            role="user_prompt",
            reset=rebuild,
        )
        self.messages = messages
        checkpoint = self.checkpoint()
        raw, pieces = [], []
        gated = chunk["user_speech_gate"]["applied"]
        ended = False
        if gated:
            raw = [self.wait_id]
            hidden, _ = self.forward(raw, selected=[0], role="no_barge_in_wait")
            pieces.append(hidden)
        else:
            for _ in range(
                min(self.args.max_new_tokens, self.args.max_intent_tokens - len(self.cached_ids))
            ):
                token = int(self.logits.argmax(-1).item())
                if token == self.eos_id:
                    ended = True
                    break
                raw.append(token)
                hidden, _ = self.forward([token], selected=[0], role="assistant_ntp")
                pieces.append(hidden)
        valid = not raw or raw[0] in (self.wait_id, self.speak_id)
        committed = raw if raw and raw[0] == self.speak_id else [self.wait_id]
        if committed == raw[: len(committed)]:
            pieces = pieces[: len(committed)]
            if len(committed) < len(raw):
                first = self.history[checkpoint["segments"]]
                self.rewind(
                    dict(
                        tokens=checkpoint["tokens"] + 1,
                        segments=checkpoint["segments"] + 1,
                        rope=first["rope_stop"],
                    )
                )
        else:
            self.rewind(checkpoint)
            hidden, _ = self.forward(committed, selected=[0], role="assistant_wait_fallback")
            pieces = [hidden]
        hidden = self.torch.cat(pieces)
        condition = self.decode(committed)
        self.messages.append(dict(role="assistant", content=condition))
        chunk.update(
            text=condition,
            condition=condition,
            token_ids=committed,
            raw_token_ids=raw,
            action="wait" if committed[0] == self.wait_id else "speak",
            action_valid=valid,
            input_rows=list(range(checkpoint["tokens"], checkpoint["tokens"] + len(committed))),
            intent_state_rows=[self.shared_rows, self.shared_rows + len(hidden)],
            ntp_ended=ended,
            intent_state_strategy="ntp_forward_layer50_v1",
            truncated=not ended and not gated and len(raw) == self.args.max_new_tokens,
        )
        chunk["user_speech_gate"]["decision_source"] = (
            "forced_wait_forward"
            if gated
            else "intent_greedy"
            if valid
            else "invalid_action_wait_forward"
        )
        chunk["reply_complete"] = self.reply.observe(local, chunk["action"], gated)
        chunk["reply_wait_chunks"] = self.reply.waiting
        self.next_local += 1
        self.next_global += 1
        self.boundaries[self.next_global] = self.checkpoint()
        self.write_sequence()
        timing = dict(
            started=began,
            finished=time.time(),
            forwarded_tokens=len(self.cached_ids) - checkpoint["tokens"] + len(suffix),
            intent_state_strategy="ntp_forward_layer50_v1",
            lora_dtype="float32",
            intent_state_rows=len(hidden),
        )
        atomic(self.directory / "intent_states" / f"C{index:05d}.json", dict(chunk, timing=timing))
        return dict(
            type="intent_states",
            request_id=self.request["id"],
            index=index,
            intent_states=hidden.float().numpy(),
            group=dict(
                start=0,
                end=len(hidden),
                section="assistant_plan",
                block=index,
                phase=index,
                role="assistant",
            ),
            timing=timing,
            chunk=copy.deepcopy(chunk),
            wait_silence=False,
            condition_version=self.condition_version,
        )

    def finish_request(self):
        if self.request is not None:
            self.chunks = [c for c in self.chunks if c["global_index"] < self.next_global]
            self.write_sequence()
        if self.trace:
            self.trace.close()
            self.trace = None
        self.request = None
        self.boundaries = {self.next_global: self.checkpoint()}

    def reset_all(self):
        self.finish_request()
        self.reset_conversation()
        selected = [i for i, keep in enumerate(self.prefix_text["keep"]) if keep]
        hidden, timing = self.forward(
            self.prefix_text["input_ids"], selected=selected, role="system"
        )
        self.boundaries[0] = self.checkpoint()
        return dict(
            prefix_rebuilt=True,
            prefix_intent_states_equal=self.torch.equal(hidden, self.prefix_intent_states),
            intent_history_cleared=True,
            prefix_forward=timing,
        )
