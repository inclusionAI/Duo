import pytest

from duo.models.intent.streaming import StreamingIntent
from duo.models.realization.modules import intent_conditioning


class DictDefaultTokenizer:
    reciprocal_chunk_token = "<chunk_708>"

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        encoded = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=False, return_dict=True
    ):
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        if not tokenize:
            return rendered
        encoded = self(rendered)
        return encoded if return_dict else encoded["input_ids"]


@pytest.fixture
def stream():
    stream = StreamingIntent.__new__(StreamingIntent)
    stream.tokenizer = DictDefaultTokenizer()
    stream.qc = intent_conditioning
    stream.case_settings = {"description": "一位穿灰色卫衣的学生。"}
    return stream


def test_system_prefix_preserves_selected_rows_with_dict_default_tokenizer(stream):
    prefix, context = stream.system_prefix()
    selected_text = "".join(
        chr(token) for token, keep in zip(prefix["input_ids"], prefix["keep"]) if keep
    )
    assert selected_text == stream.case_settings["description"] + stream.qc.SPEECH_CUE
    assert [group["section"] for group in context] == ["assistant_description", "speech_cue"]
    assert prefix["text_groups"][-1]["end"] == sum(prefix["keep"])


def test_generation_prompt_returns_ids_with_dict_default_tokenizer(stream):
    messages = [{"role": "user", "content": "你好"}]
    rendered = stream.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    expected_ids = stream.tokenizer(rendered)["input_ids"]
    assert stream.prompt(messages, generation=True) == expected_ids


def test_system_prefix_still_rejects_mismatched_ids(stream, monkeypatch):
    monkeypatch.setattr(stream, "prompt", lambda messages: [0])
    with pytest.raises(ValueError, match="Intent template token/offset mapping differs"):
        stream.system_prefix()
