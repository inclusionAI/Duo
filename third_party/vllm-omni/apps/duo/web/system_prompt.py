# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Normalize editable system messages to the Realization frame-v5 protocol."""

import re


def resolve(text):
    text = text.strip()
    if not text or len(text) > 16000:
        raise ValueError("System prompt 不能为空，最多 16000 字")
    start = "<|im_start|>system\n"
    if text.startswith(start):
        text = text[len(start) :].removesuffix("<|im_end|>").rstrip()
    if "<|" in text:
        raise ValueError("请只提供一条 system 消息，不要附加其他角色的消息")
    if not text.startswith("<chunk_"):
        description = text
        text = (
            "<chunk_708> You are the assistant in a real-time full-duplex video call.\n"
            "The assistant's visual appearance and scene are described as follows:\n"
            f'"{description}"\n'
            "Begin every speaking chunk with <speak>. Output <wait> if the assistant "
            "is waiting or continuing an unfinished utterance.\n"
            "In this chunk, the assistant should say:"
        )
    else:
        match = re.search(r'described as follows:\s*"(.*?)"\s*\n', text, re.S)
        if (
            not match
            or not text.startswith("<chunk_708>")
            or not text.endswith("In this chunk, the assistant should say:")
        ):
            raise ValueError("完整 Realization system 消息需保留 <chunk_708>、引号内的人物描述和末尾 speech cue")
        description = match[1].strip()
        text = text[: match.start(1)] + description + text[match.end(1) :]
    if not description or any(t in description for t in ("<|", "<chunk_", "<wait>", "<speak>", "<blank>")):
        raise ValueError("人物描述不能为空，也不能包含协议控制符")
    return description, text
