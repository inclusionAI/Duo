# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline-only ASR/alignment; run in an isolated intent-asr environment."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# This offline SDPA worker does not use optional Hub kernel overrides.
sys.modules["kernels"] = None
from qwen_asr import Qwen3ASRModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from video_timeline import aligned_tokens, speech_gate, token_chunks  # noqa: E402

p = argparse.ArgumentParser()
for name in ("audio", "model", "aligner", "tokenizer", "output"):
    p.add_argument("--" + name, required=True)
p.add_argument("--frames", type=int, required=True)
a = p.parse_args()
torch.set_num_threads(4)
audio = np.fromfile(a.audio, dtype="<f4")
model = Qwen3ASRModel.from_pretrained(
    a.model,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
    max_inference_batch_size=1,
    max_new_tokens=1024,
    forced_aligner=a.aligner,
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"),
)
(result,) = model.transcribe(audio=[(audio, 16000)], language=[None])
if not result.text.strip():
    raise ValueError("视频中未识别到语音")
if any(t in result.text for t in ("<|", "<wait>", "<speak>", "<blank>", "<chunk_")):
    raise ValueError("ASR 文本包含协议控制符")
(aligned,) = model.forced_aligner.align(audio=[(audio, 16000)], text=[result.text], language=[result.language])
items = [dict(text=i.text, start_ms=round(i.start_time * 1000), end_ms=round(i.end_time * 1000)) for i in aligned.items]
start = math.floor(min(i["start_ms"] for i in items) * 24 / 1000)
stop = min(a.frames, math.ceil(max(i["end_ms"] for i in items) * 24 / 1000))
if stop <= start:
    raise ValueError("对齐后的语音不在视频时间范围内")
count = math.ceil((stop - start) / 17)
tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
tokens = aligned_tokens(result.text, items, tokenizer, start / 24)
blocks = token_chunks(tokens, count)
gate = speech_gate(tokens, count)
value = dict(
    text=result.text,
    language=result.language,
    items=items,
    tokens=tokens,
    frame_start=start,
    frame_stop=stop,
    input_blocks=count,
    gate=gate,
    body_texts=[tokenizer.decode(ids, skip_special_tokens=False) if ids else "" for ids in blocks],
    last_speaking=max(i for i, ids in enumerate(blocks) if ids),
)
Path(a.output).write_text(json.dumps(value, ensure_ascii=False, indent=2))
