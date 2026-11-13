"""Isolated Qwen3-ASR transcription and forced-alignment worker."""

import argparse
import json
from pathlib import Path
import subprocess

from .defaults import DEFAULT_FORCED_ALIGNER, DEFAULT_SPEECH_RECOGNIZER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--speech-recognizer", default=DEFAULT_SPEECH_RECOGNIZER)
    parser.add_argument("--forced-aligner", default=DEFAULT_FORCED_ALIGNER)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel

    torch.set_num_threads(4)
    raw = subprocess.run(
        [
            args.ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            args.video,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    audio = np.frombuffer(raw, dtype="<f4").copy()
    if not len(audio):
        raise ValueError("The recording contains no audio samples.")
    model = Qwen3ASRModel.from_pretrained(
        args.speech_recognizer,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=1024,
        forced_aligner=args.forced_aligner,
        forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map="cuda:0"),
    )
    (result,) = model.transcribe(audio=[(audio, 16000)], language=[None], return_time_stamps=True)
    aligned = result.time_stamps
    items = (
        [
            dict(
                text=item.text,
                start_ms=round(float(item.start_time) * 1000),
                end_ms=round(float(item.end_time) * 1000),
            )
            for item in aligned.items
        ]
        if aligned
        else []
    )
    target = Path(args.output)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            dict(
                language=result.language,
                text=result.text,
                aligned_items=items,
                timing="qwen3_forced_alignment_v1",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


if __name__ == "__main__":
    main()
