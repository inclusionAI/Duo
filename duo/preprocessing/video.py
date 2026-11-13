"""CPU-only automatic token/video chunking in the Intent tokenizer environment."""

import argparse
import json
from pathlib import Path

from ..serving.common import atomic, native_imports
from .alignment import aligned_tokens, token_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    native_imports()
    import av
    import numpy as np
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from duo.models.realization.modules.intent_conditioning import visual_patch

    torch.set_num_threads(4)
    path = Path(args.prepared)
    prepared = json.loads(path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.intent, local_files_only=True)
    processor.tokenizer = tokenizer
    processor.video_processor.size = dict(shortest_edge=1024, longest_edge=16384)
    # No ASR text is allowed to manufacture protocol or vision control tokens.
    text = prepared["transcript"]["text"]
    if any(control in text for control in ("<|", "<wait>", "<speak>", "<blank>", "<chunk_")):
        raise ValueError("Recognized text contains reserved intent control tokens.")
    tokens = aligned_tokens(
        text, prepared["transcript"]["aligned_items"], tokenizer, prepared["crop"]["start"]
    )
    blocks = token_chunks(tokens, prepared["crop"]["chunks"])
    visuals, rgb = [], []
    with av.open(prepared["video"]) as container:
        for frame in container.decode(video=0):
            rgb.append(frame.to_ndarray(format="rgb24"))
            if len(rgb) == 17:
                visuals.append(visual_patch(processor, np.stack(rgb)))
                rgb.clear()
    if rgb or len(visuals) != len(blocks):
        raise ValueError("Normalized video frames do not match 17-frame blocks.")
    patch_path = path.with_name("user-patches.pt")
    torch.save(visuals, patch_path)
    prepared.update(
        tokens=tokens,
        body_ids=blocks,
        body_texts=[
            tokenizer.decode(ids, skip_special_tokens=False) if ids else "<blank>" for ids in blocks
        ],
        patches=str(patch_path),
        chunk_token=json.loads(Path(args.protocol).read_text())["chunk_tokens"]["708"],
    )
    atomic(path, prepared)


if __name__ == "__main__":
    main()
