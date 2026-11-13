"""Text allocation on the actual 17/24-second clock; never rewrite the input."""

import re
from .common import FRAMES, FPS

DEFAULT_RATE = 3
MAX_HAN = 3
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002FA1F]")


def split_manual(text):
    """Each nonempty input line is one chunk; newlines only mark boundaries."""
    chunks = []
    for line_number, line in enumerate(re.split(r"\r\n|\r|\n", text), 1):
        if not line.strip():
            continue
        if len(HAN.findall(line)) > MAX_HAN:
            raise ValueError(f"Line {line_number} is too long; split it into multiple chunks")
        chunks.append(line)
    if not chunks:
        raise ValueError("Enter one text chunk per line")
    return chunks


def split_text(text, rate=DEFAULT_RATE):
    target = rate * FRAMES / FPS
    words = re.findall(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*|[\s\S]", text)
    chunks, part, units, han = [], "", 0.0, 0
    for word in words:
        punctuation = bool(re.fullmatch(r"[，。！？、；：,.!?;:…—“”‘’「」『』（）()\s]+", word))
        if punctuation:
            if part or not chunks:
                part += word
            else:
                chunks[-1] += word
            if units and re.search(r"[，。！？、；：,.!?;:\n]", word):
                chunks.append(part)
                part, units, han = "", 0.0, 0
            continue
        word_han = len(HAN.findall(word))
        weight = max(1, len(word) * 0.4) if word[0].isascii() and word[0].isalnum() else 1
        if part and units and (units + weight > target or han + word_han > MAX_HAN):
            chunks.append(part)
            part, units, han = "", 0.0, 0
        part += word
        units += weight
        han += word_han
    if part:
        chunks.append(part)
    assert "".join(chunks) == text
    assert all(len(HAN.findall(chunk)) <= MAX_HAN for chunk in chunks)
    return chunks
