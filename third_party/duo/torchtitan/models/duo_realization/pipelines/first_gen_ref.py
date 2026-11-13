"""Inference-only immutable copies of the first generated audio/video blocks."""

from dataclasses import replace

POLICY = "generated_av_clean_kv_v1"
REGION = "first_gen_ref"


class FirstGenReference:
    def __init__(self, blocks):
        if type(blocks) is not int or blocks < 1:
            raise ValueError(
                "First generated reference requires a positive block count"
            )
        self.limit = blocks
        self.blocks = 0
        self.cache = None

    def capture(self, block, committed):
        """Reuse the normal clean commit; never encode media or retain h50 rows."""
        if block >= self.limit:
            return False
        if block != self.blocks:
            raise ValueError("Noncontiguous first generated reference")
        selected = committed.select_groups(
            lambda g: g["kind"] == "clean"
            and g["block"] == block
            and g.get("modality") in ("video", "audio")
            and not g.get(REGION, False)
        )
        if {g.get("modality") for g in selected.groups} != {"video", "audio"}:
            raise ValueError("Generated reference requires both clean video and audio")
        selected = replace(
            selected, groups=tuple(dict(g, **{REGION: True}) for g in selected.groups)
        )
        self.cache = (
            selected
            if self.cache is None
            else type(selected).concatenate(self.cache, selected)
        )
        self.blocks += 1
        return True

    @property
    def rows(self):
        return 0 if self.cache is None else self.cache.length

    def attach(self, ordinary):
        """Canonical order: original reference/shared, fixed AV copies, live AV."""
        if self.cache is None:
            return ordinary
        if any(g.get(REGION, False) for g in ordinary.groups):
            raise ValueError(
                "Remove old anchor copies before attaching the reference bank"
            )
        prefix = ordinary.select_groups(lambda g: g["kind"] in ("reference", "shared"))
        history = ordinary.select_groups(lambda g: g["kind"] == "clean")
        return type(ordinary).concatenate(prefix, self.cache, history)


def extra_workspace_bytes(index, wanted, blocks, layers, hidden_size, element_bytes):
    """Conservative extra old/new KV workspace; anchors are outside the window cap."""
    rows = sum(
        b["video_rows"] + b["stereo_audio_rows"]
        for b in index["blocks"][: min(blocks, wanted)]
    )
    return 2 * rows * 2 * layers * hidden_size * element_bytes
