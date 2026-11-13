"""A small classifier on Realization fake-score features; no training or data ownership."""

import torch
from torch import nn


class DiscriminatorHead(nn.Module):
    def __init__(self, hidden_size, width=256, heads=8):
        super().__init__()
        self.config = dict(
            hidden_size=hidden_size,
            width=width,
            heads=heads,
            policy="middle_block_attention_pool_v1",
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, width)
        self.query = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.query, std=0.02)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(
        self,
        features,
        groups,
        video_indices,
        audio_indices,
        video_weights,
        audio_weights,
    ):
        valid = torch.zeros(features.shape[1], device=features.device, dtype=torch.bool)
        valid[video_indices] = video_weights > 0
        valid[audio_indices] = audio_weights > 0
        outputs = []
        # The backbone already mixes the complete temporal sequence with its
        # positions. Pool only valid media, independently for every conversation.
        for sample in sorted({g["sample"] for g in groups if g["kind"] == "noisy"}):
            mask = torch.zeros_like(valid)
            for g in groups:
                if g["sample"] == sample and g["kind"] == "noisy":
                    mask[g["start"] : g["end"]] = True
            mask &= valid
            if not bool(mask.any()):
                continue
            x = self.projection(self.norm(features[:, mask].float()))
            pooled, _ = self.pool(self.query, x, x, need_weights=False)
            outputs.append(self.classifier(pooled).reshape(()))
        if not outputs:
            # Empty local supervision still participates in distributed backward.
            # Connect every head parameter and the input graph with zero weight.
            zero = features.sum() * 0
            for parameter in self.parameters():
                zero = zero + parameter.sum() * 0
            return zero.expand(0)
        return torch.stack(outputs)


def build_discriminator_head(hidden_size, seed):
    """Initialize on CPU without resetting any CUDA generator."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return DiscriminatorHead(hidden_size).float()
