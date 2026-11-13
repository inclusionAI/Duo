"""Optional PyTorch reference module for Duo's joint-media chunk layout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class DuoModelConfig:
    input_size: int
    hidden_size: int = 256
    output_size: int | None = None
    layers: int = 4
    attention_heads: int = 8
    mlp_ratio: float = 4.0
    token_types: int = 5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        if self.layers <= 0 or self.attention_heads <= 0:
            raise ValueError("layers and attention_heads must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.token_types <= 0:
            raise ValueError("token_types must be positive")


class JointMediaTransformer(nn.Module):
    """Small reference network operating on an externally constructed layout.

    Inputs are already embedded reference, condition, and media tokens. The
    caller owns modality-specific patchification and supplies the causal
    visibility matrix produced by :func:`duo.model.build_joint_layout`.
    """

    def __init__(self, config: DuoModelConfig) -> None:
        super().__init__()
        self.config = config
        output_size = config.output_size or config.input_size
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        self.type_embedding = nn.Embedding(config.token_types, config.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.attention_heads,
            dim_feedforward=int(config.hidden_size * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, output_size)

    def forward(
        self,
        tokens: Tensor,
        *,
        allowed: Tensor,
        token_types: Tensor,
        padding: Tensor | None = None,
    ) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [batch, sequence, channels]")
        batch, sequence, channels = tokens.shape
        if channels != self.config.input_size:
            raise ValueError("token channel size does not match the model configuration")
        if allowed.shape != (sequence, sequence) or allowed.dtype != torch.bool:
            raise ValueError("allowed must be a boolean [sequence, sequence] matrix")
        if token_types.shape != (batch, sequence):
            raise ValueError("token_types must have shape [batch, sequence]")
        if padding is not None and padding.shape != (batch, sequence):
            raise ValueError("padding must have shape [batch, sequence]")

        hidden = self.input_projection(tokens) + self.type_embedding(token_types)
        hidden = self.blocks(hidden, mask=~allowed, src_key_padding_mask=padding)
        return self.output_projection(self.output_norm(hidden))
