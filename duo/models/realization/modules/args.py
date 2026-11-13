from dataclasses import dataclass, field
from duo.runtime.protocols.model import BaseModelArgs
from .convert import DUO_MEDIA_TRANSFORMER_CONFIG


@dataclass
class DuoModelArgs(BaseModelArgs):
    values: dict = field(default_factory=lambda: dict(DUO_MEDIA_TRANSFORMER_CONFIG))

    def to_dict(self):
        return dict(self.values)

    def update_from_config(self, job_config, **kwargs):
        pass

    def get_nparams_and_flops(self, model, seq_len):
        n = sum(p.numel() for p in model.parameters())
        return n, 6 * n + 12 * self.values["num_layers"] * self.values[
            "num_attention_heads"
        ] * self.values["attention_head_dim"] * seq_len


def tiny_args():
    # Match the production head dimension, supported by both Magi and SDPA.
    return DuoModelArgs(
        values={
            **DUO_MEDIA_TRANSFORMER_CONFIG,
            "hidden_size": 256,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 128,
            "ffn_dim": 512,
            "freq_dim": 16,
            "time_embed_hidden_dim": 64,
            "time_embed_dim": 32,
            "rope_freq_dim": 4,
        }
    )
