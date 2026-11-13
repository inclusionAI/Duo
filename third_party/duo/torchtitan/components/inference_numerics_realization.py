"""Inference-only linear call shapes for full-model context parallel replicas."""

import torch
from torch import nn

POLICY = "global_rows_linear_calls_v1"


def install_cp_linear_shapes(model, cp):
    """Pad independent rowwise GEMMs to the same M as CP=1, then crop.

    BF16 GEMM algorithm selection depends on M. Padding never reaches attention
    or the sequence/mask: CP still partitions attention heads and retained KV.
    Parameter names/dtypes and the solver remain unchanged.
    """
    if cp <= 1:
        return
    from peft.tuners.lora.layer import LoraLayer

    handles = []
    for block in model.transformer_blocks:
        state = dict(local=None, global_rows=None)

        def begin(module, inputs, state=state):
            state["local"] = inputs[0].shape[1]
            state["global_rows"] = state["local"] * cp

        def finish(module, inputs, output, state=state):
            state["local"] = state["global_rows"] = None

        handles.append(block.register_forward_pre_hook(begin))
        handles.append(block.register_forward_hook(finish, always_call=True))
        for module in block.modules():
            if not isinstance(module, (nn.Linear, LoraLayer)):
                continue
            stack = []

            def pad(module, inputs, state=state, stack=stack):
                x = inputs[0] if inputs else None
                rows = None
                if (
                    state["local"] is not None
                    and isinstance(x, torch.Tensor)
                    and x.ndim == 3
                    and x.shape[1] == state["local"]
                ):
                    rows = x.shape[1]
                    extra = x.new_zeros(
                        x.shape[0], state["global_rows"] - rows, x.shape[2]
                    )
                    inputs = (torch.cat((x, extra), dim=1), *inputs[1:])
                stack.append(rows)
                return inputs

            def crop(module, inputs, output, stack=stack):
                rows = stack.pop()
                if rows is not None and output is not None:
                    return output[:, :rows].contiguous()
                return output

            handles.append(module.register_forward_pre_hook(pad))
            handles.append(module.register_forward_hook(crop, always_call=True))
    model._realization_inference_linear_handles = handles
    model._realization_inference_cp_numerics = POLICY
