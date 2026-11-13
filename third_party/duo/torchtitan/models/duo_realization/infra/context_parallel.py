"""CPU-only Realization topology and Ulysses head-padding policy; no process groups."""

from dataclasses import dataclass

SUPPORTED_CP = (1, 2, 4, 8, 16, 32)
POLICY = "realization_ulysses_head_pad_v1"


def _positive(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(name + " must be a positive integer")


def padded_head_count(heads, cp):
    _positive("attention heads", heads)
    if not isinstance(cp, int) or isinstance(cp, bool) or cp not in SUPPORTED_CP:
        raise ValueError("Realization CP must be one of 1, 2, 4, 8, 16, 32")
    return ((heads + cp - 1) // cp) * cp


@dataclass(frozen=True)
class ContextParallelPlan:
    world_size: int
    local_world_size: int
    cp: int
    dp_replicate: int
    dp_shard: int
    attention_heads: int
    communication_heads: int
    policy: str = POLICY

    @property
    def nodes(self):
        return self.world_size // self.local_world_size

    @property
    def data_world(self):
        return self.dp_replicate * self.dp_shard

    def summary(self):
        from dataclasses import asdict

        return dict(
            asdict(self),
            nodes=self.nodes,
            data_world=self.data_world,
            nodes_per_cp=max(1, self.cp // self.local_world_size),
            fsdp_shard_size=self.dp_shard * self.cp,
            heads_per_cp_rank=self.communication_heads // self.cp,
        )


def resolve_parallelism(
    *, world_size, local_world_size, cp, heads, dp_replicate=-1, dp_shard=-1
):
    """Resolve only -1 values; explicit DP dimensions are never overwritten.

    With both DP dimensions automatic, shard within one node for CP<=local,
    and across one contiguous CP group for CP>local. Replicate the remaining
    groups. A single automatic dimension is derived from the explicit one.
    """
    _positive("world size", world_size)
    _positive("GPUs per node", local_world_size)
    padded = padded_head_count(heads, cp)
    if world_size % local_world_size:
        raise ValueError(
            "Realization requires homogeneous nodes: world size must divide into GPUs per node"
        )
    if world_size % cp:
        raise ValueError("Realization CP must divide total GPUs")
    if max(cp, local_world_size) % min(cp, local_world_size):
        raise ValueError("Realization CP and GPUs per node must tile each other")
    for name, value in (("DP replicate", dp_replicate), ("DP shard", dp_shard)):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or (value != -1 and value < 1)
        ):
            raise ValueError(name + " must be -1 (automatic) or a positive integer")
    if dp_replicate == dp_shard == -1:
        dp_replicate = world_size // max(local_world_size, cp)
    if dp_replicate == -1:
        divisor = dp_shard * cp
        if world_size % divisor:
            raise ValueError("Explicit DP shard * CP must divide total GPUs")
        dp_replicate = world_size // divisor
    if dp_shard == -1:
        divisor = dp_replicate * cp
        if world_size % divisor:
            raise ValueError("Explicit DP replicate * CP must divide total GPUs")
        dp_shard = world_size // divisor
    if dp_replicate < 1 or dp_shard < 1 or dp_replicate * dp_shard * cp != world_size:
        raise ValueError(
            "Realization topology requires DP replicate * DP shard * CP = total GPUs"
        )
    return ContextParallelPlan(
        world_size, local_world_size, cp, dp_replicate, dp_shard, heads, padded
    )
