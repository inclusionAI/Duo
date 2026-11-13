"""Pipeline configuration for staged Duo generation."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from sglang_omni.config import FactoryArgs, PipelineConfig, PlacementConfig, StageConfig

PACKAGE = "sglang_omni.models.duo"


class DuoIntentStageArgs(FactoryArgs):
    max_blocks: int = Field(default=360, gt=0)
    max_open_sessions: int = Field(default=8, gt=0)


class DuoRealizationStageArgs(FactoryArgs):
    transformer_directory: str
    raven_lora_path: str
    trainables_path: str
    clean_timestep: float = Field(default=0.999, gt=0, le=1)
    capacity_tokens: int = Field(default=16_384, gt=0)
    plan_capacity_tokens: int = Field(default=512, gt=0)
    denoise_timesteps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    max_open_sessions: int = Field(default=2, gt=0)
    use_flash_attention: bool = False
    flash_num_splits: int = Field(default=1, ge=0)
    precompute_adaln: bool = False
    enable_cuda_graph: bool = False


class DuoPipelineConfig(PipelineConfig):
    """Intent/H50 session state → causal Realization session → latent stream."""

    architecture: ClassVar[str] = "Duo"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {}
    model_path: str = "Duo"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = [
        StageConfig(
            name="intent_h50",
            process="intent_h50",
            factory_path=f"{PACKAGE}.stages.create_intent_executor",
            factory=DuoIntentStageArgs(),
            next="realization",
        ),
        StageConfig(
            name="realization",
            process="realization",
            factory_path=f"{PACKAGE}.stages.create_realization_session_executor",
            factory=DuoRealizationStageArgs(
                transformer_directory="Duo/FL2VA/transformer",
                raven_lora_path="duo_raven_streaming_lora_4nfe.safetensors",
                trainables_path="trainables.safetensors",
            ),
            gpu=0,
            next="latent",
        ),
        StageConfig(
            name="latent",
            process="latent",
            factory_path=f"{PACKAGE}.stages.create_latent_sink_executor",
            terminal=True,
        ),
    ]


EntryClass = DuoPipelineConfig
