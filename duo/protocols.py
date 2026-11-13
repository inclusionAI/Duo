"""Vendor-neutral component contracts for Duo inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True, slots=True)
class UserBlock:
    """User evidence available to Intent for one clock chunk.

    The class name is retained for compatibility with the initial public API.
    """

    index: int
    video: Any
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IntentContext:
    """Causal Intent view at one chunk boundary."""

    block_index: int
    current_user: UserBlock | None
    user_history: Sequence[UserBlock]
    action_history: Sequence[str]


@dataclass(frozen=True, slots=True)
class Plan:
    """Executed intent action and the states selected to condition media."""

    action: str
    condition: Any
    text: str = ""
    stop: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeneratedBlock:
    """A completed latent chunk and its clean history representation."""

    latent: Any
    committed: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecodedBlock:
    """One publishable media chunk returned by a codec adapter."""

    index: int
    video: Any
    audio: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class IntentModel(Protocol):
    """Produces an action and chunk-aligned media condition."""

    def plan(self, context: IntentContext) -> Plan: ...


@runtime_checkable
class RealizationModel(Protocol):
    """Generates one joint audio-video latent chunk."""

    def generate(
        self,
        *,
        block_index: int,
        condition: Any,
        committed_history: Sequence[Any],
        seed: int,
    ) -> GeneratedBlock: ...


@runtime_checkable
class MediaCodec(Protocol):
    """Decodes committed latents and may delay publication for context."""

    def push(self, block_index: int, latent: Any) -> Sequence[DecodedBlock]: ...

    def flush(self) -> Sequence[DecodedBlock]: ...
