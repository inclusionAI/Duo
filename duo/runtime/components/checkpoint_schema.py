"""Compatibility helpers for the stable Duo checkpoint schema.

Intent/Realization are the public component names. Released checkpoints may
still use their original planner/generator storage keys; those keys are part of
the serialized format and do not need to be rewritten when terminology changes.
"""


_IDENTITY_KEYS = {
    "intent": ("intent", "planner"),
    "realization": ("realization", "generator"),
}


def component_identity(manifest, component):
    """Return a component identity from either the current or legacy key."""

    if component not in _IDENTITY_KEYS:
        raise ValueError(f"Unknown Duo component: {component}")
    identities = manifest.get("identity", {})
    if not isinstance(identities, dict):
        raise ValueError("Realization checkpoint has an invalid identity manifest")
    for key in _IDENTITY_KEYS[component]:
        if key in identities:
            value = identities[key]
            if not isinstance(value, dict):
                raise ValueError(f"Invalid {component} checkpoint identity")
            return value
    raise ValueError(f"Realization checkpoint is missing its {component} identity")


def intent_identity(manifest):
    return component_identity(manifest, "intent")


def realization_identity(manifest):
    return component_identity(manifest, "realization")


def resolve_family(manifest, requested):
    """Resolve a public model family to the key physically stored in DCP."""

    models = manifest.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("Realization checkpoint is missing model metadata")
    if requested in models:
        return requested
    if requested == "realization" and "generator" in models:
        return "generator"
    raise ValueError(f"Realization checkpoint does not contain family {requested!r}")


def public_family(stored):
    """Keep legacy storage terminology out of runtime/API provenance."""

    return "realization" if stored == "generator" else stored
