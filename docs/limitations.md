# Current Limitations

- Realization and Codec retain session-key fences for stale request protection. The
  default `max_retired_sessions` is 4096 active plus retired keys; once full,
  new keys are rejected until the engine is restarted. Interruptions and new
  epochs can consume keys. Removing fences is not a safe workaround.
- Use an FP8 attention profile that matches the checkpoint, tensor-parallel
  layout, geometry, and sampler.
- Keep captured conversations and calibration inputs private unless their
  owners have explicitly approved publication.
