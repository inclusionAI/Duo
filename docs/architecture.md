# Architecture

Duo v1 couples a language-based **Intent Module** with a causal audiovisual
**Realization Module**. Intent provides the cognition underlying interaction;
Realization expresses that intent as coherent stereo audio and video.

## Unified chunk timeline

An interaction is divided into fixed-duration chunks. The released clock uses
17 video frames at 24 FPS, so one chunk spans `17 / 24 ≈ 708 ms`. The video VAE
maps those 17 frames to 5 latent frames. A chunk is a time interval and does not
need to align with a word, sentence, or conversational turn.

At chunk `k`, only user evidence available before its causal boundary may
affect the Intent decision. Intent emits `<speak>` followed by language or
`<wait>`, and selected hidden states from that executed output form the current
Realization condition.

```text
user video / ASR text ──> Intent ──> chunk intent states
                                                │
reference image ────────────────────────────────┼──> Realization
                                                │          │
committed assistant media history ─────────────┘          ├──> clean commitment
                                                           │        │
                                                           │        └──> next chunk history
                                                           │
                                                           └──> codec ──> publication
```

The paper formulation additionally sends encoded user audio and video directly
to Realization. That direct connection is not enabled by the currently
released Duo v1 offline checkpoint/code path; in this release, user input
affects Realization indirectly through Intent states. Direct user audiovisual
conditioning is planned for the next release.

## Causal visibility contract

The compact training sequence uses the physical row order

```text
[reference] [shared Intent states]
[clean assistant media 0 ... n-1]       # training
[clean assistant media 0 ... k-1]       # inference prefix
[Intent states 0] [noisy assistant media 0]
[Intent states 1] [noisy assistant media 1] ...
```

with these rules:

- reference and shared rows only see their own groups;
- chunk-`k` Intent states see the reference, shared condition, completed clean
  assistant media before `k`, and their own group;
- clean assistant media additionally see their own clean rows;
- noisy assistant media see completed clean history, their chunk's Intent
  states, and their own noisy rows, but not the same chunk's clean targets or
  any future chunk.

This separates generation from commitment. Denoising may use masked refinement
internally, while the next chunk receives only completed clean history.

Some internal Python fields and legacy checkpoint keys still use `block` as a
compatibility identifier. Public documentation and UI use `chunk`; changing
those stable serialized names would break existing checkpoints and clients.

## Codec-aware publication

Latent completion and media publication are separate events. A streaming codec
can require future latent context to reconstruct earlier media. Consequently,
`MediaCodec.push()` may return zero, one, or multiple publishable chunks, and
`flush()` completes the tail at end of stream.

## Offline and online execution

The `master` branch provides deterministic offline inference: it receives a
completed recording, preprocesses it, and executes Intent, Realization, and
codec stages successively. This makes model and data-flow validation
reproducible, but it is not the paper's low-latency online scheduler.

The online implementations overlap work across adjacent chunks while preserving
causal evidence and commit ordering. They are maintained separately on the
[`inference/vllm-omni-duo`](https://github.com/inclusionAI/Duo/tree/inference/vllm-omni-duo)
and
[`inference/sglang-omni-duo`](https://github.com/inclusionAI/Duo/tree/inference/sglang-omni-duo)
branches.
