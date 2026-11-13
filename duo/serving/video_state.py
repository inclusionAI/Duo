"""Adapt actual inference state to the approved video dashboard."""

import copy
import time

SECONDS = 17 / 24


def video_url(value):
    return "../" + value[2:] if value and value.startswith("./") else value


def chunk_view(chunk):
    index = chunk["global"]
    metrics = chunk.get("dit_metrics") or {}
    reuse = metrics.get("wait_reuse") or {}
    gate = chunk.get("user_speech_gate") or {}
    mapping = dict(chunk["user"])
    mapping["url"] = video_url(mapping.get("url"))
    private = {
        "intent_states",
        "dit",
        "vae",
        "intent_states_metrics",
        "dit_metrics",
        "vae_metrics",
        "intent_state_rows",
        "intent_state_strategy",
        "input_rows",
        "raw_token_ids",
    }
    public = {key: value for key, value in chunk.items() if key not in private}
    return dict(
        public,
        requestId=f"R{chunk['requestId']:03d}",
        user=mapping,
        intent=chunk["intent_states"],
        realization=chunk["dit"],
        codec=chunk["vae"],
        timelineStart=index * SECONDS,
        timelineEnd=(index + 1) * SECONDS,
        realizationSteps=chunk["steps"],
        intentRows=chunk.get("intent_state_rows", []),
        conditionSource="base" if chunk.get("global_prompt_source") == "original" else "adapted",
        waitReuseSource=reuse.get("intent_state_source_index") if reuse.get("reused") else None,
        waitStreak=reuse.get("consecutive", 0),
        historyWindow=metrics.get("wait_kv_window", {}).get("limit_chunks", 14),
        sequenceURL=video_url(chunk.get("sequence_url")),
        userSpeechGate=dict(
            policy=gate.get("policy"),
            firstUserStartMs=gate.get("first_user_start_ms"),
            firstUserEndMs=gate.get("first_user_end_ms"),
            firstUserStartBlock=gate.get("first_user_start_block"),
            firstUserEndBlock=gate.get("first_user_end_block"),
            releaseBlock=gate.get("release_block"),
            applied=gate.get("applied", False),
            decisionSource=gate.get("decision_source"),
        ),
    )


def snapshot(state, *, authenticated=True, client=None, version=0):
    with state.lock:
        access = state.access
        people = access.public()
        actual = state.snapshot(client, version)
        hidden = not authenticated or access.hidden_stream == state.stream_id or bool(state.handoff)
        requests, chunks = [], []
        if not hidden:
            chunks = [chunk_view(chunk) for chunk in state.chunks]
            for request in actual["requests"]:
                if request.get("kind") != "video":
                    continue
                requests.append(
                    dict(
                        id=f"R{request['id']:03d}",
                        recordingId=request["recording_id"],
                        author=request["author"],
                        createdAt=request["submitted_at"] * 1000,
                        state=request["state"],
                        recording=request["recording"],
                        transcript=request.get("transcript", ""),
                        asr=request["asr"],
                        build=request["build"],
                        chunks=[c["global"] for c in request["chunks"]],
                        sequence=None,
                        sequenceURL=video_url(request.get("sequence_url")),
                        sequenceVersion=sum(
                            c["intent_states"] in ("ready", "consumed") for c in request["chunks"]
                        ),
                        inputChunks=request.get("input_chunks", 0),
                        error=request.get("error"),
                        notice=request.get("notice"),
                        recordingDownloadURL=f"../recordings/{request['stream_id']}/{request['recording_id']}/download.mp4",
                        recordingURL=f"../recordings/{request['stream_id']}/{request['recording_id']}/original"
                        + (
                            ".mp4"
                            if request["recording"]["mimeType"].startswith("video/mp4")
                            else ".webm"
                        ),
                    )
                )
        media = copy.deepcopy(actual["latest"]) if not hidden else None
        if media:
            media["url"] = video_url(media["url"])
        phase = (
            "preparing"
            if state.handoff or state.reset or state.case_change or state.model_change
            else state.phase
        )
        if phase in ("ready", "working"):
            phase = (
                "recording"
                if access.capture
                else next(
                    (
                        r["state"]
                        for r in reversed(requests)
                        if r["state"] not in ("completed", "no_speech", "failed", "cancelled")
                    ),
                    requests[-1]["state"] if requests else "idle",
                )
            )
        return dict(
            people,
            revision=time.time_ns() // 1000,
            serverTime=time.time() * 1000,
            config=dict(
                modelId=state.model_id,
                caseId=state.case_id,
                cases=state.cases.frontend(),
                descriptions={
                    **{key: value["description"] for key, value in state.cases.saved.items()},
                    state.case_id: state.case_settings["description"],
                },
                originalPrompt=state.global_prompt["enabled"],
                waitReuse=state.wait_reuse["enabled"],
                waitThreshold=state.wait_reuse["threshold"],
                waitKVWindow=state.wait_kv_window["enabled"],
            ),
            ready=not hidden
            and not state.error
            and state.phase in ("ready", "working")
            and not (state.handoff or state.reset or state.case_change or state.model_change),
            detail=state.detail,
            error=state.error,
            context=copy.deepcopy(state.context),
            gpus=state.gpus,
            session=dict(
                id=access.session_id,
                createdAt=access.created_at,
                revision=access.revision,
                requests=requests,
                chunks=chunks,
                media=media,
                capture=copy.deepcopy(access.capture),
                demo=None,
                phase=phase,
                cacheResets=state.cache_resets,
            ),
        )
