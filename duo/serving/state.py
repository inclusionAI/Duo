"""One shared FIFO and the observable lifecycle of its actual work items."""

import copy
import json
import queue
import threading
import time

from .common import RUNTIME, WAIT_SILENCE_PROMPT, atomic
from .intent import DEFAULT_RATE, split_manual
from .cases import Cases
from .models import DEFAULT_MODEL, get_model, require_checkpoint
from .access import AccessQueue


class State:
    def __init__(self, stream_id, case_id="default", model_id=DEFAULT_MODEL):
        self.stream_id, self.case_id = stream_id, case_id
        self.model_id = get_model(model_id).id
        self.cases = Cases()
        self.case_settings = self.cases.settings(case_id)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.pending = queue.Queue()
        self.preparations = queue.Queue()
        self.handoff = None
        self.access = AccessQueue(self)
        self.cache_resets = 0
        self.requests, self.chunks = [], []
        self.context = dict(
            description="Preparing intent prefix", fixedPrefix="", sharedGroups=[], warmup=[]
        )
        self.context_variants = {}
        self.latest = None
        self.phase, self.detail, self.error = "starting", "Starting inference service", None
        self.gpus, self.clients = [], {}
        self.reset, self.last_reset = None, None
        self.case_change, self.last_case_change = None, None
        self.model_change, self.last_model_change = None, None
        self.wait_reuse = dict(
            enabled=False,
            threshold=3,
            consecutive=0,
            source_index=None,
            reused=False,
            pending=False,
        )
        self.wait_reuse_revision = 0
        self.wait_silence = dict(
            enabled=False,
            applied_enabled=None,
            applied_index=None,
            pending=False,
            prompt=WAIT_SILENCE_PROMPT,
        )
        self.global_prompt = dict(
            enabled=False, applied_enabled=None, applied_index=None, pending=False
        )
        self.wait_kv_window = dict(
            enabled=False, max_chunks=None, limit_chunks=None, retained_chunks=0, applied_index=None
        )
        self.folder = RUNTIME / "sessions" / stream_id
        self.request_folder = RUNTIME / "requests" / stream_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.journal = (self.folder / "events.jsonl").open("a", buffering=1)
        atomic(self.folder / "case.json", self.case_settings)
        self.record_model_selection()

    def log(self, event):
        self.journal.write(json.dumps(dict(time=time.time(), **event), ensure_ascii=False) + "\n")

    def service(self, phase, detail):
        with self.lock:
            self.phase, self.detail = phase, detail
            self.log(dict(type="service", phase=phase, detail=detail))

    def submit(self, text, author="Guest", rate=DEFAULT_RATE, split_mode="auto"):
        raise ValueError("Use the video page to record or upload a video.")

    def submit_recording(self, member, capture, path, metadata):
        with self.lock:
            self.access.check_capture(member, capture["sessionId"], capture["id"])
            request = dict(
                id=len(self.requests) + 1,
                kind="video",
                stream_id=self.stream_id,
                session_id=self.access.session_id,
                author=member["name"],
                model_id=self.model_id,
                recording_id=capture["id"],
                video_path=str(path),
                recording=metadata,
                state="queued",
                build=0,
                chunks=[],
                elapsed=0,
                transcript="",
                asr=dict(progress=0, segments=[]),
                submitted_at=time.time(),
                input_chunks=0,
            )
            self.requests.append(request)
            self.preparations.put(request)
            self.access.capture = None
            self.access.changed()
            self.log(
                dict(type="video_uploaded", request_id=request["id"], recording_id=capture["id"])
            )
            return dict(id=request["id"], recording_id=capture["id"])

    def input_event(self, event):
        with self.changed:
            if event["stream_id"] != self.stream_id:
                return
            request = next((r for r in self.requests if r["id"] == event["request_id"]), None)
            if not request or request.get("session_id") != self.access.session_id:
                return
            request.update(event["update"])
            if event["type"] == "input_ready":
                self.pending.put(request)
            if self.phase in ("ready", "working"):
                self.phase, self.detail = "working", "Processing user video"
            self.changed.notify_all()

    def _legacy_submit(self, text, author="Guest", rate=DEFAULT_RATE, split_mode="auto"):
        if not isinstance(text, str) or not text.strip() or len(text) > 1000:
            raise ValueError("Enter 1 to 1000 characters")
        if split_mode not in ("auto", "manual"):
            raise ValueError("Split mode must be auto or manual")
        text_chunks = split_manual(text) if split_mode == "manual" else None
        input_text = text
        if text_chunks is not None:
            text = "".join(text_chunks)
        rate = DEFAULT_RATE if split_mode == "manual" else float(rate)
        if not 2 <= rate <= 10:
            raise ValueError("Chunk rate must be between 2 and 10 characters per second")
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.reset and self.reset["mode"] == "all":
                raise RuntimeError("Reset in progress. Please wait.")
            if self.case_change:
                raise RuntimeError("Case change in progress. Please wait.")
            if self.model_change:
                raise RuntimeError("Model weights are changing. Please wait.")
            request = dict(
                id=len(self.requests) + 1,
                author=str(author).strip()[:20] or "Guest",
                text=text,
                model_id=self.model_id,
                input_text=input_text,
                split_mode=split_mode,
                rate=rate,
                state="queued",
                build=0,
                chunks=[],
                elapsed=0,
                submitted=time.strftime("%H:%M"),
                submitted_at=time.time(),
            )
            if text_chunks is not None:
                request["text_chunks"] = text_chunks
            self.requests.append(request)
            self.pending.put(request)
            self.log(dict(type="submitted", request=copy.deepcopy(request)))
            return copy.deepcopy(request)

    def request_reset(self, mode):
        if mode not in ("all", "cache"):
            raise ValueError("Reset mode must be all or cache")
        with self.lock:
            if (
                self.reset
                or self.case_change
                or self.model_change
                or self.phase not in ("ready", "working")
            ):
                raise RuntimeError("Wait for the current operation before resetting.")
            self.reset = dict(id=time.time_ns(), mode=mode, state="pending", requested=time.time())
            self.log(dict(type="reset_requested", **self.reset))
            return dict(self.reset)

    def configure_wait_reuse(self, enabled, threshold):
        if type(enabled) is not bool:
            raise ValueError("The condition reuse switch must be Boolean")
        if type(threshold) is not int or threshold < 1:
            raise ValueError("The condition reuse threshold must be positive")
        with self.lock:
            if (enabled, threshold) != (self.wait_reuse["enabled"], self.wait_reuse["threshold"]):
                self.wait_reuse_revision += 1
                self.wait_reuse.update(
                    enabled=enabled,
                    threshold=threshold,
                    consecutive=0,
                    source_index=None,
                    reused=False,
                    pending=True,
                )
                self.log(dict(type="wait_reuse_configured", **self.wait_reuse))
            return dict(self.wait_reuse)

    def request_case(self, values):
        with self.lock:
            if (
                self.reset
                or self.case_change
                or self.model_change
                or self.phase not in ("ready", "working")
            ):
                raise RuntimeError("Wait for the current operation before changing Case.")
            settings = self.cases.settings(values.get("case_id", self.case_id), values)
            if settings["config_id"] == self.case_settings["config_id"]:
                return dict(state="unchanged", settings=settings)
            self.case_change = dict(
                id=time.time_ns(),
                state="pending",
                requested=time.time(),
                target=settings["case_id"],
                settings=settings,
            )
            self.log(dict(type="case_change_requested", **self.case_change))
            return copy.deepcopy(self.case_change)

    def request_model(self, model_id):
        profile = get_model(model_id)
        with self.lock:
            if (
                self.reset
                or self.case_change
                or self.model_change
                or self.phase not in ("ready", "working")
            ):
                raise RuntimeError("Wait for the current operation before changing models.")
            if profile.id == self.model_id:
                return dict(state="unchanged", model_id=self.model_id)
            require_checkpoint(profile)
            self.model_change = dict(
                id=time.time_ns(),
                state="pending",
                requested=time.time(),
                source=self.model_id,
                target=profile.id,
            )
            self.log(dict(type="model_change_requested", **self.model_change))
            return copy.deepcopy(self.model_change)

    def take_model_change(self):
        with self.lock:
            if not self.model_change:
                return None
            self.model_change["state"] = "preparing"
            self.phase, self.detail = "switching_model", "Preparing model weights"
            return copy.deepcopy(self.model_change)

    def model_change_progress(self, detail):
        with self.lock:
            self.model_change["detail"] = self.detail = detail
            self.log(dict(type="model_change_progress", detail=detail))

    def record_model_selection(self, *, persist=False):
        selection = dict(model_id=self.model_id, model=get_model(self.model_id).public())
        atomic(self.folder / "model_selection.json", selection)
        if persist:
            atomic(RUNTIME / "model_settings.json", selection)

    def activate_model(self, stream_id, model_id):
        with self.lock:
            profile = get_model(model_id)
            self.new_stream(stream_id, reason="model_change")
            self.model_id = profile.id
            self.record_model_selection(persist=True)

    def model_finished(self, event, *, error=None):
        with self.changed:
            self.last_model_change = dict(
                event, state="failed" if error else "completed", finished=time.time()
            )
            if error:
                self.last_model_change["error"] = str(error)
            self.model_change = None
            active = any(r["state"] in ("building", "running") for r in self.requests)
            self.phase, self.detail = (
                ("working", "Generating media blocks") if active else ("ready", "Waiting for input")
            )
            self.log(
                dict(
                    type="model_change_failed" if error else "model_change_completed",
                    **self.last_model_change,
                )
            )
            atomic(self.folder / "last_model_change.json", self.last_model_change)
            self.changed.notify_all()

    def set_condition_contexts(self, conditions):
        with self.lock:
            self.context_variants = {
                "silence" if c["wait_silence"] else "normal": c["context"] for c in conditions
            }
            self.select_context()

    def prompt_source(self):
        return "original" if self.global_prompt["enabled"] else "finetuned"

    def select_context(self):
        context = self.context_variants.get("normal", self.context)
        source = self.prompt_source()
        prompt = context.get("sharedVariants", {}).get(source)
        self.context = dict(
            context,
            sharedSource=source,
            sharedGroups=prompt["groups"] if prompt else context.get("sharedGroups", []),
        )

    def configure_global_prompt(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("The intent-source switch must be Boolean")
        with self.lock:
            if (
                self.phase not in ("ready", "working")
                or self.reset
                or self.case_change
                or self.model_change
            ):
                raise RuntimeError(
                    "Wait for the current operation before changing the intent source."
                )
            self.global_prompt.update(
                enabled=enabled,
                pending=self.global_prompt["applied_index"] is not None
                and enabled != self.global_prompt["applied_enabled"],
            )
            self.select_context()
            self.log(dict(type="global_prompt_configured", **self.global_prompt))
            return dict(self.global_prompt)

    def configure_wait_silence(self, enabled):
        raise ValueError("This intent uses one streaming participation path.")

    def _legacy_configure_wait_silence(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("The silence switch must be Boolean")
        with self.lock:
            if (
                self.phase not in ("ready", "working")
                or self.reset
                or self.case_change
                or self.model_change
            ):
                raise RuntimeError("Wait for the current operation before changing the prompt.")
            self.wait_silence.update(
                enabled=enabled,
                pending=self.wait_silence["applied_index"] is not None
                and enabled != self.wait_silence["applied_enabled"],
            )
            self.select_context()
            self.log(dict(type="wait_silence_configured", **self.wait_silence))
            return dict(self.wait_silence)

    def configure_wait_kv_window(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("The short-window switch must be Boolean")
        with self.lock:
            if (
                self.phase not in ("ready", "working")
                or self.reset
                or self.case_change
                or self.model_change
            ):
                raise RuntimeError("Wait for the current operation before changing the window.")
            self.wait_kv_window["enabled"] = enabled
            self.log(dict(type="wait_kv_window_configured", **self.wait_kv_window))
            return dict(self.wait_kv_window)

    def intent_states_ready(self, item):
        with self.lock:
            chunk = self.chunks[item["index"]]
            chunk.update(item["chunk"])
            self.stage(
                dict(
                    type="stage",
                    request_id=item["request_id"],
                    index=item["index"],
                    stage="intent_states",
                    state="ready",
                    metrics=item["timing"],
                )
            )

    def select_intent_states(self, feature):
        with self.lock:
            item = dict(feature, global_prompt_source=self.prompt_source())
            item["chunk"] = dict(item["chunk"], global_prompt_source=item["global_prompt_source"])
            self.chunks[item["index"]].update(item["chunk"])
            self.requests[item["request_id"] - 1]["sequence_url"] = item["chunk"]["sequence_url"]
            self.wait_silence.update(
                applied_enabled=item["wait_silence"], applied_index=item["index"], pending=False
            )
            self.global_prompt.update(
                applied_enabled=item["global_prompt_source"] == "original",
                applied_index=item["index"],
                pending=False,
            )
            return item

    def take_case_change(self):
        with self.lock:
            if not self.case_change:
                return None
            self.case_change["state"] = "preparing"
            self.phase, self.detail = "switching", "Preparing Case"
            return copy.deepcopy(self.case_change)

    def activate_case(self, stream_id, settings, context):
        with self.lock:
            self.new_stream(stream_id, reason="case_change")
            self.case_id, self.case_settings = settings["case_id"], copy.deepcopy(settings)
            self.context = context
            self.cases.save(settings)
            atomic(self.folder / "case.json", self.case_settings)

    def case_finished(self, event):
        with self.changed:
            self.last_case_change = dict(event, state="completed", finished=time.time())
            self.case_change = None
            self.phase, self.detail = "ready", "Waiting for input"
            self.log(dict(type="case_change_completed", **self.last_case_change))
            atomic(self.folder / "last_case_change.json", self.last_case_change)
            self.changed.notify_all()

    def take_reset(self):
        with self.lock:
            if not self.reset:
                return None
            self.reset["state"] = "running"
            self.phase, self.detail = "resetting", "Resetting inference state"
            return dict(self.reset)

    def new_stream(self, stream_id, reason="full_reset"):
        with self.lock:
            for r in self.requests:
                if (
                    r["state"] == "running"
                    and r["chunks"]
                    and all(
                        c[s] == "consumed"
                        for c in r["chunks"]
                        for s in ("intent_states", "dit", "vae")
                    )
                ):
                    self.finish(r)
                if r["state"] != "completed":
                    r.update(state="cancelled", finished=time.time(), reason=reason)
            atomic(self.folder / "closed.json", self.snapshot())
            while True:
                try:
                    self.pending.get_nowait()
                    self.pending.task_done()
                except queue.Empty:
                    break
            self.journal.close()
            self.stream_id = stream_id
            self.requests, self.chunks, self.clients = [], [], {}
            self.cache_resets = 0
            self.wait_silence.update(applied_enabled=None, applied_index=None, pending=False)
            self.global_prompt.update(applied_enabled=None, applied_index=None, pending=False)
            self.latest = None
            self.folder = RUNTIME / "sessions" / stream_id
            self.request_folder = RUNTIME / "requests" / stream_id
            self.folder.mkdir(parents=True, exist_ok=True)
            self.journal = (self.folder / "events.jsonl").open("a", buffering=1)
            atomic(self.folder / "case.json", self.case_settings)
            self.record_model_selection()

    def reset_finished(self, event):
        with self.changed:
            self.last_reset = dict(event, state="completed", finished=time.time())
            self.reset = None
            if event["mode"] == "cache":
                self.cache_resets += 1
            active = any(r["state"] in ("building", "running") for r in self.requests)
            self.phase, self.detail = (
                ("working", "Generating media blocks") if active else ("ready", "Waiting for input")
            )
            self.log(dict(type="reset_completed", **self.last_reset))
            atomic(self.folder / "last_reset.json", self.last_reset)
            self.changed.notify_all()

    def building(self, request):
        with self.lock:
            request.update(state="building", started=request.get("started", time.time()))
            self.phase, self.detail = "working", "Building intent sequence"

    def planned(self, event):
        with self.lock:
            r = self.requests[event["request_id"] - 1]
            model = get_model(self.model_id)
            for info in event["chunks"]:
                c = dict(
                    info,
                    requestId=r["id"],
                    intent_states="pending",
                    dit="pending",
                    vae="pending",
                    step=0,
                    steps=model.steps,
                    model_id=model.id,
                )
                c["global"] = c.pop("global_index")
                if c["global"] < len(self.chunks):
                    self.chunks[c["global"]].update(info)
                    continue
                assert c["global"] == len(self.chunks)
                self.chunks.append(c)
                r["chunks"].append(c)
            r.update(state="running", build=100, sequence_url=event["sequence_url"])
            if self.phase in ("ready", "working"):
                self.phase, self.detail = "working", "Generating media blocks"
            self.log(event)

    def stage(self, event):
        with self.changed:
            c = self.chunks[event["index"]]
            c[event["stage"]] = event["state"]
            if "step" in event:
                c["step"] = event["step"]
            if "steps" in event:
                c["steps"] = event["steps"]
            if "metrics" in event:
                c[event["stage"] + "_metrics"] = event["metrics"]
            self.log(event)
            self.changed.notify_all()

    def published(self, event):
        with self.changed:
            self.latest = event["media"]
            self.chunks[event["index"]]["vae"] = "consumed"
            for request in self.requests:
                if (
                    request.get("ntp_done")
                    and request["state"] == "running"
                    and all(
                        c[s] == "consumed"
                        for c in request["chunks"]
                        for s in ("intent_states", "dit", "vae")
                    )
                ):
                    self.finish(request)
            self.log(event)
            atomic(self.folder / "latest.json", event["media"])
            self.changed.notify_all()
        self.prune_media()

    def ntp_ended(self, event):
        with self.changed:
            request = self.requests[event["request_id"] - 1]
            request.update(
                ntp_done=True,
                ntp_end=event["next_index"],
                ntp_end_reason=event.get("reason", "next_user_input"),
            )
            if event.get("reason") == "context_limit":
                request["notice"] = "The intent context is full. Reset the session to continue."
                # An unprocessed input plan must not remain permanently pending.
                keep = [c for c in request["chunks"] if c["global"] < event["next_index"]]
                if len(keep) != len(request["chunks"]):
                    del self.chunks[event["next_index"] :]
                    request["chunks"] = keep
            if all(
                c[s] == "consumed"
                for c in request["chunks"]
                for s in ("intent_states", "dit", "vae")
            ):
                self.finish(request)
            self.changed.notify_all()

    def prune_media(self):
        with self.lock:
            if not self.latest:
                return
            now = time.time()
            self.clients = {k: v for k, v in self.clients.items() if now - v["seen"] < 180}
            protected = {v["version"] for v in self.clients.values()}
            before = self.latest["version"] - 8
        for path in (RUNTIME / "results" / self.stream_id).glob("v*.mp4"):
            if path.stem[1:].isdigit():
                version = int(path.stem[1:])
                if version < before and version not in protected:
                    path.unlink(missing_ok=True)

    def finish(self, request):
        with self.lock:
            assert all(
                c[s] == "consumed"
                for c in request["chunks"]
                for s in ("intent_states", "dit", "vae")
            )
            request.update(
                state="completed",
                finished=time.time(),
                elapsed=time.time() - request.get("started", request["submitted_at"]),
            )
            self.log(
                dict(
                    type="completed",
                    request_id=request["id"],
                    queues={"intent_states": 0, "dit": 0, "vae": 0},
                    elapsed=request["elapsed"],
                )
            )
            atomic(self.request_folder / str(request["id"]) / "completed.json", request)

    def fail(self, error):
        with self.changed:
            self.phase, self.detail, self.error = "error", "Inference service error", str(error)
            for r in self.requests:
                if r["state"] in ("running", "building"):
                    r.update(state="failed", error=str(error))
            self.log(dict(type="error", error=str(error)))
            self.changed.notify_all()

    def snapshot(self, client=None, version=0):
        with self.lock:
            if client:
                self.clients[client] = dict(version=version, seen=time.time())
            requests = copy.deepcopy(self.requests)
            for r in requests:
                if r["state"] in ("running", "building"):
                    r["elapsed"] = time.time() - r["started"]
            return dict(
                stream_id=self.stream_id,
                case_id=self.case_id,
                phase=self.phase,
                detail=self.detail,
                model_id=self.model_id,
                model=get_model(self.model_id).public(),
                model_change=copy.deepcopy(self.model_change),
                last_model_change=copy.deepcopy(self.last_model_change),
                error=self.error,
                requests=requests,
                context=self.context,
                context_variants=self.context_variants,
                latest=self.latest,
                reset=self.reset,
                last_reset=self.last_reset,
                case_settings=self.case_settings,
                case_change=self.case_change,
                last_case_change=self.last_case_change,
                saved_case_settings=self.cases.saved,
                wait_reuse=dict(self.wait_reuse),
                wait_silence=dict(self.wait_silence),
                global_prompt=dict(self.global_prompt),
                wait_kv_window=dict(self.wait_kv_window),
                gpus=self.gpus,
                server_time=time.time(),
            )
