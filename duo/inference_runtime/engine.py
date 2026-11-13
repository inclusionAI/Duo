"""Resident clean-derived DiT on GPU1; GPU0 produces intent states and decodes in parallel."""

from concurrent.futures import Future
from multiprocessing import Pipe
import copy
import os
import queue
import socket
import subprocess
import threading
import time

from ..serving.common import (
    ROOT,
    RUNTIME,
    LOCAL,
    atomic,
    read,
    environment,
    native_imports,
    clock_block,
    new_stream_id,
)
from ..serving.resources import Resources
from ..serving.models import get_model
from ..preprocessing import InputPreparation
from ..runtime.components.checkpoint_schema import intent_identity, resolve_family


def _loaded_parameter_layout(model, entry):
    """Validate the resident parameter form and return its LoRA module count."""

    if model._duo_mode != entry["mode"]:
        raise ValueError("Loaded Realization parameter mode differs from the manifest")
    if model._duo_lora != entry["lora"]:
        raise ValueError("Loaded Realization LoRA layout differs from the manifest")
    if model.role_embedding_enabled != entry["role_embedding"]:
        raise ValueError("Loaded Realization role embedding differs from the manifest")
    if entry["mode"] == "full":
        if model._duo_lora is not None:
            raise ValueError("Full Realization weights must not have LoRA adapters")
        return 0
    if entry["mode"] != "lora" or len(model._duo_lora["modules"]) != 317:
        raise ValueError("Released LoRA weights must contain 317 adapter modules")
    return len(model._duo_lora["modules"])


class GPU0:
    def __init__(self, state, resources, config):
        self.state = state
        self.ready = Future()
        self.features = queue.Queue()
        self.condition_updates = queue.Queue()
        self.resets = queue.Queue()
        self.case_changes = queue.Queue()
        self.conditions = {}
        self.failure = None
        self.closing = False
        # Use a bidirectional control channel between the two GPU workers.
        self.connection, child = Pipe(True)
        self.send_lock = threading.Lock()
        self.log = (RUNTIME / "gpu0.log").open("ab")
        command = [
            config.executable(config.executables.intent_python),
            "-B",
            "-u",
            "-m",
            "duo.serving.intent_codec_worker",
            "--fd",
            str(child.fileno()),
            "--stream-id",
            state.stream_id,
            "--case",
            state.case_id,
            "--settings",
            str(state.folder / "case.json"),
            "--config",
            str(config.config_path),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment("intent", config.inference.intent_gpu, settings=config),
            pass_fds=(child.fileno(),),
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        child.close()
        resources.child(self.process.pid)
        threading.Thread(target=self.receive, daemon=True, name="gpu0-output").start()

    def receive(self):
        try:
            while True:
                event = self.connection.recv()
                kind = event["type"]
                if kind == "ready":
                    self.conditions = {c["version"]: c for c in event["conditions"]}
                    self.state.set_condition_contexts(event["conditions"])
                    self.ready.set_result(event)
                elif kind == "planned":
                    self.state.planned(event)
                elif kind == "intent_states":
                    self.state.intent_states_ready(event)
                    self.features.put(event)
                elif kind == "ntp_request_end":
                    self.state.ntp_ended(event)
                elif kind == "conditions":
                    self.conditions.update({c["version"]: c for c in event["conditions"]})
                    self.state.set_condition_contexts(event["conditions"])
                    self.condition_updates.put(event)
                elif kind == "stage":
                    self.state.stage(event)
                elif kind == "media":
                    self.state.published(event)
                elif kind == "reset_done":
                    self.resets.put(event)
                elif kind == "case_ready":
                    self.case_changes.put(event)
                elif kind == "service":
                    self.state.service(event["phase"], event["detail"])
                elif kind == "fatal":
                    raise RuntimeError(event["error"])
        except (EOFError, OSError):
            if not self.closing:
                self.fail(RuntimeError("GPU0 worker disconnected"))
        except BaseException as exc:
            self.fail(exc)

    def fail(self, error):
        if self.failure or self.closing:
            return
        self.failure = error
        if not self.ready.done():
            self.ready.set_exception(error)
        for target in (self.features, self.condition_updates, self.resets, self.case_changes):
            target.put(error)
        self.state.fail(error)

    def send(self, action, **values):
        if self.failure:
            raise self.failure
        with self.send_lock:
            self.connection.send(dict(action=action, **values))

    @staticmethod
    def result(target, timeout=None):
        result = target.get(timeout=timeout)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        self.closing = True
        if self.process.poll() is None:
            try:
                self.connection.send(dict(action="close"))
                self.process.wait(timeout=20)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        self.connection.close()
        self.log.close()


def native_args(state, config, model_id=None):
    native_imports()
    from duo.runtime.components.inference_config import parser, validate

    profile = get_model(model_id or state.model_id)
    args = parser(profile.stage).parse_args(
        [
            "--checkpoint",
            str(profile.checkpoint),
            "--checkpoint-family",
            profile.family,
            "--checkpoint-cache",
            str(LOCAL / "dit/checkpoints"),
            "--checkpoint-reserve-gb",
            "100",
            "--exp-name",
            state.stream_id,
            "--local-dir",
            str(LOCAL / "dit" / state.stream_id),
            "--output-dir",
            str(RUNTIME / "results" / state.stream_id),
            "--data-config",
            str(config.case_catalog),
            "--feature-config",
            "",
            "--sample-id",
            state.case_id,
            "--decode",
            "duo_5_17",
            "--tail-policy",
            "pad",
            "--intent",
            config.resolve(config.weights.intent),
            "--intent-adapter",
            config.resolve(config.weights.intent_adapter),
            "--intent-protocol-manifest",
            config.resolve(config.weights.intent_protocol),
            "--duo",
            config.resolve(config.weights.codec),
            "--sliding-window-size",
            "238",
            "--max-seq-len",
            "0",
            "--context-parallel-degree",
            "1",
            "--height",
            str(config.inference.height),
            "--width",
            str(config.inference.width),
            "--attention",
            config.inference.attention,
            "--dtype",
            config.inference.dtype,
            "--steps",
            str(profile.steps),
            "--flow-shift",
            str(profile.flow_shift),
            "--seed",
            str(config.inference.seed),
            "--ffmpeg",
            config.executable(config.executables.ffmpeg),
        ]
    )
    args = validate(args)
    args.rope, args.adaln_timestep = "new", "new"
    args.effective_intent_states_timestep = "fixed_1"
    args.intent_state_strategy = "ntp_forward_layer50_v1"
    args.assistant_text_source = "intent_generated_history"
    return args


class Engine:
    def __init__(self, state, config):
        self.state = state
        self.config = config
        self.resources = Resources(config)
        self.stopping = threading.Event()
        self.gpu0 = None
        self.request = None
        self.input_gate = threading.RLock()
        self.inputs = InputPreparation(state, self.resources, self.stopping, config)
        self.feeder = threading.Thread(target=self.feed_inputs, name="intent-input", daemon=True)
        self.reset_wait_reuse()
        self.thread = threading.Thread(target=self.run, name="dit-engine", daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        try:
            while not self.stopping.is_set():
                try:
                    self.resources.acquire()
                    break
                except BlockingIOError:
                    if self.state.phase != "waiting_gpus":
                        self.state.service("waiting_gpus", "Waiting for the configured GPUs")
                    self.stopping.wait(1)
            if self.stopping.is_set():
                return
            self.gpu0 = GPU0(self.state, self.resources, self.config)
            native_imports()
            import torch
            from duo.runtime.components import inferencer as native

            self.torch = torch
            torch.set_num_threads(4)
            self.args = native_args(self.state, self.config)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
            profile = get_model(self.state.model_id)
            self.state.service("loading_dit", "Loading " + profile.label)
            self.native = native.DuoRuntime(self.args)
            self.model, self.source, _ = native.load_model(
                self.args, self.native.dims, self.native.device
            )
            setup = self.gpu0.ready.result()
            assert setup["intent"] == self.args.checkpoint_intent
            manifest = read(profile.checkpoint / "complete.json")
            storage_family = resolve_family(manifest, profile.family)
            self.lora_modules = _loaded_parameter_layout(
                self.model, manifest["models"][storage_family]
            )
            self.select_initial_prompt(setup["conditions"])
            self.reference = torch.from_numpy(setup["reference"]).bfloat16()
            self.loaded_at = time.time()
            self.backbone_loaded_at = self.loaded_at
            self.new_session()
            self.state.record_model_selection(persist=True)
            self.inputs.thread.start()
            self.feeder.start()
            while not self.stopping.is_set():
                self.apply_handoff()
                self.apply_model_change()
                self.apply_case_change()
                self.apply_reset()
                try:
                    feature = self.gpu0.result(self.gpu0.features, timeout=0.5)
                except queue.Empty:
                    if self.gpu0.failure:
                        raise self.gpu0.failure
                    with self.state.lock:
                        working = any(
                            r["state"] in ("queued", "asr", "building", "running")
                            for r in self.state.requests
                        )
                    if not working:
                        if self.state.phase != "ready":
                            self.state.service(
                                "ready",
                                "Waiting for the next video"
                                if self.state.requests
                                else "Waiting for a recording",
                            )
                        self.resources.idle()
                    continue
                self.resources.active()
                self.run_chunk(feature)
                self.request = None
        except BaseException as exc:
            import traceback

            traceback.print_exc()
            if not self.stopping.is_set():
                self.state.fail(f"{type(exc).__name__}: {exc}")
        finally:
            self.stopping.set()
            if self.inputs.thread.ident is not None:
                self.inputs.close()
            if self.gpu0:
                self.gpu0.close()
            if hasattr(self, "session"):
                self.session.kv = None
                self.session.model = None
                self.model = None
                import gc

                gc.collect()
                self.torch.cuda.empty_cache()
            self.resources.close()

    def set_shared_prompt(self, version, source):
        if source not in self.gpu0.conditions[version]["shared_variants"]:
            self.gpu0.send("original_prompt")
            self.gpu0.result(self.gpu0.condition_updates)
        public = self.gpu0.conditions[version]["shared_variants"][source]
        self.shared = self.torch.from_numpy(public["shared"]).bfloat16()
        self.shared_groups = public["shared_groups"]
        self.condition_version, self.global_prompt_source = version, source

    def select_initial_prompt(self, conditions):
        selected = conditions[0]
        self.set_shared_prompt(selected["version"], self.state.prompt_source())

    def feed_inputs(self):
        """Deliver prepared User input independently of the DiT consumer."""
        while not self.stopping.is_set():
            try:
                request = self.state.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if request is None:
                    return
                with self.input_gate:
                    with self.state.lock:
                        valid = (
                            request["stream_id"] == self.state.stream_id
                            and request["session_id"] == self.state.access.session_id
                            and not self.state.handoff
                        )
                        if valid:
                            self.state.building(request)
                    if valid:
                        self.resources.active()
                        self.gpu0.send(
                            "request",
                            request={
                                key: request[key]
                                for key in ("id", "author", "prepared", "stream_id")
                            },
                        )
            except Exception as error:
                if not self.stopping.is_set():
                    self.state.fail(error)
                return
            finally:
                self.state.pending.task_done()

    def apply_handoff(self):
        with self.state.lock:
            operation = copy.deepcopy(self.state.handoff)
        if operation is None:
            return False
        with self.state.lock:
            self.state.model_change = self.state.case_change = None
            self.state.reset = dict(
                id=time.time_ns(), mode="all", state="pending", requested=time.time()
            )
        self.apply_reset()
        if operation["case_id"] and self.state.case_id != operation["case_id"]:
            with self.state.lock:
                settings = self.state.cases.settings(
                    operation["case_id"], dict(opening_chunk_ids=[])
                )
                self.state.case_change = dict(
                    id=time.time_ns(),
                    state="pending",
                    requested=time.time(),
                    target=operation["case_id"],
                    settings=settings,
                )
            self.apply_case_change()
        if operation["model_id"] and self.state.model_id != operation["model_id"]:
            with self.state.lock:
                self.state.model_change = dict(
                    id=time.time_ns(),
                    state="pending",
                    requested=time.time(),
                    source=self.state.model_id,
                    target=operation["model_id"],
                )
            if not self.apply_model_change():
                raise RuntimeError("The requested model profile is unavailable.")
        with self.state.lock:
            if self.state.handoff and self.state.handoff["id"] == operation["id"]:
                self.state.handoff = None
                self.state.access.hidden_stream = None
                self.state.access.changed()
        return True

    def new_session(self):
        from duo.models.realization.pipelines.streaming import MediaSession

        self.reset_wait_reuse()
        cache = dict(
            sample_id=self.state.case_id,
            reference=self.reference,
            codec=self.args.checkpoint_codec,
            blocks=[],
        )
        self.session = MediaSession(
            self.model, cache, copy.copy(self.args), self.native.device, sampler=self.sample
        )
        self.cache_start, self.cache_epoch = 0, 0
        self.reset_wait_kv_window_status()
        atomic(
            self.state.folder / "model.json",
            dict(
                source=self.source,
                args=vars(self.args),
                model=get_model(self.state.model_id).public(),
                backbone_loaded=self.backbone_loaded_at,
                parameter_bytes=sum(p.numel() * p.element_size() for p in self.model.parameters()),
                gpu=1,
                cp=1,
                lora_modules=self.lora_modules,
                shared_intent_state_rows=len(self.shared),
                global_prompt_source=self.global_prompt_source,
                chunk_intent_state_source="finetuned_intent_v2",
                intent_layer=50,
                history_chunks=self.args.sliding_window_size // self.session.geometry.frames,
                loaded=self.loaded_at,
            ),
        )

    def apply_model_change(self):
        operation = self.state.take_model_change()
        if operation is None:
            return False
        from ..serving.model_weights import read_weights, install_weights, WeightSwapError

        self.resources.active()
        boundary = self.session.next_block
        self.wait_published(boundary - 1)
        profile = get_model(operation["target"])
        self.state.model_change_progress("Reading and validating " + profile.label)
        # Disk, metadata and tensor checks complete before any live weight is
        # touched. A failed read leaves the old stream and its requests intact.
        try:
            args = native_args(self.state, self.config, profile.id)
            values, source, manifest = read_weights(self.model, profile)
            identity = manifest["identity"]
            if (
                intent_identity(manifest) != self.args.checkpoint_intent
                or identity.get("codec") != self.args.checkpoint_codec
            ):
                raise ValueError("Target checkpoint requires a different Intent or Codec")
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)) or self.stopping.is_set():
                raise
            self.state.model_finished(operation, error=error)
            return False
        self.state.model_change_progress("Loading " + profile.label)
        try:
            install_weights(self.model, values)
        except WeightSwapError as error:
            self.state.model_finished(operation, error=error)
            return False
        del values
        old_stream, stream_id = self.state.stream_id, new_stream_id()
        released = self.session.kv.nbytes if self.session.kv is not None else 0
        self.session.kv = None
        self.state.model_change_progress("Starting a new media stream")
        # Reuse the established full-reset protocol: both Intent suffix queues
        # and VAE/encoder state are drained, while weights/reference stay loaded.
        self.gpu0.send("reset", mode="all", stream_id=stream_id)
        details = self.gpu0.result(self.gpu0.resets)
        details.pop("type")
        while True:
            try:
                self.gpu0.features.get_nowait()
            except queue.Empty:
                break
        self.state.activate_model(stream_id, profile.id)
        args.exp_name = stream_id
        args.local_dir = str(LOCAL / "dit" / stream_id)
        args.output_dir = str(RUNTIME / "results" / stream_id)
        args.checkpoint_intent = intent_identity(manifest)
        args.checkpoint_codec = self.args.checkpoint_codec
        self.args, self.source, self.loaded_at = args, source, time.time()
        self.native.args = args
        self.new_session()
        info = read(RUNTIME / "server_info.json")
        atomic(RUNTIME / "server_info.json", dict(info, stream_id=stream_id, model_id=profile.id))
        self.state.model_finished(
            dict(
                operation,
                old_stream=old_stream,
                stream_id=stream_id,
                after_chunk=boundary - 1,
                next_chunk=0,
                **details,
                dit_kv_cleared=True,
                released_dit_bytes=released,
                weights_reloaded=True,
                backbone_reloaded=False,
                intent_reloaded=False,
                vae_reloaded=False,
                steps=args.steps,
                flow_shift=args.flow_shift,
            )
        )
        return True

    def apply_case_change(self):
        operation = self.state.take_case_change()
        if operation is None:
            return False
        self.resources.active()
        boundary = self.session.next_block
        self.wait_published(boundary - 1)
        old_stream, stream_id = self.state.stream_id, new_stream_id()
        released = self.session.kv.nbytes if self.session.kv is not None else 0
        self.session.kv = None
        self.gpu0.send("change_case", stream_id=stream_id, settings=operation["settings"])
        setup = self.gpu0.result(self.gpu0.case_changes)
        assert setup["intent"] == self.args.checkpoint_intent
        # All old GPU0 events precede this acknowledgment on the pipe.
        while True:
            try:
                self.gpu0.features.get_nowait()
            except queue.Empty:
                break
        self.gpu0.conditions = {c["version"]: c for c in setup["conditions"]}
        self.select_initial_prompt(setup["conditions"])
        self.reference = self.torch.from_numpy(setup["reference"]).bfloat16()
        with self.state.lock:
            self.state.activate_case(stream_id, operation["settings"], setup["context"])
            self.state.set_condition_contexts(setup["conditions"])
        self.args.exp_name, self.args.sample_id = stream_id, self.state.case_id
        self.args.data_config = str(self.config.case_catalog)
        self.args.local_dir = str(LOCAL / "dit" / stream_id)
        self.args.output_dir = str(RUNTIME / "results" / stream_id)
        self.new_session()
        info = read(RUNTIME / "server_info.json")
        atomic(
            RUNTIME / "server_info.json", dict(info, stream_id=stream_id, case=self.state.case_id)
        )
        self.state.case_finished(
            dict(
                operation,
                old_stream=old_stream,
                stream_id=stream_id,
                after_chunk=boundary - 1,
                next_chunk=0,
                dit_kv_cleared=True,
                released_dit_bytes=released,
                vae_restarted=True,
                model_reloaded=False,
            )
        )
        return True

    def wait_published(self, last):
        if last < 0:
            return
        with self.state.changed:
            while not self.state.latest or self.state.latest["end"] < last:
                if self.stopping.is_set():
                    raise InterruptedError("Service stopping")
                if self.state.error:
                    raise RuntimeError(self.state.error)
                self.state.changed.wait(timeout=1)

    def apply_reset(self):
        operation = self.state.take_reset()
        if operation is None:
            return False
        self.resources.active()
        boundary = self.session.next_block
        self.wait_published(boundary - 1)
        old_stream = self.state.stream_id
        full = operation["mode"] == "all"
        stream_id = new_stream_id() if full else old_stream
        released = self.session.kv.nbytes if self.session.kv is not None else 0
        self.session.kv = None
        self.cache_start, self.cache_epoch = boundary, self.cache_epoch + 1
        with self.input_gate:
            self.gpu0.send("reset", mode=operation["mode"], stream_id=stream_id)
            details = self.gpu0.result(self.gpu0.resets)
            details.pop("type")
            if full:
                # The acknowledgment is after all old GPU0 events. Only now is it
                # safe to discard queued features and restart the public indices.
                while True:
                    try:
                        self.gpu0.features.get_nowait()
                    except queue.Empty:
                        break
                self.state.new_stream(stream_id)
        if full:
            self.args.exp_name = stream_id
            self.args.local_dir = str(LOCAL / "dit" / stream_id)
            self.args.output_dir = str(RUNTIME / "results" / stream_id)
            self.new_session()
            info = read(RUNTIME / "server_info.json")
            atomic(RUNTIME / "server_info.json", dict(info, stream_id=stream_id))
        else:
            self.reset_wait_reuse()
            self.reset_wait_kv_window_status()
        self.state.reset_finished(
            dict(
                operation,
                **details,
                old_stream=old_stream,
                stream_id=stream_id,
                after_chunk=boundary - 1,
                next_chunk=0 if full else boundary,
                cache_epoch=self.cache_epoch,
                dit_kv_cleared=True,
                released_dit_bytes=released,
                model_reloaded=False,
                global_clock_preserved=not full,
            )
        )
        return full

    def reset_wait_reuse(self):
        with self.state.lock:
            self.wait_count = 0
            self.wait_intent_states = self.wait_source = None
            self.wait_reuse_revision = self.state.wait_reuse_revision
            self.state.wait_reuse.update(
                consecutive=0, source_index=None, reused=False, pending=False
            )

    def select_wait_intent_states(self, hidden, condition, index):
        """Replace only the DiT condition value; queues and positions stay current."""
        with self.state.lock:
            if self.wait_reuse_revision != self.state.wait_reuse_revision:
                self.reset_wait_reuse()
            settings = self.state.wait_reuse
            reused = False
            if settings["enabled"] and condition == "<wait>":
                self.wait_count += 1
                if self.wait_count == settings["threshold"]:
                    self.wait_intent_states, self.wait_source = hidden.clone(), index
                elif self.wait_count > settings["threshold"]:
                    hidden, reused = self.wait_intent_states, True
            else:
                self.wait_count = 0
                self.wait_intent_states = self.wait_source = None
            settings.update(
                consecutive=self.wait_count,
                source_index=self.wait_source,
                reused=reused,
                pending=False,
            )
            return hidden, dict(
                settings,
                index=index,
                intent_state_source_index=self.wait_source if reused else index,
            )

    def reset_wait_kv_window_status(self):
        with self.state.lock:
            self.state.wait_kv_window.update(
                max_chunks=self.args.sliding_window_size // self.session.geometry.frames,
                limit_chunks=None,
                retained_chunks=0,
                applied_index=None,
            )

    def select_wait_kv_window(self, condition):
        with self.state.lock:
            enabled = self.state.wait_kv_window["enabled"]
        frames = self.session.geometry.frames
        maximum = self.args.sliding_window_size
        limited = enabled and condition == "<wait>"
        self.session.args.sliding_window_size = min(maximum, 2 * frames) if limited else maximum
        return dict(
            enabled=enabled,
            limited=limited,
            limit_chunks=self.session.args.sliding_window_size // frames,
            max_chunks=maximum // frames,
        )

    def sample(self, sequence, history, index):
        torch = self.torch
        from duo.data.sequence import block_seed
        from duo.models.realization.pipelines.inference import sample_sequence

        assert {g["block"] for g in sequence.refiner_groups} == {-1, index}
        expected = self.current_features["intent_states"].to(
            device=sequence.device, dtype=sequence.refiner_text.dtype
        )
        assert torch.equal(sequence.refiner_text[0], expected)
        # In causal inference, intent states always indexes the clean t=1 slot.
        assert not sequence.time_ids[sequence.text_indices].any()
        steps = 0
        sampling_times = []
        began = time.time()

        def after_step(module, inputs, outputs):
            nonlocal steps
            torch.cuda.synchronize()
            if self.stopping.is_set():
                raise InterruptedError("Service stopping")
            if self.gpu0.failure:
                raise self.gpu0.failure
            steps += 1
            sampling_times.append(float(inputs[1][1]))
            self.state.stage(
                dict(
                    type="stage",
                    request_id=self.request["id"],
                    index=index,
                    stage="dit",
                    state="running",
                    step=steps,
                    steps=self.args.steps,
                )
            )

        handle = self.model.register_forward_hook(after_step)
        try:
            # The distilled path uses the checkpoint-declared re-noise schedule.
            result = sample_sequence(
                self.model,
                sequence,
                stage=self.args.stage,
                steps=self.args.steps,
                shift=self.args.flow_shift,
                seed=block_seed(self.args.seed, self.state.case_id, index),
                history=history,
            )
        finally:
            handle.remove()
        assert steps == self.args.steps
        self.sample_proof = dict(
            index=index,
            steps=steps,
            intent_state_rows=len(expected),
            shared_rows=len(self.shared),
            model_id=self.state.model_id,
            weight_family=self.args.checkpoint_family,
            sampling_stage=self.args.stage,
            sampling_times=sampling_times,
            future_intent_state_rows=0,
            shared_current_exact=True,
            clean_intent_states_timestep=True,
            denoise_started=began,
            denoise_finished=time.time(),
        )
        return result

    def run_chunk(self, feature):
        torch = self.torch
        index = feature["index"]
        if index != self.session.next_block:
            raise ValueError("intent states FIFO index differs from DiT next chunk")
        request = self.state.requests[feature["request_id"] - 1]
        self.request = request
        self.session.observations["blocks"].append(clock_block(index))
        folder = self.state.request_folder / str(request["id"])
        item = self.state.select_intent_states(feature)
        condition_changed = item["condition_version"] != self.condition_version
        changed = condition_changed or item["global_prompt_source"] != self.global_prompt_source
        if changed:
            self.set_shared_prompt(item["condition_version"], item["global_prompt_source"])
            self.reset_wait_reuse()
        hidden = torch.from_numpy(item["intent_states"]).bfloat16()
        condition = item["chunk"]["condition"]
        hidden, wait_reuse = self.select_wait_intent_states(hidden, condition, index)
        wait_kv_window = self.select_wait_kv_window(condition)
        group = dict(item["group"], start=len(self.shared), end=len(self.shared) + len(hidden))
        self.current_features = dict(
            intent_states=torch.cat((self.shared, hidden)),
            text_groups=[*self.shared_groups, group],
            protocol="reciprocal_frame_chatml_v5",
        )
        prompt_metrics = dict(
            enabled=item["wait_silence"],
            condition_version=self.condition_version,
            paired_consumption=False,
            ntp_forward=True,
        )
        global_prompt_metrics = dict(
            enabled=self.global_prompt_source == "original",
            source=self.global_prompt_source,
            chunk_intent_state_source="finetuned_intent_v2",
            condition_version=self.condition_version,
            shared_rows=len(self.shared),
            shared_kv_replaced=False,
        )
        if changed:
            from ..serving.wait_silence import refresh_shared

            refreshed = refresh_shared(self.session, self.current_features, index)
            prompt_metrics.update(refreshed)
            global_prompt_metrics.update(refreshed)
        self.state.stage(
            dict(
                type="stage",
                request_id=request["id"],
                index=index,
                stage="intent_states",
                state="consumed",
            )
        )
        self.state.stage(
            dict(
                type="stage",
                request_id=request["id"],
                index=index,
                stage="dit",
                state="running",
                step=0,
                metrics=dict(
                    wait_reuse=wait_reuse,
                    wait_silence=prompt_metrics,
                    wait_kv_window=wait_kv_window,
                    global_prompt=global_prompt_metrics,
                ),
            )
        )
        latent, metrics, part = self.session.generate(self.current_features, index)
        metrics["wait_reuse"] = wait_reuse
        metrics.update(
            model_id=self.state.model_id,
            weight_family=self.args.checkpoint_family,
            steps=self.args.steps,
        )
        metrics["wait_silence"] = prompt_metrics
        metrics["global_prompt"] = global_prompt_metrics
        assert metrics["media_window"]["start_block"] == max(
            0, index + 1 - wait_kv_window["limit_chunks"]
        )
        # Widening the limit cannot restore KV already evicted by a WAIT.
        self.cache_start = max(self.cache_start, metrics["media_window"]["start_block"])
        visible_start = self.cache_start
        assert {g["block"] for g in self.session.kv.groups if g["kind"] == "clean"} == set(
            range(visible_start, index + 1)
        )
        retained = index + 1 - visible_start
        metrics["media_window"] = dict(
            start_block=visible_start,
            stop_block=index + 1,
            frames=retained * self.session.geometry.frames,
        )
        metrics["wait_kv_window"] = dict(
            wait_kv_window, retained_chunks=retained, start_block=visible_start
        )
        with self.state.lock:
            self.state.wait_kv_window.update(
                limit_chunks=wait_kv_window["limit_chunks"],
                retained_chunks=retained,
                applied_index=index,
            )
        assert latent["video"].shape[1] == 5 and latent["audio"].shape == (2, 32, 29)
        atomic(
            folder / "dit" / f"C{index:05d}.json",
            dict(
                self.sample_proof,
                **metrics,
                request_id=request["id"],
                intent_local=item["chunk"]["local"],
                cache_epoch=self.cache_epoch,
                cache_start_block=self.cache_start,
                peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
            ),
        )
        self.current_features = None
        del hidden, item, part
        self.state.stage(
            dict(
                type="stage",
                request_id=request["id"],
                index=index,
                stage="dit",
                state="ready",
                metrics=metrics,
            )
        )
        self.gpu0.send(
            "latent",
            request_id=request["id"],
            index=index,
            video=latent["video"].float().numpy(),
            audio=latent["audio"].float().numpy(),
        )
        del latent

    def close(self):
        self.stopping.set()
        self.state.pending.put(None)
        if self.feeder.ident is not None:
            self.feeder.join(timeout=3)
        self.thread.join(timeout=30)
