"""Isolated RAVEN Intent/VAE/ASR worker; the Realization process never imports RAVEN.

CPU tensors travel through request-scoped local files. Stdout carries only
request IDs/status; all dependency and model logging goes to stderr.
"""

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def release_cuda_workspace(device):
    """Return unused allocator blocks before another process uses this GPU.

    Live model weights and committed media KV remain resident and unchanged.
    """
    import torch

    torch.cuda.synchronize(device)
    metrics = dict(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_before_bytes=torch.cuda.memory_reserved(device),
    )
    torch.cuda.empty_cache()
    metrics["reserved_after_bytes"] = torch.cuda.memory_reserved(device)
    return metrics


class Helpers:
    def __init__(self, args, *, worker_script=None, gpu_index=0):
        self.args = args
        root = Path(args.local_dir) / "ipc"
        root.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix="helpers-", dir=root))
        config = self.folder / "args.json"
        config.write_text(json.dumps(vars(args)))
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            DUO_HELPER_PARENT_PID=str(os.getpid()),
        )
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RUN_ID",
            "PYTHONHOME",
            "PYTHONPATH",
        ):
            env.pop(key, None)
        prefix = Path(args.helpers_python).parent.parent
        conda = os.environ.get("CUDA_HOME", str(prefix.parent / "raven"))
        env.update(
            VIRTUAL_ENV=str(prefix),
            CONDA_PREFIX=conda,
            PATH=str(prefix / "bin") + ":" + conda + "/bin:" + env.get("PATH", ""),
            LD_LIBRARY_PATH=conda + "/lib:" + conda + "/targets/x86_64-linux/lib",
        )
        # The helper only uses the first GPU of the already authorized visible set.
        visible = env.get("CUDA_VISIBLE_DEVICES")
        if visible:
            env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[gpu_index]
        elif worker_script is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        env["MAGI_ATTENTION_WORKSPACE_BASE"] = str(self.folder / "magi")
        for key, name in (
            ("TMPDIR", "tmp"),
            ("XDG_CACHE_HOME", "xdg"),
            ("HF_HOME", "hf"),
            ("TORCH_EXTENSIONS_DIR", "extensions"),
            ("TRITON_CACHE_DIR", "triton"),
        ):
            path = self.folder / name
            path.mkdir()
            env[key] = str(path)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        self.log = (Path(args.output_dir) / "helpers.log").open("a")
        self.process = subprocess.Popen(
            [
                args.helpers_python,
                str(worker_script or Path(__file__).resolve()),
                "--worker",
                str(config),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.counter = 0
        self.buffer = b""
        self.progress = None
        try:
            ready = self.read()
            if ready.get("ready") is not True:
                raise RuntimeError("Helper failed to initialize")
            self.runtime = ready["runtime"]
        except BaseException:
            self.close()
            raise

    def emit(self, event):
        if self.progress is not None:
            self.progress(event)
        else:
            print(json.dumps(dict(event="helper_progress", **event)), flush=True)

    def read(self, identifier=None, action="initialize"):
        start = last_done = time.monotonic()
        next_report = start + 30
        status = dict(phase=action)
        sequence = 0
        while True:
            now = time.monotonic()
            if now - last_done >= self.args.helper_timeout:
                raise TimeoutError(
                    f"Realization helper action={action} request={identifier} "
                    f"phase={status.get('phase')} block={status.get('block')} "
                    f"completed_blocks={status.get('completed_blocks')} elapsed={now-start:.1f}s "
                    f"without_progress={now-last_done:.1f}s; see {self.log.name}"
                )
            if now >= next_report:
                self.emit(
                    dict(
                        status,
                        request_id=identifier,
                        action=action,
                        elapsed_seconds=now - start,
                        waiting_seconds=now - last_done,
                        heartbeat=True,
                        alive=self.process.poll() is None,
                    )
                )
                next_report = now + 30
            if b"\n" not in self.buffer:
                timeout = min(last_done + self.args.helper_timeout, next_report) - now
                if not self.selector.select(max(0.0, timeout)):
                    continue
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(
                        f"Realization helper exited during {action}; see {self.log.name}"
                    )
                self.buffer += chunk
                continue
            line, self.buffer = self.buffer.split(b"\n", 1)
            result = json.loads(line)
            if identifier is not None and result.get("id") != identifier:
                raise RuntimeError("Mismatched helper response")
            if result.get("type") == "progress":
                if result.get("sequence") != sequence + 1:
                    raise RuntimeError("Noncontiguous helper progress")
                sequence += 1
                status.update(result["progress"])
                now = time.monotonic()
                if result["progress"].get("completed") is True:
                    last_done = now
                self.emit(
                    dict(
                        status,
                        request_id=identifier,
                        action=action,
                        elapsed_seconds=now - start,
                        waiting_seconds=now - last_done,
                    )
                )
                continue
            return result

    def call(self, action, **values):
        import torch

        if getattr(self, "closed", False):
            raise RuntimeError("Realization helper is closed after an earlier failure")
        self.counter += 1
        source = self.folder / f"{self.counter}.request.pt"
        target = self.folder / f"{self.counter}.response.pt"
        torch.save(values, source)
        try:
            self.process.stdin.write(
                json.dumps(dict(id=self.counter, action=action)) + "\n"
            )
            self.process.stdin.flush()
            result = self.read(self.counter, action)
            payload = (
                torch.load(target, map_location="cpu", weights_only=False)
                if result.get("ok")
                else None
            )
        except BaseException:
            # Finish owned-process cleanup BEFORE rank-zero broadcasts failure.
            self.close(abort=True)
            raise
        else:
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "Helper failed"))
            return payload
        finally:
            source.unlink(missing_ok=True)
            target.unlink(missing_ok=True)

    def close(self, abort=False):
        if getattr(self, "closed", False):
            return
        self.closed = True

        def signal_group(sig):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                pass

        if self.process.poll() is None:
            if abort:
                signal_group(signal.SIGTERM)
            else:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                self.process.wait(timeout=3 if abort else 5)
            except subprocess.TimeoutExpired:
                signal_group(signal.SIGKILL)
                self.process.wait(timeout=5)
        # The leader may already be gone while an ASR child still owns the group.
        signal_group(signal.SIGKILL)
        self.selector.close()
        for pipe in (self.process.stdin, self.process.stdout):
            try:
                pipe.close()
            except BrokenPipeError:
                pass
        self.log.close()
        shutil.rmtree(self.folder)


class Worker:
    def __init__(self, args):
        self.args = args
        self.planner = self.codecs = self.asr = None
        self.progress = lambda **values: None

    def select(self):
        from copy import copy

        from scripts.inference.batch import select_samples

        args = copy(self.args)
        if not (
            args.sample_id or args.sample_ids or args.all_samples or args.max_samples
        ):
            args.max_samples = 1
        return select_samples(args)

    def open(
        self,
        record,
        saved,
        output,
        local,
        checkpoint_codec=None,
        checkpoint_intent=None,
    ):
        import torch
        from data import build_dataset
        from scripts.inference.utils import history, observations
        from torchtitan.models.duo_realization.modules import media_features as features

        if self.planner is None:
            from scripts.model.codecs import Codecs
            from torchtitan.models.duo_realization.pipelines.intent_planner import (
                Planner,
            )

            self.planner = Planner(self.args, intent_dtype=torch.bfloat16)
            self.codecs = Codecs(
                self.args,
                vae_dtype=torch.float32,
                video_autocast_dtype=torch.float16,
                feature_dtype=torch.bfloat16,
            )
        if self.asr is not None:
            self.asr.close()
            self.asr = None
        if (
            self.args.mode == "closed_loop"
            and getattr(self.args, "realization_inference_mode", "causal") == "causal"
        ):
            from scripts.inference.asr.client import ASRClient

            self.asr = ASRClient(self.args, Path(output))
        self.record = record
        self.output = Path(output)
        self.local = Path(local)
        self.planner.capture_h50 = None
        if getattr(self.args, "save_latents", False):
            from data.inference_export import ExportCache

            self.planner.capture_h50 = ExportCache(local).capture
        self.generated = []
        self.frames = []
        self.waves = []
        self.controls = []
        self.visuals = {}
        language = record["canonical"].get("language", "").lower()
        self.language = self.args.language or {
            "zh": "chinese",
            "en": "english",
            "ja": "japanese",
            "chinese": "chinese",
            "english": "english",
            "japanese": "japanese",
        }.get(language)
        if self.asr is not None and self.language is None:
            raise ValueError("Specify --language for this dataset")
        preparation = record.get("preparation", {})
        padded = preparation.get("tail_policy") == "pad"
        duration = (
            preparation["original_frames"] * 1000 / 24
            if padded
            else record["canonical"]["duration_ms"]
        )
        user_video = features.read_video(
            record["canonical"]["source"]["user_video"],
            duration,
            self.args.height,
            self.args.width,
        )
        if padded:
            features.pad_video_tail(user_video, preparation["output_frames"])
        self.user_rgb = user_video["frames"]
        self.cache = None
        if self.args.feature_config or self.args.feature_manifest:
            source = self.args.feature_config or dict(
                source="raven_features", manifest=self.args.feature_manifest
            )
            dataset = build_dataset(
                source,
                record["split"],
                feature_root=self.args.feature_root or None,
                video_codec=getattr(self.args, "decode", "old"),
            )
            index = next(
                (
                    i
                    for i, r in enumerate(dataset)
                    if r["sample_id"] == record["sample_id"]
                ),
                None,
            )
            if index is None:
                raise ValueError("Selected features do not contain this sample/split")
            item = dataset.get(index)
            if (item.get("intent", {}).get("mode") == "origin") != getattr(
                self.args, "origin_intent", False
            ):
                raise ValueError("Feature Intent mode differs from runtime")
            self.cache = dict(
                codec=dataset.codec,
                intent=item.get("intent", {}),
                sample_id=item["sample_id"],
                reference=item["reference"],
                blocks=[
                    {k: b[k] for k in ("start_ms", "end_ms")} for b in item["blocks"]
                ],
            )
        else:
            self.cache = observations.encode(
                record, self.args, self.codecs, self.user_rgb
            )
        if tuple(self.cache["reference"].shape) != (
            24,
            1,
            self.args.height // 16,
            self.args.width // 16,
        ):
            raise ValueError("Reference resolution differs")
        if len(self.cache["blocks"]) != len(record["blocks"]):
            raise ValueError("Features/timeline length differs")
        for a, b in zip(self.cache["blocks"], record["blocks"]):
            if (a["start_ms"], a["end_ms"]) != (b["start_ms"], b["end_ms"]):
                raise ValueError("Features/timeline clock differs")
        from data.inference_media import runtime_identity, validate_checkpoint_codec

        validate_checkpoint_codec(checkpoint_codec, self.codecs.video.identity())
        if checkpoint_intent and checkpoint_intent != getattr(
            self.args, "intent_identity", {}
        ):
            raise ValueError("Checkpoint Intent configuration differs from runtime")
        runtime_identity(
            local,
            self.codecs.video.identity(),
            getattr(self.args, "intent_identity", {}),
            self.cache,
        )
        for block, state in enumerate(saved):
            if self.args.mode == "oracle":
                history.validate_oracle_state(
                    record, state, block, self.planner.tokenizer
                )
            self.generated.append(state["latent"])
            from data.inference_media import restore_packets

            self.frames.extend(p["frames"] for p in restore_packets(state, block))
            self.waves.append(state["wave"])
            self.controls.append(state["plan"]["controls"])
        self.codecs.restore(
            self.cache["reference"], [x["video"] for x in self.generated]
        )
        if saved and "helper_rng" in saved[-1]:
            import random

            import numpy as np

            rng = saved[-1]["helper_rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
        return self.cache

    def prepare_visuals(self, blocks):
        from torchtitan.models.duo_realization.modules import intent_conditioning

        for k, b in enumerate(self.record["blocks"][:blocks]):
            if ("user", k) not in self.visuals:
                self.visuals["user", k] = intent_conditioning.visual_patch(
                    self.planner.processor,
                    self.user_rgb[b["frame_start"] : b["frame_stop"]],
                )
                self.progress(
                    phase="user_visuals",
                    completed=True,
                    block=k,
                    completed_blocks=k + 1,
                    total_blocks=blocks,
                )

    def plan(self, block, bodies):
        self.prepare_visuals(block if self.args.mode == "closed_loop" else block + 1)
        return self.planner.plan(
            self.record,
            self.visuals,
            bodies,
            block,
            self.args.mode,
            self.args.max_new_tokens,
        )

    def prepare_oracle(self, blocks):
        if self.args.mode != "oracle":
            raise ValueError("Oracle preparation requires oracle mode")
        # Preparation is a derived cache, not an advance of resumed RNG state.
        before = self.rng()
        try:
            self.prepare_visuals(blocks)
            return self.planner.prepare_sequence(
                self.record,
                self.visuals,
                blocks,
                "oracle",
                self.args.max_new_tokens,
                self.progress,
            )
        finally:
            self.rng(restore=before)

    def plan_joint(self, blocks):
        if self.args.mode == "oracle":
            result = self.prepare_oracle(blocks)
        else:
            self.prepare_visuals(blocks)
            result = self.planner.prepare_sequence(
                self.record,
                self.visuals,
                blocks,
                self.args.mode,
                self.args.max_new_tokens,
                self.progress,
            )
        for plan in result["plans"]:
            plan["realization_inference_mode"] = "bidirectional"
        return dict(result, helper_rng=self.rng())

    def rng(self, restore=None):
        import random

        import numpy as np
        import torch

        if restore is not None:
            random.setstate(restore["python"])
            np.random.set_state(restore["numpy"])
            torch.set_rng_state(restore["torch"])
            torch.cuda.set_rng_state_all(restore["cuda"])
        return dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
        )

    def decode(self, block, latent, plan, planned_bodies=None):
        import time

        import torch
        from scripts.inference.utils import history, media

        if len(self.generated) != block:
            raise ValueError("Noncontiguous helper history")
        row = self.record["blocks"][block]
        torch.cuda.synchronize()
        start = time.monotonic()
        packets, wave, metrics = self.codecs.decode(
            self.cache["reference"],
            [x["video"] for x in self.generated],
            latent["video"],
            latent["audio"],
            row["audio_stop_sample"] - row["audio_start_sample"],
        )
        torch.cuda.synchronize()
        metrics["decode_seconds"] = time.monotonic() - start
        if self.args.mode == "oracle":
            bodies, asr, revision = history.oracle_feedback(
                self.record, block, self.planner.tokenizer, plan
            )
        elif (
            getattr(self.args, "realization_inference_mode", "causal")
            == "bidirectional"
        ):
            if planned_bodies is None or set(planned_bodies) != set(range(block + 1)):
                raise ValueError("Missing joint NTP text prefix")
            bodies = planned_bodies
            asr = dict(
                status="skipped", reason="bidirectional_ntp_text_v1", seconds=0.0
            )
            revision = dict(
                source="intent_ntp_full_sequence",
                text_by_block=[
                    self.planner.tokenizer.decode(
                        bodies[i],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    for i in range(block + 1)
                ],
            )
        else:
            scratch = self.local / ".asr-prefix.wav"
            media.write_wav(scratch, torch.cat(self.waves + [wave], dim=1))
            try:
                asr = self.asr.transcribe(scratch, self.language)
            finally:
                scratch.unlink(missing_ok=True)
            bodies, revision = history.reassign_words(
                asr["transcription"],
                self.record["blocks"][: block + 1],
                self.controls + [plan["controls"]],
                self.planner.tokenizer,
            )
        metrics["asr_seconds"] = asr.get("seconds")
        metrics["seam_delta"] = (
            float((wave[:, 0] - self.waves[-1][:, -1]).abs().max())
            if self.waves
            else None
        )
        from data.inference_media import current_media, packet_state, publish_packets

        self.generated.append(latent)
        self.waves.append(wave)
        self.controls.append(plan["controls"])
        for packet in packets:
            if packet["block"] != len(self.frames):
                raise ValueError("Noncontiguous video publication")
            self.frames.append(packet["frames"])
        write_start = time.monotonic()
        publications = publish_packets(
            self.output, packets, self.waves, self.args.ffmpeg
        )
        metrics["media_write_seconds"] = time.monotonic() - write_start
        import random

        import numpy as np

        helper_rng = dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
        )
        return dict(
            helper_rng=helper_rng,
            latent=latent,
            video_packets=packet_state(packets),
            publications=publications,
            wave=wave,
            plan=plan,
            bodies=bodies,
            asr=asr,
            revision=revision,
            metrics=metrics,
            media=current_media(publications, block),
        )

    def finish(self):
        import numpy as np
        import torch
        from data.inference_media import mark_flush, publish_packets
        from scripts.inference.utils import media

        packets = self.codecs.finish()
        for packet in packets:
            if packet["block"] != len(self.frames):
                raise ValueError("Noncontiguous EOF video publication")
            self.frames.append(packet["frames"])
        self.progress(phase="finish_media", completed=False)
        write_start = time.monotonic()
        publications = publish_packets(
            self.output, packets, self.waves, self.args.ffmpeg
        )
        if self.codecs.video.spec.delayed:
            mark_flush(self.output, len(self.generated), publications)
        if len(self.frames) != len(self.generated):
            raise ValueError("Incomplete video at EOF")
        wav, mp4 = self.output / "assistant.tmp.wav", self.output / "assistant.tmp.mp4"
        media.write_wav(wav, torch.cat(self.waves, dim=1))
        media.write_video(mp4, np.concatenate(self.frames), wav, self.args.ffmpeg)
        wav.replace(self.output / "assistant.wav")
        mp4.replace(self.output / "assistant.mp4")
        self.progress(
            phase="finish_media",
            completed=True,
            stage_seconds=time.monotonic() - write_start,
        )
        return dict(
            video=str(self.output / "assistant.mp4"),
            audio=str(self.output / "assistant.wav"),
        )

    def close(self):
        if self.asr is not None:
            self.asr.close()


def start_parent_watchdog():
    """Supervise a private worker group before importing GPU dependencies."""
    from torchtitan.components.helper_lifecycle_realization import identity

    parent = int(os.environ["DUO_HELPER_PARENT_PID"])
    helper = os.getpid()
    parent_start, helper_start = identity(parent), identity(helper)
    if os.getppid() != parent or parent_start is None:
        raise RuntimeError("Realization parent exited before helper startup")
    subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).with_name("helper_lifecycle_realization.py")),
            str(helper),
            helper_start,
            str(parent),
            parent_start,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def worker(config):
    # Establish supervision before importing GPU dependencies or loading models.
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    start_parent_watchdog()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    root = Path(__file__).resolve().parents[2]
    raven = root / "projects/RAVEN-main"
    sys.path[:0] = [str(raven), str(root)]
    import torch

    torch.set_num_threads(2)
    args = SimpleNamespace(**json.loads(config.read_text()))
    import random

    import numpy as np

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    work = Worker(args)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    import importlib.metadata
    import platform

    runtime = dict(
        python=platform.python_version(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "numpy")
        },
    )
    protocol.write(json.dumps(dict(ready=True, runtime=runtime)) + "\n")
    protocol.flush()
    try:
        for line in sys.stdin:
            request = json.loads(line)
            identifier = request["id"]
            sequence = 0
            stage_started = time.monotonic()
            stage = None

            def progress(**values):
                nonlocal sequence, stage_started, stage
                key = (values.get("phase"), values.get("block"))
                if key != stage:
                    stage, stage_started = key, time.monotonic()
                sequence += 1
                protocol.write(
                    json.dumps(
                        dict(
                            id=identifier,
                            type="progress",
                            sequence=sequence,
                            progress=dict(
                                values, stage_seconds=time.monotonic() - stage_started
                            ),
                        )
                    )
                    + "\n"
                )
                protocol.flush()

            work.progress = progress
            if type(identifier) is not int or identifier < 1:
                raise ValueError("Invalid helper request ID")
            try:
                if request["action"] not in (
                    "select",
                    "open",
                    "plan",
                    "prepare_oracle",
                    "plan_joint",
                    "rng",
                    "decode",
                    "finish",
                ):
                    raise ValueError("Invalid action")
                payload = torch.load(
                    config.parent / f"{identifier}.request.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                result = getattr(work, request["action"])(**payload)
                progress(phase="save_response", completed=False)
                torch.save(result, config.parent / f"{identifier}.response.pt")
                progress(phase="save_response", completed=True)
                reply = dict(id=identifier, ok=True)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                reply = dict(
                    id=identifier, ok=False, error=f"{type(exc).__name__}: {exc}"
                )
            protocol.write(json.dumps(reply) + "\n")
            protocol.flush()
    finally:
        work.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Internal helper worker")
    worker(Path(sys.argv[2]))
