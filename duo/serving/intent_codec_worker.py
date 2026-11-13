"""GPU0: sequential Intent/VAE GPU calls, concurrent CPU encoding and IPC."""

import argparse
from multiprocessing.connection import Connection
import os
import queue
import threading
import time

from ..config import load_service_config
from .common import RUNTIME, atomic, read, native_imports, clock_block
from ..models.intent import StreamingIntent

native_imports()


class Codecs:
    def __init__(self, intent, stream_id, emit, config):
        import torch
        from ..models.codecs.streaming_video import StreamingVideoCodec
        from duo.models.realization.modules.audio_vae import AudioVAE
        from .media import Publisher

        self.torch, self.emit = torch, emit
        emit(dict(type="service", phase="loading_vae", detail="Loading media codecs"))
        self.video = StreamingVideoCodec(
            duo=config.resolve(config.weights.codec), device="cuda:0", decode_backend="solarx"
        )
        self.reference = self.video.encode_reference(intent.reference_rgb)
        self.video.start_decode(self.reference)
        self.video._get_solarx_decoder()
        self.audio = AudioVAE(
            self.video.args, vae_dtype=torch.float32, feature_dtype=torch.bfloat16, keep_on_gpu=True
        ).to("cuda:0")
        self.next_block = 0
        self.decode_start = 0
        self.publisher = Publisher(
            stream_id, intent.reference_rgb.shape[1], intent.reference_rgb.shape[0], emit
        )
        self.save_reference(intent)

    def save_reference(self, intent):
        atomic(
            RUNTIME
            / "case"
            / intent.case_id
            / intent.case_settings["config_id"]
            / "reference.json",
            dict(
                shape=list(self.reference.shape),
                dtype=str(self.reference.dtype),
                freshly_encoded=True,
                source=intent.source["media"]["assistant_image"],
                decoder="solarx",
                decoder_delay=0,
            ),
        )

    def change_case(self, intent, stream_id):
        from .media import Publisher

        self.publisher.close()
        self.video.finish_decode()
        self.reference = self.video.encode_reference(intent.reference_rgb)
        self.video.start_decode(self.reference)
        self.video._get_solarx_decoder()
        self.next_block = self.decode_start = 0
        self.publisher = Publisher(
            stream_id, intent.reference_rgb.shape[1], intent.reference_rgb.shape[0], self.emit
        )
        self.save_reference(intent)

    def decode(self, item):
        torch = self.torch
        request, index = item["request_id"], item["index"]
        assert index == self.next_block
        self.emit(
            dict(type="stage", request_id=request, index=index, stage="dit", state="consumed")
        )
        self.emit(dict(type="stage", request_id=request, index=index, stage="vae", state="running"))
        began = time.time()
        packets = self.video.decode_block(torch.from_numpy(item["video"]))
        assert len(packets) == 1 and packets[0]["block"] == index - self.decode_start
        row = clock_block(index)
        samples = row["audio_stop_sample"] - row["audio_start_sample"]
        wave, metrics = self.audio.decode_block(torch.from_numpy(item["audio"]), samples)
        assert len(packets[0]["frames"]) == 17
        self.next_block += 1
        self.emit(
            dict(
                type="stage",
                request_id=request,
                index=index,
                stage="vae",
                state="ready",
                metrics=dict(
                    packets[0]["metrics"],
                    **metrics,
                    decode_start=self.decode_start,
                    started=began,
                    finished=time.time(),
                ),
            )
        )
        self.publisher.submit(request, index, packets[0]["frames"], wave.numpy())

    def reset(self, stream_id=None):
        # finish_decode releases SolarX's actual KV, unlike reset(), which
        # only invalidates slots and retains the allocated K/V buffers.
        self.video.finish_decode()
        self.video.start_decode(self.reference)
        if stream_id is not None:
            from .media import Publisher

            self.publisher.close()
            self.next_block = 0
            self.publisher = Publisher(
                stream_id, self.publisher.width, self.publisher.height, self.emit
            )
        self.decode_start = self.next_block
        return dict(
            vae_kv_cleared=self.video._solarx_decoder is None,
            decode_start=self.decode_start,
            reference_retained=True,
        )

    def close(self):
        self.publisher.close()
        self.video.finish_decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--case", default="default")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_service_config(args.config)
    import torch

    torch.set_num_threads(4)
    connection = Connection(args.fd)
    lock = threading.Lock()
    commands = queue.Queue()

    def emit(event):
        with lock:
            connection.send(event)

    def receive():
        try:
            while True:
                command = connection.recv()
                commands.put(command)
                if command["action"] == "close":
                    return
        except EOFError:
            commands.put(dict(action="close"))

    threading.Thread(target=receive, daemon=True, name="gpu0-input").start()
    codecs = None
    try:
        intent = StreamingIntent(
            args.case,
            emit,
            RUNTIME / "requests" / args.stream_id,
            settings=read(args.settings),
            config_path=args.config,
        )
        inputs = queue.Queue()
        exhausted = False
        codecs = Codecs(intent, args.stream_id, emit, config)
        emit(
            dict(
                type="ready",
                pid=os.getpid(),
                context=intent.public_context(),
                reference=codecs.reference.float().numpy(),
                shared=intent.prefix_intent_states[: intent.shared_rows].float().numpy(),
                shared_groups=intent.shared_groups,
                condition_version=intent.condition_version,
                conditions=[intent.public_condition()],
                intent=intent.args.intent_identity,
            )
        )
        while True:
            # New real input replaces continuation imagery at the next producer
            # boundary. Already queued intent states remains ordered and is never rebuilt.
            if not inputs.empty() and (
                intent.request is None or exhausted or intent.next_local >= intent.input_count
            ):
                if intent.request is not None:
                    if not exhausted:
                        emit(
                            dict(
                                type="ntp_request_end",
                                request_id=intent.request["id"],
                                next_index=intent.next_global,
                            )
                        )
                    intent.finish_request()
                request = inputs.get_nowait()
                chunks = intent.begin(request, intent.next_global)
                emit(
                    dict(
                        type="planned",
                        request_id=request["id"],
                        chunks=chunks,
                        sequence_url=intent.sequence_url,
                    )
                )
                exhausted = False
            if intent.request is not None and not exhausted and commands.empty():
                item = intent.next_intent_states()
                if item["type"] == "ntp_done":
                    exhausted = True
                    emit(
                        dict(
                            type="ntp_request_end",
                            request_id=intent.request["id"],
                            next_index=intent.next_global,
                            reason=item["reason"],
                        )
                    )
                else:
                    emit(item)
                continue
            command = commands.get()
            action = command["action"]
            if action == "close":
                break
            if action == "request":
                if command["request"]["stream_id"] == args.stream_id:
                    inputs.put(command["request"])
            elif action == "latent":
                codecs.decode(command)
            elif action == "original_prompt":
                emit(dict(type="conditions", conditions=[intent.ensure_original_prompt()]))
            elif action == "reset":
                if command["mode"] == "all":
                    intent_info = intent.reset_all()
                    inputs = queue.Queue()
                    exhausted = False
                    args.stream_id = command["stream_id"]
                    intent.request_root = RUNTIME / "requests" / args.stream_id
                else:
                    intent_info = dict(intent_unchanged=True, intent_states_queue_preserved=True)
                codec_info = codecs.reset(args.stream_id if command["mode"] == "all" else None)
                emit(
                    dict(
                        type="reset_done", mode=command["mode"], intent=intent_info, **codec_info
                    )
                )
            elif action == "change_case":
                args.stream_id = command["stream_id"]
                intent.request_root = RUNTIME / "requests" / args.stream_id
                inputs = queue.Queue()
                exhausted = False
                intent.configure_case(command["settings"])
                codecs.change_case(intent, args.stream_id)
                emit(
                    dict(
                        type="case_ready",
                        context=intent.public_context(),
                        reference=codecs.reference.float().numpy(),
                        shared=intent.prefix_intent_states[: intent.shared_rows].float().numpy(),
                        shared_groups=intent.shared_groups,
                        condition_version=intent.condition_version,
                        conditions=[intent.public_condition()],
                        intent=intent.args.intent_identity,
                    )
                )
            else:
                raise ValueError("Unknown GPU0 action: " + action)
    except BaseException as exc:
        import traceback

        traceback.print_exc()
        emit(dict(type="fatal", error=f"GPU0: {type(exc).__name__}: {exc}"))
        return 1
    finally:
        if codecs is not None:
            codecs.close()
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
