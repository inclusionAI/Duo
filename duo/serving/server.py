"""Shared request API, real telemetry and byte-range serving on the live port."""

import argparse
from base64 import b64decode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit

from ..config import load_service_config
from .common import ROOT, RUNTIME, atomic, owner, new_stream_id
from ..inference_runtime import Engine
from .state import State
from .media import recording_mp4
from .models import MODELS, configure_models, get_model, saved_model, require_checkpoint
from .access import AccessError
from ..preprocessing.pipeline import MAX_UPLOAD, UPLOADS
from .video_state import snapshot as video_snapshot


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ("200", "206"):
            super().log_message(fmt, *args)

    @property
    def state(self):
        return self.server.state

    def json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/video/upload":
                self.upload()
                return
            size = int(self.headers.get("Content-Length", "0"))
            limit = 32768
            if path == "/api/cases":
                with self.state.lock:
                    member = self.state.access.authenticate(self.token, self.connection_id)
                    if member["role"] != "admin":
                        raise AccessError("Only the active administrator can create a Case.")
                self.connection.settimeout(30)
                limit = 7 * 1024 * 1024
            if not 0 < size <= limit:
                raise ValueError("Invalid request length")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("The request body must be a JSON object")
            with self.state.lock:
                room = self.state.access
                token = self.token
                if path == "/api/video/depart":
                    token = token or data.get("token", "")
                connection_id = self.connection_id or data.get("connectionId", "")
                result = None
                authenticated = True
                if path == "/api/video/join":
                    result = room.join(
                        data.get("name"), data.get("role"), data.get("password"), connection_id
                    )
                elif path == "/api/video/resume":
                    room.resume(token, connection_id)
                elif path == "/api/video/depart":
                    room.disconnect(token, connection_id)
                    self.json(dict(ok=True))
                    return
                else:
                    member = room.authenticate(token, connection_id)
                    if path == "/api/video/leave":
                        room.remove(member, "You left the session.")
                        authenticated = False
                    elif path == "/api/video/kick":
                        result = room.moderate(member, data.get("memberId"))
                    elif path == "/api/video/accept":
                        room.accept(member, data.get("turnId"))
                    elif path == "/api/video/touch":
                        room.touch(member, data.get("sessionId"))
                    elif path == "/api/video/capture":
                        result = room.begin_capture(member, data.get("sessionId"))
                    elif path == "/api/video/cancel-capture":
                        room.require_operator(member, data.get("sessionId"), ready=False)
                        if room.capture and room.capture["id"] == data.get("captureId"):
                            room.capture = None
                            room.changed()
                    else:
                        room.require_operator(member, data.get("sessionId"))
                        if path in ("/api/case", "/api/cases", "/api/model") and member["role"] != "admin":
                            raise AccessError(
                                "Only the active administrator can change model or Case."
                            )
                        if path == "/api/reset":
                            result = self.state.request_reset(data.get("mode"))
                            if data["mode"] == "all":
                                room.invalidate()
                        elif path == "/api/cases":
                            reference = b64decode(data.get("reference", ""), validate=True)
                            if not 0 < len(reference) <= 5 * 1024 * 1024:
                                raise ValueError("Choose a JPEG, PNG, or WebP reference up to 5 MiB.")
                            case_id = self.state.cases.create(data.get("label"), data.get("description"), reference)
                            result = self.state.request_case({"case_id": case_id})
                            room.invalidate()
                        elif path == "/api/case":
                            result = self.state.request_case(dict(data, opening_chunk_ids=[]))
                            if result["state"] != "unchanged":
                                room.invalidate()
                        elif path == "/api/model":
                            result = self.state.request_model(data.get("model_id"))
                            if result["state"] != "unchanged":
                                room.invalidate()
                        elif path == "/api/wait-reuse":
                            result = self.state.configure_wait_reuse(
                                data.get("enabled"), data.get("threshold")
                            )
                        elif path == "/api/wait-kv-window":
                            result = self.state.configure_wait_kv_window(data.get("enabled"))
                        elif path == "/api/global-prompt":
                            result = self.state.configure_global_prompt(data.get("enabled"))
                        elif path == "/api/requests":
                            raise ValueError("Record or upload a video to submit input.")
                        else:
                            raise AccessError("Endpoint not found.", 404)
                self.json(
                    dict(
                        result=result, state=video_snapshot(self.state, authenticated=authenticated)
                    )
                )
        except AccessError as exc:
            self.json(dict(error=str(exc)), exc.status)
        except (ValueError, TypeError) as exc:
            self.json(dict(error=str(exc)), 400)
        except RuntimeError as exc:
            self.json(dict(error=str(exc)), 503)
        except TimeoutError:
            self.json(dict(error="Upload timed out. Try again."), 408)
        except OSError as exc:
            self.log_error("request I/O: %s", exc)
            self.json(dict(error="The recording could not be saved. Try again."), 503)

    @property
    def token(self):
        value = self.headers.get("Authorization", "")
        return value.removeprefix("Bearer ") if value.startswith("Bearer ") else ""

    @property
    def connection_id(self):
        return self.headers.get("X-Connection-ID", "")

    def upload(self):
        query = parse_qs(urlsplit(self.path).query)
        capture_id = query.get("capture", [""])[0]
        session_id = query.get("session", [""])[0]
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 < size <= MAX_UPLOAD:
            raise ValueError("Recording size must be between 0 and 512 MiB.")
        mime = self.headers.get("Content-Type", "").split(";", 1)[0]
        if mime not in ("video/webm", "video/mp4"):
            raise ValueError("Upload a browser-recorded WebM or MP4 video.")
        metadata = json.loads(self.headers.get("X-Recording-Metadata", "{}"))
        if not isinstance(metadata, dict):
            raise ValueError("Invalid recording metadata.")
        metadata = {key: metadata.get(key, 0) for key in ("duration", "width", "height", "fps")}
        if any(
            type(value) not in (float, int) or value < 0 or not __import__("math").isfinite(value)
            for value in metadata.values()
        ):
            raise ValueError("Invalid recording metadata.")
        metadata.update(mimeType=mime, size=size)
        room = self.state.access
        with self.state.lock:
            member = room.authenticate(self.token, self.connection_id)
            capture = dict(room.check_capture(member, session_id, capture_id))
            if room.capture.get("uploading"):
                raise AccessError("This recording is already uploading.", 409)
            room.capture["uploading"] = True
            folder = UPLOADS / self.state.stream_id / capture_id
            folder.mkdir(parents=True, exist_ok=True)
        temporary = None
        self.connection.settimeout(30)
        try:
            with tempfile.NamedTemporaryFile(dir=folder, suffix=".upload", delete=False) as target:
                temporary = Path(target.name)
                remaining = size
                while remaining:
                    block = self.rfile.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError("Video upload was interrupted. Record again.")
                    target.write(block)
                    remaining -= len(block)
                    with self.state.lock:
                        member = room.authenticate(self.token, self.connection_id)
                        room.check_capture(member, session_id, capture_id)
            with self.state.lock:
                member = room.authenticate(self.token, self.connection_id)
                room.check_capture(member, session_id, capture_id)
                path = folder / ("original.mp4" if mime == "video/mp4" else "original.webm")
                temporary.replace(path)
                result = self.state.submit_recording(member, capture, path, metadata)
                self.json(dict(result=result, state=video_snapshot(self.state)))
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
            with self.state.lock:
                if room.capture and room.capture["id"] == capture_id:
                    room.capture.pop("uploading", None)

    def events(self):
        room = self.state.access
        try:
            with self.state.lock:
                lease = room.connected(self.token, self.connection_id)
        except AccessError as exc:
            self.json(dict(error=str(exc)), exc.status)
            return
        self.connection.settimeout(5)
        if hasattr(socket, "TCP_USER_TIMEOUT"):
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 5000)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while not self.server.stop.is_set():
                with self.state.lock:
                    room.sweep()
                    member = next((m for m in room.members if m["token"] == self.token), None)
                    alive = (
                        member
                        and member["connectionId"] == self.connection_id
                        and member.get("eventLease") == lease
                        and member["connected"]
                    )
                    data = video_snapshot(self.state, authenticated=bool(member))
                self.wfile.write(
                    ("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode()
                )
                self.wfile.flush()
                if not alive:
                    break
                self.server.stop.wait(1)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass
        finally:
            with self.state.lock:
                room.disconnect(self.token, self.connection_id, lease)

    def do_GET(self):
        self.respond()

    def do_HEAD(self):
        self.respond()

    def respond(self):
        request = urlsplit(self.path)
        name = unquote(request.path)
        if name in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "./video/index.html")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if name == "/api/video/events":
            self.events()
            return
        if name == "/api/video/state":
            query = parse_qs(request.query)
            with self.state.lock:
                member = self.state.access.heartbeat(self.token, self.connection_id) if self.token else None
                version = query.get("version", ["0"])[0]
                self.json(
                    video_snapshot(
                        self.state,
                        authenticated=bool(member),
                        client=member["id"] if member else None,
                        version=int(version) if version.isdigit() else 0,
                    )
                )
            return
        if name == "/api/models":
            self.json([model.public() for model in MODELS.values()])
            return
        if name == "/api/frontend-catalog":
            with self.state.lock:
                self.json(
                    dict(
                        models=[
                            dict(
                                id=model.id,
                                label=model.label,
                                detail="Configured checkpoint",
                                sampler=f"{model.steps} sampling steps",
                            )
                            for model in MODELS.values()
                        ],
                        cases=self.state.cases.frontend(),
                    )
                )
            return
        if name == "/api/cases":
            with self.state.lock:
                self.json(self.state.cases.public())
            return
        if name == "/api/state":
            query = parse_qs(request.query)
            value = query.get("version", ["0"])[0]
            self.json(
                self.state.snapshot(
                    query.get("client", [None])[0], int(value) if value.isdigit() else 0
                )
            )
            return
        export_recording = False
        if name.startswith("/api/cases/") and name.endswith("/reference-image"):
            case_id = name[len("/api/cases/") : -len("/reference-image")].strip("/")
            try:
                path = self.state.cases.assistant_image(case_id)
            except (FileNotFoundError, ValueError):
                self.send_error(404)
                return
            base, relative = path.parent, path.name
        elif name.startswith("/recordings/"):
            base, relative = UPLOADS, name[len("/recordings/") :]
            parts = Path(relative).parts
            export_recording = len(parts) == 3 and parts[-1] == "download.mp4"
            if export_recording:
                folder = (base / relative).parent.resolve()
                if not folder.is_relative_to(base):
                    self.send_error(404)
                    return
                try:
                    exported = recording_mp4(folder, self.server.ffmpeg)
                except FileNotFoundError:
                    self.send_error(404)
                    return
                except (OSError, subprocess.SubprocessError) as exc:
                    self.log_error("recording export failed: %s", exc)
                    self.send_error(503, "MP4 conversion failed. Try again.")
                    return
                base, relative = folder, exported.name
            if Path(relative).suffix not in (".mp4", ".webm"):
                self.send_error(404)
                return
        elif name.startswith("/requests/"):
            base, relative = RUNTIME / "requests", name[len("/requests/") :]
        elif name.startswith("/media/"):
            base, relative = RUNTIME / "results", name[len("/media/") :]
        else:
            if self.server.frontend_root and name.startswith("/video/"):
                base, relative = self.server.frontend_root, name[len("/video/") :]
            else:
                base, relative = ROOT / "frontend_live", name.lstrip("/") or "index.html"
        path = (base / relative).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
            if match and any(match.groups()):
                a, b = match.groups()
                start = int(a) if a else max(0, size - int(b))
                end = min(size - 1, int(b)) if a and b else size - 1
            else:
                start = size
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if export_recording:
            self.send_header("Content-Disposition", 'attachment; filename="recording.mp4"')
        self.send_header(
            "Cache-Control", "public, max-age=86400" if path.suffix == ".mp4" else "no-cache"
        )
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD":
            try:
                with path.open("rb") as source:
                    source.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = source.read(min(remaining, 512 * 1024))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass


def telemetry(state, stop):
    while not stop.is_set():
        try:
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            rows = []
            for line in raw.splitlines():
                index, name, total, used, utilization = [x.strip() for x in line.split(",")]
                rows.append(
                    dict(
                        index=int(index),
                        name=name,
                        total_mib=float(total),
                        used_mib=float(used),
                        utilization=float(utilization),
                    )
                )
            with state.lock:
                state.gpus = rows
        except (subprocess.SubprocessError, ValueError):
            with state.lock:
                state.gpus = []
        stop.wait(2)


def sweep_access(state, stop):
    next_cleanup = 0
    while not stop.wait(1):
        with state.lock:
            state.access.sweep()
            stream_id = state.stream_id
        if time.monotonic() >= next_cleanup:
            # Only temporary uploads owned by this service expire; generated
            # Assistant results and historical inference files are retained.
            for folder in UPLOADS.glob("S*"):
                try:
                    if (
                        folder.name != stream_id
                        and not folder.is_symlink()
                        and folder.is_dir()
                        and time.time() - folder.stat().st_mtime > 86400
                    ):
                        shutil.rmtree(folder)
                except OSError:
                    pass
            next_cleanup = time.monotonic() + 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=1194)
    parser.add_argument("--config", default=os.environ.get("DUO_CONFIG", "configs/inference.json"))
    parser.add_argument("--case")
    parser.add_argument("--model", default="duo")
    parser.add_argument(
        "--frontend-root", type=Path, help="Optional frontend directory served under /video/"
    )
    args = parser.parse_args()
    config = load_service_config(args.config)
    os.environ["DUO_CONFIG"] = str(config.config_path)
    os.environ["DUO_CASE_CATALOG"] = str(config.case_catalog)
    os.environ["DUO_RUNTIME_DIR"] = str(config.runtime_dir)
    configure_models(config)
    args.case = args.case or config.default_case
    if args.model not in MODELS:
        parser.error(f"unknown model profile: {args.model}")
    model_id = args.model or saved_model()
    require_checkpoint(get_model(model_id))
    stream_id = new_stream_id()
    state = State(stream_id, args.case, model_id)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    server.state = state
    server.frontend_root = args.frontend_root.resolve() if args.frontend_root else None
    server.ffmpeg = config.executable(config.executables.ffmpeg)
    engine = Engine(state, config)
    stop = threading.Event()
    server.stop = stop
    threading.Thread(target=telemetry, args=(state, stop), daemon=True).start()
    threading.Thread(target=sweep_access, args=(state, stop), daemon=True).start()

    def shutdown(signum, frame):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    public_url = (
        os.environ.get("DUO_PUBLIC_URL") or f"http://127.0.0.1:{args.port}/video/index.html"
    )
    atomic(
        RUNTIME / "server_info.json",
        dict(
            owner(),
            port=args.port,
            stream_id=stream_id,
            case=args.case,
            model_id=model_id,
            url=public_url,
            mode="video_closed_loop",
            config=config.public(),
        ),
    )
    engine.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
