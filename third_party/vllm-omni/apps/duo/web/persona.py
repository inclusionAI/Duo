# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import base64
import hashlib
import io
import json
import os
import shutil
import uuid
from pathlib import Path

from display_geometry import OUTPUT_SIZES
from PIL import Image, ImageOps
from system_prompt import resolve

ROOT = Path(__file__).resolve().parent
PRESETS = json.loads((ROOT / "personas.json").read_text())
SESSIONS = ROOT.parent / "personas"


async def prepare(packet):
    key = packet["preset"]
    if key not in PRESETS:
        raise ValueError("未知人物预设")
    orientation = packet.get("orientation", "portrait")
    if orientation not in OUTPUT_SIZES:
        raise ValueError("未知画面方向")
    system_message = None
    if packet.get("prompt_mode") == "system":
        fields, system_message = resolve(packet["system_prompt"])
    else:
        fields = {k: packet[k].strip() for k in ("appearance", "environment", "initial_pose")}
        if any(
            not v or len(v) > 800 or any(c in v for c in ("<|", "<chunk_", "<wait>", "<speak>", "<blank>"))
            for v in fields.values()
        ):
            raise ValueError("人物、环境和初始姿态均需填写，每项最多 800 字，不接受协议控制符")
    folder = SESSIONS / uuid.uuid4().hex
    folder.mkdir(mode=0o700)
    try:
        if packet.get("image") or orientation == "landscape":
            if packet.get("image"):
                raw = base64.b64decode(packet["image"], validate=True)
                if len(raw) > 3 << 20:
                    raise ValueError("参考图过大")
                source = io.BytesIO(raw)
            else:
                source = ROOT / "portraits" / key / "portrait.jpg"
            with Image.open(source) as img:
                if img.width * img.height > 24_000_000:
                    raise ValueError("图片像素过多")
                img = ImageOps.pad(
                    ImageOps.exif_transpose(img).convert("RGB"),
                    OUTPUT_SIZES[orientation],
                    method=Image.Resampling.LANCZOS,
                    color="black",
                )
                img.save(folder / "portrait.jpg", quality=94)
                pixels = (folder / "portrait.jpg").read_bytes()
            model_root = Path(os.environ["DUO_MODEL_ROOT"]).resolve()
            weights = (model_root / "FL2VA/video_vae/source/model.safetensors").stat()
            identity = f"{model_root}:{weights.st_size}:{weights.st_mtime_ns}".encode()
            cache_key = hashlib.sha256(
                identity
                + pixels
                + (model_root / "FL2VA/video_vae/config.json").read_bytes()
                + (ROOT / "encode_portrait.py").read_bytes()
            ).hexdigest()
            cache = SESSIONS.parent / "portrait-cache"
            cache.mkdir(exist_ok=True)
            cached = cache / (cache_key + ".pt")
            if cached.exists():
                shutil.copyfile(cached, folder / "reference.pt")
            else:
                await encode_portrait(folder)
                temporary = cache / (cache_key + ".pending")
                shutil.copyfile(folder / "reference.pt", temporary)
                temporary.replace(cached)
        else:
            shutil.copyfile(ROOT / "portraits" / key / "reference.pt", folder / "reference.pt")
        (folder / "description.json").write_text(json.dumps(fields, ensure_ascii=False))
        if system_message is not None:
            (folder / "system_message.txt").write_text(system_message)
        return folder
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise


_encoder = None
_encoder_lock = asyncio.Lock()


async def start_encoder():
    global _encoder
    if _encoder is None:
        _encoder = await asyncio.create_subprocess_exec(
            os.environ["DUO_ENGINE_PYTHON"],
            str(ROOT / "encode_portrait.py"),
            "--serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": os.environ["DUO_ENGINE_PYTHONPATH"]},
        )


async def encode_portrait(folder):
    async with _encoder_lock:
        await start_encoder()

        async def exchange():
            _encoder.stdin.write((json.dumps(dict(folder=str(folder))) + "\n").encode())
            await _encoder.stdin.drain()
            line = await _encoder.stdout.readline()
            if not line:
                raise RuntimeError("参考图编码器已退出，请查看网关日志")
            reply = json.loads(line)
            if reply["folder"] != str(folder):
                raise RuntimeError("参考图编码结果不属于当前请求")
            if "error" in reply:
                raise ValueError(reply["error"])
            print("Portrait encoded", reply, flush=True)

        job = asyncio.create_task(exchange())
        try:
            await asyncio.shield(job)
        except asyncio.CancelledError:
            # Drain this reply before another request can reuse the same pipe.
            await job
            raise


async def close_encoder():
    if _encoder is not None:
        _encoder.stdin.close()
        await _encoder.wait()
