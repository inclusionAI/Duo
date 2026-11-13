import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parent
BASE_ROOT = ROOT
RUNTIME = Path(os.environ.get("DUO_RUNTIME_DIR", ROOT / "runtime")).resolve()
INPUTS = ROOT / "configs"
LOCAL = RUNTIME / "cache"
PYTHON = os.environ.get("DUO_REALIZATION_PYTHON", os.sys.executable)
INTENT_PYTHON = os.environ.get("DUO_INTENT_PYTHON", os.sys.executable)
FFMPEG = os.environ.get("DUO_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("DUO_FFPROBE", "ffprobe")
FPS = 24
FRAMES = 17
WAIT_SILENCE_PROMPT = "在<wait>的时候不许说话，安静倾听，嘴巴闭上。"


def new_stream_id():
    return datetime.now(timezone.utc).strftime("S%Y%m%d_%H%M%S_%f")


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def owner(pid=None):
    pid = os.getpid() if pid is None else pid
    return dict(pid=pid, start_id=identity(pid), time=time.time())


def environment(name, gpu, *, settings=None):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        OSS_SYNC="0",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=str(ROOT),
    )
    for key, folder in dict(
        TMPDIR="tmp",
        TEMP="tmp",
        TMP="tmp",
        XDG_CACHE_HOME="xdg",
        HF_HOME="hf",
        TORCH_HOME="torch",
        TRITON_CACHE_DIR="triton",
        TORCHINDUCTOR_CACHE_DIR="inductor",
        TORCH_EXTENSIONS_DIR="extensions",
        CUDA_CACHE_PATH="cuda",
        PYTORCH_KERNEL_CACHE_PATH="kernels",
        DUO_RUNTIME_CACHE_ROOT="runtime",
    ).items():
        path = LOCAL / name / folder
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "NODE_RANK",
        "TORCHELASTIC_RUN_ID",
        "PYTHONHOME",
    ):
        env.pop(key, None)
    python = INTENT_PYTHON if gpu == 0 else PYTHON
    ffmpeg = FFMPEG
    if settings is not None:
        python = (
            settings.executables.intent_python
            if gpu == settings.inference.intent_gpu
            else settings.executables.realization_python
        )
        ffmpeg = settings.executables.ffmpeg
        env["DUO_CONFIG"] = str(settings.config_path)
        env["DUO_RUNTIME_DIR"] = str(settings.runtime_dir)
    env["PATH"] = (
        str(Path(python).parent) + ":" + str(Path(ffmpeg).parent) + ":" + env.get("PATH", "")
    )
    return env


def native_imports():
    """Compatibility hook: all runtime modules now live in the Duo package."""


def clock_block(index):
    start, stop = index * FRAMES, (index + 1) * FRAMES
    return dict(
        block_id=index,
        frame_start=start,
        frame_stop=stop,
        start_ms=start * 1000 / FPS,
        end_ms=stop * 1000 / FPS,
        audio_start_sample=(start * 32000 + FPS - 1) // FPS,
        audio_stop_sample=(stop * 32000 + FPS - 1) // FPS,
        audio_padded_samples=29 * 800,
        committed={"user": [], "assistant": []},
    )
