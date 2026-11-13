#!/usr/bin/env python3
"""Launch the complete Duo backend and browser demo."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Duo end-to-end inference")
    parser.add_argument("--config", default=str(root / "configs/inference.json"))
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--case")
    args = parser.parse_args()
    os.chdir(root)
    config_path = Path(args.config).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    realization_gpu = raw.get("inference", {}).get("realization_gpu", 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(realization_gpu)
    runtime_value = raw.get("runtime", {}).get("directory", "runtime")
    runtime_path = Path(runtime_value).expanduser()
    if not runtime_path.is_absolute():
        runtime_path = root / runtime_path
    os.environ["DUO_RUNTIME_DIR"] = str(runtime_path.resolve())
    requested = raw.get("executables", {}).get("realization_python", sys.executable)
    executable = (
        str((root / requested).resolve())
        if "/" in requested and not Path(requested).is_absolute()
        else shutil.which(requested) or requested
    )
    if (
        not os.environ.get("DUO_REALIZATION_REEXEC")
        and Path(executable).resolve() != Path(sys.executable).resolve()
    ):
        env = dict(os.environ, DUO_REALIZATION_REEXEC="1", DUO_CONFIG=str(config_path))
        os.execve(executable, [executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    os.environ["DUO_CONFIG"] = str(config_path)
    from duo.serving.server import main as serve

    command = [
        "--config",
        args.config,
        "--port",
        str(args.port),
        "--frontend-root",
        str(root / "demo"),
    ]
    if args.case:
        command += ["--case", args.case]
    sys.argv = [sys.argv[0], *command]
    serve()


if __name__ == "__main__":
    main()
