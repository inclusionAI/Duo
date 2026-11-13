"""Single CPU coordinator; independent CP process groups never share collectives."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from .inference_state_realization import atomic_json, check_batch_namespace


def gpu_groups(pool, cp):
    if cp < 1 or not pool or len(pool) != len(set(pool)) or len(pool) % cp:
        raise ValueError(
            "GPU pool must contain unique devices and divide into complete CP groups"
        )
    return [pool[i : i + cp] for i in range(0, len(pool), cp)]


def assignments(records, count):
    return [records[i::count] for i in range(count)]


def local_checkpoint_view(checkpoint, root):
    """Materialize an independent, verified view while the source lease is held."""
    from tools.ckpt_to_oss.duo_protocol import verify_file

    from .storage_realization import Checkpoint

    root = Path(root) / "local-source" / checkpoint.sha256
    members = {}
    for name, record in checkpoint.manifest["files"].items():
        source = checkpoint.folder / name
        if checkpoint.partial and not source.exists() and not source.is_symlink():
            continue
        verify_file(source, record)
        members[name] = record
    controls = (
        ("manifest.json", "node-complete.json")
        if checkpoint.partial
        else ("complete.json",)
    )
    if checkpoint.partial and (checkpoint.folder / "complete.json").exists():
        controls += ("complete.json",)
    control_bytes = {}
    for name in controls:
        source = checkpoint.folder / name
        if source.is_symlink() or not source.is_file():
            raise ValueError("Missing or incomplete checkpoint marker: " + str(source))
        control_bytes[name] = source.read_bytes()

    def verify_view(folder):
        if folder.is_symlink():
            raise ValueError("Checkpoint view must be a regular directory")
        for name, data in control_bytes.items():
            target = folder / name
            if (
                target.is_symlink()
                or not target.is_file()
                or target.read_bytes() != data
            ):
                raise ValueError("Checkpoint view marker differs: " + str(target))
        view = Checkpoint(checkpoint.storage, folder, str(folder))
        if view.sha256 != checkpoint.sha256 or view.partial != checkpoint.partial:
            raise ValueError("Checkpoint view identity differs")
        for name, record in members.items():
            verify_file(folder / name, record)

    if root.exists() or root.is_symlink():
        verify_view(root)
        return str(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(
        record if isinstance(record, int) else record["size"]
        for record in members.values()
    )
    needed += sum(map(len, control_bytes.values())) + checkpoint.storage.reserve_bytes
    if shutil.disk_usage(root.parent).free < needed:
        raise OSError(f"Insufficient checkpoint view space: need {needed} free bytes")
    temporary = Path(tempfile.mkdtemp(prefix=checkpoint.sha256 + ".", dir=root.parent))
    try:
        for name in members:
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint.folder / name, target)
        for name, data in control_bytes.items():
            (temporary / name).write_bytes(data)
        verify_view(temporary)
        temporary.rename(root)
    finally:
        # Only remove the unpublished directory created by this invocation.
        shutil.rmtree(temporary, ignore_errors=True)
    return str(root)


def run(args):
    from . import storage_realization
    from .inference_config_realization import conditioning_identity
    from .inference_runtime_realization import select_records

    output, local = Path(args.output_dir), Path(args.local_dir)
    check_batch_namespace(output, local, args.resume)
    from torchtitan.models.duo_realization.modules.rope import (
        checkpoint_rope,
        resolve_rope,
    )

    storage_realization.configure(
        cache=str(Path(args.checkpoint_cache) / "coordinator"),
        reserve_bytes=int(args.checkpoint_reserve_gb * 1e9),
        local_world=1,
    )
    source = storage_realization.current().resolve(args.checkpoint)
    args.rope = resolve_rope(args.rope, checkpoint_rope(source.manifest["identity"]))
    from torchtitan.models.duo_realization.modules.adaln_timestep import (
        checkpoint_adaln_timestep,
        resolve_adaln_timestep,
    )

    args.checkpoint_adaln_timestep = checkpoint_adaln_timestep(
        source.manifest["identity"]
    )
    args.adaln_timestep = resolve_adaln_timestep(
        getattr(args, "adaln_timestep", None), source.manifest["identity"]
    )
    records = select_records(args)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        pool = visible.split(",")
    else:
        import torch

        pool = [str(i) for i in range(torch.cuda.device_count())]
    groups = gpu_groups(pool, args.context_parallel_degree)
    groups = groups[: min(len(groups), len(records))]
    selected = [[r["split"], r["sample_id"]] for r in records]
    settings = {
        k: v
        for k, v in vars(args).items()
        if k
        not in (
            "resume",
            "local_dir",
            "output_dir",
            "checkpoint_cache",
            "runtime_cache",
            "case_worker_config",
        )
    }
    batch_path = output / "batch.json"
    if batch_path.exists():
        batch = json.loads(batch_path.read_text())
        if (
            not args.resume
            or batch.get("selection") != selected
            or batch.get("inference_settings") != settings
        ):
            raise ValueError(
                "Batch selection/settings changed or old recovery semantics"
            )
    else:
        if args.resume:
            raise FileNotFoundError("Missing batch recovery")
        output.mkdir(parents=True, exist_ok=True)
        local.mkdir(parents=True, exist_ok=True)
        batch = dict(
            **conditioning_identity(args.mode, args.realization_inference_mode),
            selection=selected,
            inference_settings=settings,
            samples=[
                dict(split=r["split"], sample_id=r["sample_id"], status="pending")
                for r in records
            ],
        )
        atomic_json(batch_path, batch)
    children = []
    handles = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in previous:
        signal.signal(sig, terminate)
    started = time.monotonic()
    try:
        with ExitStack() as leases:
            storage_realization.configure(
                cache=str(Path(args.checkpoint_cache) / "coordinator"),
                reserve_bytes=int(args.checkpoint_reserve_gb * 1e9),
                local_world=1,
            )
            checkpoint = None
            if not args.checkpoint.startswith("oss://"):
                checkpoint = storage_realization.current().resolve(args.checkpoint)
                leases.enter_context(checkpoint.lease())
            for number, (devices, rows) in enumerate(
                zip(groups, assignments(records, len(groups)))
            ):
                folder = local / "case-groups" / f"group-{number}"
                folder.mkdir(parents=True, exist_ok=True)
                status = folder / "result.json"
                status.unlink(missing_ok=True)
                worker_args = dict(
                    vars(args),
                    checkpoint_cache=str(
                        Path(args.checkpoint_cache) / args.exp_name / f"group-{number}"
                    ),
                )
                if checkpoint is not None:
                    worker_args["checkpoint"] = local_checkpoint_view(
                        checkpoint, folder
                    )
                config = folder / "worker.json"
                atomic_json(
                    config,
                    dict(
                        args=worker_args,
                        records=rows,
                        result=str(status),
                        multiple=len(records) > 1,
                    ),
                )
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=",".join(devices),
                    NPROC=str(len(devices)),
                    DUO_RUNTIME_CACHE_ROOT=str(folder / "runtime"),
                    DUO_CACHE_RUN_ID=f"{args.exp_name}-group-{number}",
                    PYTHONDONTWRITEBYTECODE="1",
                )
                for key in (
                    "RANK",
                    "LOCAL_RANK",
                    "WORLD_SIZE",
                    "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                    "GROUP_RANK",
                    "ROLE_RANK",
                    "TORCHELASTIC_RUN_ID",
                ):
                    env.pop(key, None)
                command = [sys.executable]
                if len(devices) > 1:
                    command += [
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nnodes=1",
                        f"--nproc-per-node={len(devices)}",
                    ]
                command += [
                    "-m",
                    f"torchtitan.infer_{args.stage}_realization",
                    "--checkpoint",
                    worker_args["checkpoint"],
                ]
                if args.stage == "cm":
                    command.append(
                        "--cm_infer" if args.cm_sampling_mode == "cm" else "--ode_infer"
                    )
                command += ["--case-worker-config", str(config)]
                log_path = output / f"group-{number}.log"
                stream = log_path.open("a")
                handles.append(stream)
                # Inherit the bounded guard's process group so forced cleanup
                # also drains torchrun descendants after coordinator death.
                child = subprocess.Popen(
                    command, env=env, stdout=stream, stderr=subprocess.STDOUT
                )
                children.append(
                    dict(process=child, status=status, records=rows, log=str(log_path))
                )
            while True:
                by_id = {}
                for child in children:
                    if child["status"].is_file():
                        value = json.loads(child["status"].read_text())
                        by_id.update(
                            {(r["split"], r["sample_id"]): r for r in value["samples"]}
                        )
                    if child["process"].poll() is not None:
                        for record in child["records"]:
                            key = record["split"], record["sample_id"]
                            if key not in by_id or by_id[key]["status"] in (
                                "pending",
                                "running",
                            ):
                                by_id[key] = dict(
                                    split=key[0],
                                    sample_id=key[1],
                                    status="failed",
                                    error="Case group exited; inspect " + child["log"],
                                )
                updated = [
                    by_id.get(tuple(key), row)
                    for key, row in zip(selected, batch["samples"])
                ]
                if updated != batch["samples"]:
                    batch["samples"] = updated
                    atomic_json(batch_path, batch)
                if all(child["process"].poll() is not None for child in children):
                    break
                time.sleep(1)
            failed = sum(r["status"] != "completed" for r in batch["samples"])
            elapsed = time.monotonic() - started
            result = dict(
                samples=batch["samples"],
                failed=failed,
                batch_seconds=elapsed,
                completed_cases_per_second=(len(records) - failed) / elapsed,
                gpu_groups=groups,
                counts={
                    s: sum(r["status"] == s for r in batch["samples"])
                    for s in ("pending", "running", "completed", "failed")
                },
            )
            atomic_json(output / "batch_result.json", result)
            if failed:
                raise RuntimeError(
                    f"{failed} case(s) failed; inspect group logs and per-case states"
                )
    finally:
        for child in children:
            if child["process"].poll() is None:
                child["process"].terminate()
        for child in children:
            try:
                child["process"].wait(timeout=30)
            except subprocess.TimeoutExpired:
                child["process"].kill()
                child["process"].wait(timeout=10)
        for stream in handles:
            stream.close()
        for sig, value in previous.items():
            signal.signal(sig, value)
