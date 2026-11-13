"""A training Intent service using the shared supervised helper transport."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


class TrainingQwen:
    def __init__(self, config, dataset, dims, device, local_dir):
        import torch.distributed as dist
        from data.dataset_utils import load_config
        from torchtitan.models.duo_realization.pipelines.dmd_conditioning import (
            POLICY,
            runtime_config,
        )

        self.c, self.dataset, self.device = config, dataset, device
        self.group = dims.world_mesh["cp"].get_group() if dims.cp_enabled else None
        self.leader = (
            dist.get_global_rank(self.group, 0)
            if self.group is not None
            else dist.get_rank()
        )
        self.helper = None
        self.settings = (
            dataset.rollout_runtime
            if hasattr(dataset, "rollout_runtime")
            else runtime_config(dataset.manifest, load_config(config.features))
        )
        self.identity = dict(
            policy=POLICY,
            intent=dataset.manifest["intent"],
            extraction={
                k: self.settings[k]
                for k in (
                    "intent_max_pixels",
                    "max_intent_tokens",
                    "intent_dtype",
                    "intent_attention",
                )
            },
        )
        self.local_dir = Path(local_dir) / f"intent-rank-{dist.get_rank()}"
        # A CPU-only handshake binds the separate runtime before resume checks.
        status = [None]
        if dist.get_rank() == self.leader:
            try:
                status[0] = (True, self._helper().runtime)
            except BaseException as error:
                self.close()
                status[0] = (False, f"{type(error).__name__}: {error}")
        if self.group is not None:
            dist.broadcast_object_list(status, src=self.leader, group=self.group)
        if not status[0][0]:
            raise RuntimeError("Training Intent startup failed: " + status[0][1])
        self.identity["runtime"] = status[0][1]

    def _helper(self):
        if self.helper is None:
            from torchtitan.components.helpers_inference_realization import Helpers

            python = self.c.intent_rollout_python or sys.executable
            if not Path(python).is_file():
                raise FileNotFoundError(
                    "Specify --realization.intent-rollout-python: " + python
                )
            args = SimpleNamespace(
                **self.settings,
                helpers_python=python,
                local_dir=str(self.local_dir),
                output_dir=str(self.local_dir),
                helper_timeout=self.c.intent_rollout_timeout,
                seed=self.c.seed,
                max_new_tokens=self.c.intent_rollout_max_new_tokens,
            )
            self.helper = Helpers(
                args,
                worker_script=Path(__file__).resolve(),
                gpu_index=int(os.environ.get("LOCAL_RANK", 0)),
            )
        return self.helper

    def materialize(self, prepared, device):
        import torch.distributed as dist
        from data.dataloader_realization import materialize_realization
        from data.dataloader_realization_sequence import prepare_sequence
        from torchtitan.models.duo_realization.pipelines.dmd_conditioning import (
            window_record,
        )

        count = len(prepared["sessions"])
        if (
            count == 0
            or len(prepared["windows"]) != count
            or len(prepared["record_indices"]) != count
        ):
            raise ValueError("Online condition session/window/index counts differ")
        status = [None]
        if dist.get_rank() == self.leader:
            try:
                requests = []
                for (session, valid), selected, index in zip(
                    prepared["sessions"],
                    prepared["windows"],
                    prepared["record_indices"],
                ):
                    source = self.dataset.records[index]["rollout_source"]
                    requests.append(
                        dict(
                            record=window_record(source, selected, session),
                            rollout_start=selected["rollout_start"] - selected["start"],
                        )
                    )
                from torchtitan.components.helpers_inference_realization import (
                    release_cuda_workspace,
                )

                release_cuda_workspace(device)
                value = self._helper().call("conditions", requests=requests)
                status[0] = (True, value)
            except BaseException as error:
                self.close()
                status[0] = (False, f"{type(error).__name__}: {error}")
        if self.group is not None:
            dist.broadcast_object_list(status, src=self.leader, group=self.group)
        ok, value = status[0]
        if not ok:
            raise RuntimeError("Training Intent helper failed: " + value)
        if not isinstance(value, list) or len(value) != count:
            raise RuntimeError(
                "Training Intent returned a different number of conditions"
            )
        sessions = [
            (dict(session, **item["features"]), valid)
            for (session, valid), item in zip(prepared["sessions"], value)
        ]
        packed = prepare_sequence(sessions, **prepared["options"])
        inputs, targets, layout = materialize_realization(packed, device)
        layout["intent_rollout"] = [item["metadata"] for item in value]
        return inputs, targets, layout

    def close(self):
        if self.helper is not None:
            self.helper.close()
            self.helper = None


def worker(config_path):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from torchtitan.components.helpers_inference_realization import (
        start_parent_watchdog,
    )

    start_parent_watchdog()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    import signal

    import torch
    from torchtitan.models.duo_realization.modules import intent
    from torchtitan.models.duo_realization.pipelines.dmd_conditioning import (
        rollout_conditions,
        user_visuals,
    )

    args = SimpleNamespace(**json.loads(Path(config_path).read_text()))
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    processor = model = None
    import importlib.metadata as metadata
    import platform

    runtime = dict(
        python=platform.python_version(),
        cuda=torch.version.cuda,
        versions={
            name: metadata.version(name)
            for name in ("torch", "transformers", "peft", "numpy", "av", "Pillow")
        },
    )
    protocol.write(json.dumps(dict(ready=True, runtime=runtime)) + "\n")
    try:
        for line in sys.stdin:
            request = json.loads(line)
            identifier = request["id"]
            sequence = 0

            def progress(**values):
                nonlocal sequence
                sequence += 1
                protocol.write(
                    json.dumps(
                        dict(
                            id=identifier,
                            type="progress",
                            sequence=sequence,
                            progress=values,
                        )
                    )
                    + "\n"
                )

            try:
                if request["action"] != "conditions":
                    raise ValueError("Unknown training Intent action")
                payload = torch.load(
                    Path(config_path).parent / f"{identifier}.request.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                if model is None:
                    progress(phase="load_intent", completed=False)
                    processor, model, _ = intent.load(
                        args,
                        dtype=getattr(torch, args.intent_dtype),
                        inference_rollout=True,
                    )
                    if args.intent_identity != args.expected_intent:
                        raise ValueError("Online/cached Intent identities differ")
                    progress(phase="load_intent", completed=True)
                results = []
                try:
                    model.to(args.device)
                    for item in payload["requests"]:
                        visuals = user_visuals(item["record"], processor)
                        results.append(
                            rollout_conditions(
                                model,
                                processor.tokenizer,
                                item["record"],
                                visuals,
                                item["rollout_start"],
                                args,
                                progress=progress,
                            )
                        )
                finally:
                    model.to("cpu")
                    torch.cuda.empty_cache()
                target = Path(config_path).parent / f"{identifier}.response.pt"
                temporary = target.with_suffix(".tmp")
                torch.save(results, temporary)
                temporary.replace(target)
                protocol.write(json.dumps(dict(id=identifier, ok=True)) + "\n")
            except BaseException as error:
                import traceback

                traceback.print_exc()
                protocol.write(
                    json.dumps(
                        dict(
                            id=identifier,
                            ok=False,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
                    + "\n"
                )
                break
    finally:
        if model is not None:
            del model
            torch.cuda.empty_cache()
        protocol.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("Internal training Intent worker")
    worker(sys.argv[2])
