# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of prepared Realization FA3 configuration and launcher isolation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PREPARE = Path(__file__).resolve().parents[4] / "apps/duo/prepare.py"


@pytest.fixture
def inputs(tmp_path):
    image = tmp_path / "reference.png"
    Image.new("RGB", (4, 4)).save(image)
    artifact = tmp_path / "artifact"
    artifact.write_text("dummy")
    engine = tmp_path / "fake engine"
    engine.write_text('#!/bin/sh\nprintf "%s\\n" "${DUO_FP8_FA3_LIBRARY-unset}" "${DUO_FP8_ATTENTION-unset}"\n')
    engine.chmod(0o755)
    library = tmp_path / "library with spaces.so"
    library.write_text("dummy")
    args = []
    for name in (
        "model-root",
        "intent-adapter",
        "intent-protocol",
        "realization-adapter",
        "asr-model",
        "reference-latent",
    ):
        args += ["--" + name, str(artifact)]
    args += ["--reference-image", str(image), "--engine-python", str(engine), "--web-python", sys.executable]
    return args, library


@pytest.mark.parametrize("precision,configured", [("bf16", False), ("fp8", False), ("fp8", True)])
def test_prepared_launcher_does_not_inherit_library(tmp_path, inputs, precision, configured):
    args, library = inputs
    runtime = tmp_path / "runtime"
    cmd = [sys.executable, str(PREPARE), "--runtime-dir", str(runtime), *args, "--precision", precision]
    if precision == "fp8":
        cmd += ["--fp8-profiles", str(tmp_path / "profile.rank{rank}.pt")]
    if configured:
        cmd += ["--fa3-library", str(library)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    env = os.environ.copy()
    env.update(DUO_FA3_LIBRARY="/inherited/duo.so", DUO_FP8_FA3_LIBRARY="/inherited/runtime.so")
    result = subprocess.run(["bash", str(runtime / "serve.sh")], env=env, check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [
        str(library.resolve()) if configured else "unset",
        "1" if precision == "fp8" else "unset",
    ]
    assert ("DUO_FA3_LIBRARY=" in (runtime / "environment.sh").read_text()) is configured


@pytest.mark.parametrize("precision,library_kind", [("bf16", "file"), ("fp8", "directory"), ("fp8", "missing")])
def test_fa3_library_validation(tmp_path, inputs, precision, library_kind):
    args, library = inputs
    path = {"file": library, "directory": tmp_path, "missing": tmp_path / "missing.so"}[library_kind]
    cmd = [
        sys.executable,
        str(PREPARE),
        "--runtime-dir",
        str(tmp_path / "runtime"),
        *args,
        "--precision",
        precision,
        "--fa3-library",
        str(path),
    ]
    if precision == "fp8":
        cmd += ["--fp8-profiles", str(tmp_path / "profile.rank{rank}.pt")]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--fa3-library" in result.stderr
    assert not (tmp_path / "runtime").exists()
