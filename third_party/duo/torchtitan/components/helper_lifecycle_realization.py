"""Stdlib-only Linux watchdog for the helper's private process group."""

import os
import signal
import sys
import time
from pathlib import Path


def identity(pid):
    try:
        fields = (
            (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        )
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def kill_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def watch(helper, helper_start, parent, parent_start):
    # Separate watchdog session: it survives both rank-zero and helper death,
    # and can enforce SIGKILL even if CUDA prevents a Python signal handler running.
    while identity(helper) == helper_start and identity(parent) == parent_start:
        time.sleep(0.5)
    current = identity(helper)
    if current is not None and current != helper_start:
        return  # PID was reused; never signal another task.
    kill_group(helper, signal.SIGTERM)
    time.sleep(3)
    if identity(helper) in (None, helper_start):
        kill_group(helper, signal.SIGKILL)


if __name__ == "__main__":
    watch(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4])
