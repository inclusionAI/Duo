"""Process-local GPU lease used by the standalone public demo."""

from contextlib import contextmanager
import fcntl
import threading

from .common import RUNTIME, atomic, owner


class Resources:
    def __init__(self, config):
        self.config = config
        self.lock = None
        self.guard = threading.RLock()
        self.owners = [owner()]
        self.holds = 0

    def acquire(self):
        path = RUNTIME / "gpu.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = path.open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise
        self.write()

    def write(self):
        atomic(
            RUNTIME / "gpu_lease.json",
            {
                "gpus": [self.config.inference.intent_gpu, self.config.inference.realization_gpu],
                "owners": self.owners,
            },
        )

    def child(self, pid):
        with self.guard:
            self.owners.append(owner(pid))
            self.write()

    @contextmanager
    def hold(self):
        with self.guard:
            self.holds += 1
        try:
            yield
        finally:
            with self.guard:
                self.holds -= 1

    def idle(self):
        return None

    def active(self):
        return None

    def close(self):
        if self.lock is not None:
            atomic(RUNTIME / "gpu_lease.json", {"gpus": [], "owners": []})
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None
