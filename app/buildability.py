"""Step 4 — Client side of the buildability worker pool.

Starts a small pool of `aizynth_worker.py` processes in the AiZynthFinder
environment and hands each incoming molecule to a free worker. Each check
takes ~10 s (tree search against reaction rules), so N workers ≈ N×
throughput; the pool also keeps the web server responsive throughout.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "app" / "aizynth_worker.py"
PYTHON = ROOT / ".venv-aizynth" / "bin" / "python"

POOL_SIZE = 2  # workers × ~10 s each ≈ 5 molecules in ~25 s wall time


class BuildabilityError(RuntimeError):
    pass


class _Worker:
    """One AiZynthFinder subprocess speaking JSON lines."""

    def __init__(self) -> None:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [str(PYTHON), str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(ROOT),
            env=env,
        )
        ready = self.proc.stdout.readline()
        if not ready or "ready" not in ready:
            raise BuildabilityError("buildability worker failed to start")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def ask(self, smiles: str, request_id: int) -> dict:
        self.proc.stdin.write(json.dumps({"smiles": smiles, "id": request_id}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise BuildabilityError("worker died")
        return json.loads(line)


class BuildabilityChecker:
    """A pool of workers, one lock per worker; requesters wait for a free one."""

    def __init__(self, pool_size: int = POOL_SIZE) -> None:
        if not PYTHON.exists():
            raise BuildabilityError("AiZynthFinder environment missing (.venv-aizynth)")

        self._workers: queue.Queue[_Worker] = queue.Queue()
        for _ in range(pool_size):
            self._workers.put(_Worker())

    def check_smiles_batch(
        self,
        smiles_list: list[str],
        per_item_timeout: float = 60.0,
        on_progress=None,
    ) -> list[dict]:
        del per_item_timeout  # bounded by the worker's expansion_time instead
        results: list[dict | None] = [None] * len(smiles_list)
        progress_lock = threading.Lock()
        done = [0]

        def run_one(i: int, smi: str) -> None:
            worker = self._workers.get()
            try:
                results[i] = worker.ask(smi, i)
            except Exception as err:  # noqa: BLE001 — report, don't crash the batch
                results[i] = {"smiles": smi, "error": str(err)[:200]}
                if worker.alive():
                    self._workers.put(worker)
                else:
                    try:
                        self._workers.put(_Worker())  # replace the dead one
                    except BuildabilityError:
                        pass
            else:
                self._workers.put(worker)
            finally:
                with progress_lock:
                    done[0] += 1
                    if on_progress:
                        on_progress(done[0], len(smiles_list))

        threads = [threading.Thread(target=run_one, args=(i, s)) for i, s in enumerate(smiles_list)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return [r if r is not None else {"smiles": s, "error": "no result"} for r, s in zip(results, smiles_list)]

    def shutdown(self) -> None:
        while not self._workers.empty():
            worker = self._workers.get()
            if worker.alive():
                worker.proc.kill()


_checker: BuildabilityChecker | None = None
_checker_lock = threading.Lock()


def get_checker() -> BuildabilityChecker:
    global _checker
    with _checker_lock:
        if _checker is None:
            _checker = BuildabilityChecker()
        return _checker
