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
STARTUP_TIMEOUT_S = 60.0  # ready-line deadline; model load is ~1 s normally


class BuildabilityError(RuntimeError):
    pass


def _readline_with_timeout(stream, timeout: float) -> str:
    """Read one line from a pipe-backed text stream with a deadline.

    ``stream.readline()`` has no timeout parameter, and ``select()`` on the
    underlying fd can disagree with Python's buffered text layer (a line may
    already sit in the buffer while the fd looks empty), so the read runs in a
    short-lived helper thread and we wait on it — matching this module's
    thread-pool design.  The helper is a daemon: on timeout the caller kills
    the worker, which closes the pipe and lets the blocked reader exit.

    Raises TimeoutError("timed out after Ns") when nothing arrives in time.
    """
    result: list[str] = []

    def read_line() -> None:
        result.append(stream.readline())

    reader = threading.Thread(target=read_line, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        raise TimeoutError(f"timed out after {timeout:g}s")
    return result[0] if result else ""


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
        try:
            ready = _readline_with_timeout(self.proc.stdout, STARTUP_TIMEOUT_S)
        except TimeoutError as err:
            self.kill()
            raise BuildabilityError(f"buildability worker failed to start ({err})") from err
        if not ready or "ready" not in ready:
            raise BuildabilityError("buildability worker failed to start")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        """Kill the subprocess and reap it (used for hung workers)."""
        if self.alive():
            self.proc.kill()
        try:
            self.proc.wait(timeout=5)  # don't leave a zombie behind
        except subprocess.TimeoutExpired:
            pass

    def ask(self, smiles: str, request_id: int, timeout: float) -> dict:
        self.proc.stdin.write(json.dumps({"smiles": smiles, "id": request_id}) + "\n")
        self.proc.stdin.flush()
        line = _readline_with_timeout(self.proc.stdout, timeout)
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
        """Check molecules in parallel across the pool.

        per_item_timeout bounds each worker response: the worker's
        expansion_time only bounds the tree search itself, not hangs (stuck
        model load, OOM, blocked I/O).  A timed-out item gets an error dict
        and its worker is killed and replaced, never reused.
        """
        results: list[dict | None] = [None] * len(smiles_list)
        progress_lock = threading.Lock()
        done = [0]

        def run_one(i: int, smi: str) -> None:
            worker = self._workers.get()
            try:
                results[i] = worker.ask(smi, i, per_item_timeout)
            except TimeoutError as err:
                # A timed-out worker may still be chewing on the old request;
                # putting it back would corrupt the JSON-lines protocol for the
                # next caller.  Kill and replace — same pattern as a dead one.
                results[i] = {"smiles": smi, "error": str(err)[:200]}
                worker.kill()
                try:
                    self._workers.put(_Worker())
                except BuildabilityError:
                    pass
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
