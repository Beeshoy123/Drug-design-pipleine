"""Step 4 — Client side of the buildability worker.

Starts `aizynth_worker.py` in the AiZynthFinder environment once, keeps it
running, and offers `check_smiles()` for one-shot checks plus
`check_smiles_batch()` that reports progress per finished molecule.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "app" / "aizynth_worker.py"
PYTHON = ROOT / ".venv-aizynth" / "bin" / "python"


class BuildabilityError(RuntimeError):
    pass


class BuildabilityChecker:
    """One worker process, reused for every molecule check."""

    def __init__(self) -> None:
        if not PYTHON.exists():
            raise BuildabilityError("AiZynthFinder environment missing (.venv-aizynth)")

        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def _ensure_worker(self) -> subprocess.Popen:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [str(PYTHON), str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(ROOT),
            env=env,
        )
        ready = self._proc.stdout.readline()
        if not ready or "ready" not in ready:
            raise BuildabilityError("buildability worker failed to start")
        return self._proc

    def check_smiles(self, smiles: str, timeout: float = 120.0) -> dict:
        return self.check_smiles_batch([smiles], per_item_timeout=timeout, on_progress=None)[0]

    def check_smiles_batch(
        self,
        smiles_list: list[str],
        per_item_timeout: float = 120.0,
        on_progress=None,
    ) -> list[dict]:
        """Check molecules one after another; on_progress(done, total) callback."""
        with self._lock:  # one worker, one request at a time
            proc = self._ensure_worker()
            results: list[dict] = []
            total = len(smiles_list)
            for i, smi in enumerate(smiles_list):
                proc.stdin.write(json.dumps({"smiles": smi, "id": i}) + "\n")
                proc.stdin.flush()

                def read_line() -> str:
                    return proc.stdout.readline()

                line = read_line()
                if not line:
                    self._proc = None  # worker died; next call restarts it
                    results.append({"smiles": smi, "error": "worker died"})
                else:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        results.append({"smiles": smi, "error": "bad worker output"})
                if on_progress:
                    on_progress(i + 1, total)
            return results

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        self._proc = None


_checker: BuildabilityChecker | None = None
_checker_lock = threading.Lock()


def get_checker() -> BuildabilityChecker:
    global _checker
    with _checker_lock:
        if _checker is None:
            _checker = BuildabilityChecker()
        return _checker
