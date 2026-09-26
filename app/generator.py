"""Step 3 — Generation: run REINVENT4's sampler as a subprocess and collect SMILES.

REINVENT4 is a command-line tool driven by TOML config files, so the
simplest robust integration is: write a small TOML, run `reinvent`, read
the CSV it produces. CPU sampling of a few hundred molecules takes only
seconds, which is fine for a synchronous API call in a hobby app.
"""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (app/ -> codebase)
PRIOR = ROOT / "runs" / "models" / "reinvent_pubchem.prior"
REINVENT_BIN = ROOT / ".venv-reinvent" / "bin" / "reinvent"


class GenerationError(RuntimeError):
    pass


def sample_molecules(num_molecules: int = 100, seed: int | None = None) -> list[str]:
    """Sample unique, valid molecules from the de novo prior. Raises GenerationError."""
    if not REINVENT_BIN.exists():
        raise GenerationError("REINVENT4 is not installed (missing .venv-reinvent)")
    if not PRIOR.exists():
        raise GenerationError(
            "Prior model missing — download it with: "
            "curl -sL -o runs/models/reinvent_pubchem.prior "
            "https://zenodo.org/api/records/20701824/files/reinvent_pubchem.prior/content"
        )

    with tempfile.TemporaryDirectory(prefix="reinvent_run_") as tmp:
        tmpdir = Path(tmp)
        config = tmpdir / "sampling.toml"
        output = tmpdir / "sampled.csv"

        lines = [
            'run_type = "sampling"',
            'device = "cpu"',
            "",
            "[parameters]",
            f'model_file = "{PRIOR}"',
            f'output_file = "{output}"',
            f"num_smiles = {int(num_molecules)}",
            "unique_molecules = true",
            "randomize_smiles = true",
        ]
        config.write_text("\n".join(lines) + "\n")

        cmd = [str(REINVENT_BIN), str(config)]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]

        started = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(tmpdir),
        )
        elapsed = time.time() - started

        if proc.returncode != 0 or not output.exists():
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise GenerationError(f"REINVENT failed after {elapsed:.1f}s:\n{tail}")

        smiles: list[str] = []
        with open(output, newline="") as fh:
            for row in csv.DictReader(fh):
                smi = (row.get("SMILES") or "").strip()
                if smi:
                    smiles.append(smi)
        return smiles
