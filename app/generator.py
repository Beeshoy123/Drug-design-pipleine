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

# RL (staged_learning) pacing, measured on this box (1 CPU core, 2026-09):
# ~30 s process startup (torch import + prior load) + ~4 s per RL step at
# batch_size 64 (30 steps = ~150 s wall).  The budgets below keep a run
# comfortably under the 240 s subprocess timeout; exceeding them raises
# before wasting minutes.  (An earlier 5-step bench suggested ~2.6 s/step —
# startup amortization fooled it; 4.0 s/step is the honest long-run number.)
RL_STARTUP_BUDGET_S = 30.0
RL_PER_STEP_BUDGET_S = 4.0
RL_TIMEOUT_S = 240
RL_DEFAULT_NUM_STEPS = 30   # ~2.5 min wall on this CPU (200 steps would be ~14 min)
RL_DEFAULT_BATCH_SIZE = 64
RL_MAX_NUM_STEPS = 45       # 30 s + 45 * 4 s = 210 s, the last value inside the budget

# Filled by run_staged_learning() with run details beyond the bare SMILES list
# (timing, mean rewards, CSV row counts) for API responses and debugging.
last_rl_run_info: dict = {}


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


def _reinvent_subprocess_env() -> dict:
    """Environment for the REINVENT subprocess, with our plugin dir discoverable.

    REINVENT4 finds user scoring components by walking every
    reinvent_plugins/components/ namespace-package portion on PYTHONPATH
    (reinvent/scoring/importer.py) — app/ must be on it or the reward plugin
    is silently skipped and RL would run unsteered.
    """
    import reward_plugin

    env = dict(os.environ)
    parent = str(reward_plugin.PLUGIN_PARENT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{parent}{os.pathsep}{existing}" if existing else parent
    return env


def _check_reinvent_available() -> None:
    if not REINVENT_BIN.exists():
        raise GenerationError("REINVENT4 is not installed (missing .venv-reinvent)")
    if not PRIOR.exists():
        raise GenerationError(
            "Prior model missing — download it with: "
            "curl -sL -o runs/models/reinvent_pubchem.prior "
            "https://zenodo.org/api/records/20701824/files/reinvent_pubchem.prior/content"
        )


def run_staged_learning(
    reference_signature: dict[str, int],
    num_steps: int = RL_DEFAULT_NUM_STEPS,
    batch_size: int = RL_DEFAULT_BATCH_SIZE,
    seed: int | None = None,
) -> list[str]:
    """Steered generation: RL fine-tuning with the pharmacophore reward as the
    live signal (REINVENT4 ``staged_learning`` run mode + our custom scoring
    component, registered via app/reinvent_plugins/).

    Returns the SMILES of every valid unique molecule generated across the
    run (REINVENT writes one summary CSV per stage; later steps are the most
    strongly steered, and the CSV's step column keeps the chronology).
    Details land in the module-level ``last_rl_run_info`` dict.

    Defaults are CPU-honest, NOT the REINVENT-tutorial 200 steps: on this
    1-core box a step takes ~4 s at batch 64 (measured), so 30 steps ≈
    2.5 min wall including ~30 s startup, while 200 steps ≈ 14 min and
    would blow the timeout. Raises ValueError for budgets > the 240 s cap.
    Raises GenerationError when the subprocess fails.
    """
    import reward_plugin

    num_steps = int(num_steps)
    batch_size = int(batch_size)
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not reference_signature:
        raise ValueError("reference_signature is empty — no reward signal to steer with")

    estimated_s = RL_STARTUP_BUDGET_S + num_steps * RL_PER_STEP_BUDGET_S
    if estimated_s > RL_TIMEOUT_S - 30:  # keep headroom for scoring/ranking after the run
        raise ValueError(
            f"num_steps={num_steps} at batch_size={batch_size} is not feasible on this CPU: "
            f"~{num_steps * RL_PER_STEP_BUDGET_S:.0f}s of stepping + {RL_STARTUP_BUDGET_S:.0f}s "
            f"startup exceeds the {RL_TIMEOUT_S}s budget. Use num_steps <= {RL_MAX_NUM_STEPS} ({RL_DEFAULT_NUM_STEPS} is the default)."
        )

    _check_reinvent_available()

    with tempfile.TemporaryDirectory(prefix="reinvent_rl_") as tmp:
        tmpdir = Path(tmp)
        scoring_path = tmpdir / "stage1_scoring.toml"
        scoring_path.write_text(reward_plugin.build_stage_scoring_toml(reference_signature))

        config = tmpdir / "staged_learning.toml"
        log = tmpdir / "reinvent_rl.log"
        # Schema follows reinvent/runmodes/RL/validation.py (v4.8.24).  NB: the
        # shipped configs/staged_learning.toml example sets
        # `unique_sequences = true`, which this version's RLConfig REJECTS
        # (extra_forbidden) — verified — so it is intentionally absent here.
        lines = [
            'run_type = "staged_learning"',
            'device = "cpu"',
            'json_out_config = "_staged_learning.json"',
            "",
            "[parameters]",
            f'prior_file = "{PRIOR}"',
            f'agent_file = "{PRIOR}"',
            'summary_csv_prefix = "rl"',
            "use_checkpoint = false",
            "purge_memories = false",
            f"batch_size = {batch_size}",
            "randomize_smiles = true",
            "",
            "[learning_strategy]",
            'type = "dap"',
            "sigma = 128",
            "rate = 0.0001",
            "",
            "[[stage]]",
            'chkpt_file = "agent.chkpt"',
            'termination = "simple"',
            "max_score = 1.0",
            "min_steps = 0",
            f"max_steps = {num_steps}",
            "",
            "[stage.scoring]",
            'type = "geometric_mean"',
            'filename = "stage1_scoring.toml"',
            'filetype = "toml"',
        ]
        config.write_text("\n".join(lines) + "\n")

        cmd = [str(REINVENT_BIN), str(config), "-l", str(log)]
        if seed is not None:
            cmd += ["--seed", str(int(seed))]

        started = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=RL_TIMEOUT_S,
                cwd=str(tmpdir),
                env=_reinvent_subprocess_env(),
            )
        except subprocess.TimeoutExpired as err:
            raise GenerationError(
                f"RL run exceeded {RL_TIMEOUT_S}s (num_steps={num_steps}, batch_size={batch_size}). "
                "Reduce num_steps on this CPU."
            ) from err
        elapsed = time.time() - started

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-800:]
            raise GenerationError(f"REINVENT RL failed after {elapsed:.1f}s:\n{tail}")

        csv_files = sorted(tmpdir.glob("rl_*.csv"))
        if not csv_files:
            raise GenerationError(
                f"REINVENT RL produced no summary CSV after {elapsed:.1f}s\n"
                f"{(proc.stderr or proc.stdout or '')[-400:]}"
            )

        # Summary CSV (per stage, from reinvent/runmodes/RL/reports/csv_summmary.py):
        # Agent, Prior, Target, Score, SMILES, SMILES_state, <component name>,
        # <component name> (raw), step — batch_size rows per step.
        # SMILES_state: 1 = valid, 2 = duplicate, 0 = invalid.
        all_smiles: list[str] = []
        seen: set[str] = set()
        rows_by_step: dict[int, list[float]] = {}
        csv_rows = 0
        for csv_file in csv_files:
            with open(csv_file, newline="") as fh:
                for row in csv.reader(fh):
                    if not row or row[0] == "Agent":
                        continue  # header
                    csv_rows += 1
                    if len(row) < 6 or row[5].strip() != "1":
                        continue  # invalid or duplicate
                    smi = row[4].strip()
                    if not smi:
                        continue
                    if smi not in seen:
                        seen.add(smi)
                        all_smiles.append(smi)
                    try:
                        rows_by_step.setdefault(int(row[-1]), []).append(float(row[3]))
                    except ValueError:
                        pass

        step_scores = {step: sum(v) / len(v) for step, v in sorted(rows_by_step.items())}
        last_rl_run_info.clear()
        last_rl_run_info.update(
            {
                "num_steps": num_steps,
                "batch_size": batch_size,
                "steps_run": max(rows_by_step) if rows_by_step else 0,
                "elapsed_s": round(elapsed, 1),
                "csv_rows": csv_rows,
                "valid_unique_smiles": len(all_smiles),
                "mean_reward_first_step": round(next(iter(step_scores.values())), 3)
                if step_scores
                else None,
                "mean_reward_last_step": round(step_scores[max(step_scores)], 3)
                if step_scores
                else None,
                "mode": "rl_staged_learning",
            }
        )
        return all_smiles
