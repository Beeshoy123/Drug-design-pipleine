# Drug-Design Pipeline

A hobby pipeline that **invents new drug-like molecules, checks they can
actually be made, and reports the best candidates** — built as a friendly
front-end around proven open-source chemistry tools.

> 🧒 **New here?** Read [`docs/ELI5_PIPELINE.md`](docs/ELI5_PIPELINE.md) —
> the whole plan explained like you're 5, with analogies.

## The Plan

| Step | Tool | What it does |
|---|---|---|
| 1 | *(you)* | Give a reference molecule you like (SMILES string) |
| 2 | **RDKit** | Extract the **pharmacophore** — the 3D "bumps & notches" pattern that makes it active |
| 3 | **REINVENT4** | AI chef invents new molecules matching that pattern, trained by your scoring rules |
| 4 | **AiZynthFinder** | Checks each invention is actually buildable ("can we buy the ingredients?") |
| 5 | *(our app)* | Ranked report of the best, makeable candidates |

REINVENT4 and AiZynthFinder are command-line only — our job is building the
app around them. [GenUI](https://github.com/martin-sicho/genui) is kept as a
design reference only (see [`docs/Genui_NOTES.md`](docs/Genui_NOTES.md)).

## Repository Layout

```
app/               web app (FastAPI + RDKit): SMILES → molecule preview + properties
scripts/           environment setup
runs/              generated molecules, previews, reports (git-ignored)
tools/             cloned tool repos (git-ignored, see tools/README.md)
docs/              explanations and design notes
```

## Run the App

```shell
sh ./scripts/setup_envs.sh                                        # once (idempotent)
.venv-reinvent/bin/python -m uvicorn main:app --app-dir app \
    --host 0.0.0.0 --port 8080                                    # serve
```

Then open `http://localhost:8080`, type a SMILES (or click an example), and
see the molecule drawn with its drug-properties. In the Freebuff workspace
the preview commands are already registered (`sh ./scripts/setup_envs.sh` as
install, uvicorn on port 8080 as the dev server).

## Status

- ✅ **REINVENT4 cloned** → `tools/REINVENT4` (v4.8.24) — setup instructions in [`tools/README.md`](tools/README.md)
- ✅ **AiZynthFinder cloned** → `tools/aizynthfinder` (v4.4.1) — setup instructions in [`tools/README.md`](tools/README.md)
- ✅ **Environments installed** → `.venv-reinvent` + `.venv-aizynth` (Python 3.12, CPU torch), recreate with `sh ./scripts/setup_envs.sh` — both tools verified working
- ✅ **GenUI evaluated** → cloned as design reference only, not run — see [`docs/Genui_NOTES.md`](docs/Genui_NOTES.md); RDKit SVG previews verified working (`runs/preview_*.svg`)
- ✅ **Step 1 preview app live** → `app/` FastAPI server: SMILES → RDKit SVG + drug-properties (weight, LogP, TPSA, Lipinski check), served via Freebuff preview
- ✅ **Step 2 pharmacophore extraction live** → `app/pharmacophore.py`: 6 feature families (SMARTS), 3D embedding + inter-feature distances, highlighted depiction, and a feature-signature similarity score ready for Step 3 filtering
- ✅ **Step 3 generation live** → `app/generator.py` + `app/scoring.py`: REINVENT4 (CPU, `reinvent_pubchem.prior` from Zenodo) invents ~100 valid molecules in seconds, each scored against the reference pattern — 35% signature match + 25% 3D geometry match (top-24 only, Gaussian soft distance matching) + 25% QED + 15% Lipinski, weighted geometric mean — and ranked in the UI
- ✅ **Step 3 RL-steered generation live** → `/api/generate_rl`: REINVENT4 `staged_learning` fine-tunes the agent *during* generation against the pharmacophore reward, via a custom scoring component (`app/reinvent_plugins/components/comp_pharmacophore_similarity.py`, registered through REINVENT4's plugin mechanism). CPU-honest defaults: 30 steps × batch 64 ≈ 2.5 min on the 1-core box (200 steps ≈ 14 min and is refused). Measured effect: mean reward 0.19 → 0.28 over the run and top candidates reach 100% pattern match vs 83% unsteered
- ✅ **Step 4 buildability live** → `app/aizynth_worker.py` + `app/buildability.py`: AiZynthFinder retrosynthesis in a worker subprocess (ONNX USPTO models + 11 MB molbloom ZINC filter — light enough for 1.9 GB RAM), returns makeable ✓/✗, route score, step count, purchasable building blocks, and the ordered synthesis route (each disconnection step with precursor SMILES and in-stock flags); UI badge per candidate with route tooltip
- ✅ **Step 5 full pipeline report live** → one-click "Run whole pipeline": invent 100 → rank → check top 5 makeability → final ranked winners table on screen with building blocks **and the retrosynthesis routes**, plus CSV and print-to-PDF export
- 🏁 **The original 5-step plan is complete.** Optional upgrades: full ZINC stock on a bigger machine, 3D pharmacophore scoring in REINVENT, RL fine-tuning runs
