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
- ✅ **Step 3 generation live** → `app/generator.py` + `app/scoring.py`: REINVENT4 (CPU, `reinvent_pubchem.prior` from Zenodo) invents ~100 valid molecules in seconds, each scored against the reference pattern (60% signature match / 25% QED / 15% Lipinski, geometric mean) and ranked in the UI
- ⬜ Prior models + reaction data (Zenodo / figshare)
- ⬜ Pipeline glue (RDKit pharmacophores → REINVENT4 → AiZynthFinder)
- ⬜ Web front-end
