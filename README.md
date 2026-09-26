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
app around them (inspired by [GenUI](https://github.com/MolecularAI/GUI-Generative-UI)).

## Repository Layout

```
tools/REINVENT4/   cloned tool repos (git-ignored, see tools/README.md)
docs/              explanations and design notes
```

## Status

- ✅ **REINVENT4 cloned** → `tools/REINVENT4` (v4.8.24) — setup instructions in [`tools/README.md`](tools/README.md)
- ✅ **AiZynthFinder cloned** → `tools/aizynthfinder` (v4.4.1) — setup instructions in [`tools/README.md`](tools/README.md)
- ✅ **Environments installed** → `.venv-reinvent` + `.venv-aizynth` (Python 3.12, CPU torch), recreate with `sh ./scripts/setup_envs.sh` — both tools verified working
- ⬜ Prior models + reaction data (Zenodo / figshare)
- ⬜ Pipeline glue (RDKit pharmacophores → REINVENT4 → AiZynthFinder)
- ⬜ Web front-end
