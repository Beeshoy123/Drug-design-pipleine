# External Tools

The actual tool code is cloned from GitHub and **not** committed to this repo
(each tool has its own git history). Re-clone with the commands below.

## REINVENT4 — the "recipe inventor"

- **Repo:** https://github.com/MolecularAI/REINVENT4
- **Cloned at:** tag `v4.8` (v4.8.24), commit `660d2c9cec9ea1ced1b452394fd38452e478fc8b` (2026-09-23)
- **Role in pipeline:** Step 3 — invents new drug-like molecules (SMILES strings)

```shell
git clone https://github.com/MolecularAI/REINVENT4.git tools/REINVENT4
cd tools/REINVENT4 && git checkout v4.8
```

Setup (Python ≥ 3.11 required — use `uv`). **Installed in this repo:**
`.venv-reinvent` (Python 3.12, CPU-only PyTorch 2.12.0). Recreate anytime with
`sh ./scripts/setup_envs.sh`.

```shell
# 1. system libs RDKit needs for drawing (Debian/Ubuntu)
sudo apt-get install -y libxrender1 libxext6 libsm6
# 2. environment
uv venv .venv-reinvent --python 3.12
uv pip install --python .venv-reinvent/bin/python torch==2.12.0 torchvision \
    --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-reinvent/bin/python scipy   # assumed by plotting code
uv pip install --python .venv-reinvent/bin/python tools/REINVENT4
.venv-reinvent/bin/reinvent --version
# => REINVENT 4.8.24 (C) AstraZeneca 2017, 2023 using PyTorch 2.12.0+cpu.
```

Notes: installed with `-d none`-equivalent extras (no OpenEye — it needs a
paid license; `scipy` added by hand because REINVENT4's plotting code assumes
it comes with the extras). Runs on CPU automatically when no GPU is present.

Prior models (the pre-trained "brains") are downloaded separately from
[Zenodo](https://doi.org/10.5281/zenodo.15641296) and expected in
`tools/REINVENT4/priors/`.

## AiZynthFinder — the "can we actually make it?" checker

- **Repo:** https://github.com/MolecularAI/aizynthfinder
- **Cloned at:** tag `v4.4.1` (v4.4.1), commit `21ff546d5f22331b078390a2f12dc04defc3f39c` (2026-04-13)
- **Role in pipeline:** Step 4 — retrosynthesis: works backwards from a molecule to purchasable building blocks

```shell
git clone https://github.com/MolecularAI/aizynthfinder.git tools/aizynthfinder
cd tools/aizynthfinder && git checkout v4.4.1
```

Setup (Python ≥ 3.10, < 3.13). **Installed in this repo:** `.venv-aizynth`
(Python 3.12) from the cloned source. Recreate anytime with
`sh ./scripts/setup_envs.sh`.

```shell
uv venv .venv-aizynth --python 3.12
uv pip install --python .venv-aizynth/bin/python tools/aizynthfinder
.venv-aizynth/bin/aizynthcli --help   # works; rdchiral SyntaxWarnings are harmless
```

We skip the optional TensorFlow extra — the default policy networks run on
the much lighter `onnxruntime`.

Reaction-model + stock data files come from figshare, downloaded once with:

```shell
download_public_data my_folder   # writes model files + config.yml
```

Handy to know: it ships three entry points — `aizynthcli` (command line),
`aizynthapp` (a minimal built-in web UI), and the Python API we'll use in the
pipeline.

## GenUI — reference only, do not run

- **Backend:** https://github.com/martin-sicho/genui — cloned `tools/genui`
  (shallow, tip `992b72a`, 2022-09)
- **Frontend:** https://github.com/martin-sicho/genui-gui — cloned
  `tools/genui-gui` (shallow, tip `db1faee`, 2022-09)
- **Role:** design reference for our own front-end (job flow, result fields,
  dashboard widgets). **Not installed** — see `docs/Genui_NOTES.md` for why
  (Django+Postgres+Redis+Docker stack, 2021-era pins, integrates DrugEx not
  REINVENT4).

```shell
git clone --depth 1 https://github.com/martin-sicho/genui.git tools/genui
git clone --depth 1 https://github.com/martin-sicho/genui-gui.git tools/genui-gui
```

## Roadmap

- [x] REINVENT4 (github.com/MolecularAI/REINVENT4) — Step 3, the AI chef
- [x] AiZynthFinder (github.com/MolecularAI/aizynthfinder) — Step 4, "can we actually make it?"
- [x] GenUI + genui-gui (martin-sicho) — reference for our front-end, not installed
- [ ] Pipeline glue + front-end app
