# GenUI — What It Is and What We Do With It

**Short answer: we cloned it as reference material, we do NOT run it.**

---

## What GenUI actually is

A research web platform (paper: Sicho et al. 2021) for de novo molecular
generation, QSAR modelling and chemical-space maps. It is **not a small
demo** — it's a whole platform:

| Piece | Repo | What it needs to run |
|---|---|---|
| Backend API | `martin-sicho/genui` (Django 4.1, last commit 2022-09) | Postgres, Redis, Celery workers, Docker |
| Web frontend | `martin-sicho/genui-gui` (React 18, last commit 2022-09) | talks to the backend API |
| Deployment | `martin-sicho/genui-docker` | Docker Compose stack |

## Why we don't run it (ELI5)

> GenUI is like a huge, fancy restaurant kitchen from a cookbook. We wanted
> a look at its floor plan for ideas — we did **not** want to cook with it.

1. **It's a stack, not a tool** — Postgres + Redis + Celery + Django just to
   generate molecules. Our pipeline calls REINVENT4/AiZynthFinder directly.
2. **Frozen in time** — backend pins 2021-era libraries (`numpy==1.21`,
   `rdkit-pypi==2021.3.5`, Django 4.1). Installing it would mean building a
   fragile old Python world next to our fresh one.
3. **Wrong generator built in** — it integrates **DrugEx v3**, not REINVENT4.
   Wiring REINVENT4 in would be a custom port anyway... which is exactly what
   our own app already is.
4. **Our tooling is simpler now** — a single small server + RDKit covers the
   preview needs that made us look at GenUI in the first place.

## What we keep from it (the reference value)

We cloned both repos to `tools/genui` and `tools/genui-gui` for study:

- **UX flow:** how a generation job is submitted, watched (progress bar), and
  browsed afterwards — see `genui-gui/src/views/pages` (dashboard-style
  widgets) and the generators app in `genui/src/genui/generators`.
- **What a molecule "result" should carry:** SMILES, computed descriptors,
  scores, and drawn structure — see `genui/src/genui/compounds`.
- **Frontend patterns:** React dashboard widgets (`react-grid-layout`,
  `react-plotly.js`) and a REST-API-driven UI.

## The important discovery: we already have previews

We don't need GenUI to *see* molecules. RDKit (installed for REINVENT4) draws
structures as SVG from any SMILES — e.g. `runs/preview_aspirin.svg` was
generated with exactly the same RDKit our pipeline uses. In our own app this
becomes: *type a SMILES → see the molecule, instantly.*

## When would running GenUI make sense?

Only if you someday want its QSAR-model-building and chemical-space-map
features as a product. For this pipeline, running it would be all cost,
no benefit. If you're curious anyway: `genui-docker` is the one-command path
(requires Docker on your machine).
