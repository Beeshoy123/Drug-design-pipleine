"""Drug-design pipeline — web app.

Step 1: SMILES (or drug NAME via PubChem) → drawn molecule + properties.
Step 2: pharmacophore pattern extraction.
Step 3: REINVENT4 generation scored against the reference pattern.
Step 4: AiZynthFinder buildability checks.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from buildability import BuildabilityError, get_checker
from generator import (
    GenerationError,
    RL_MAX_NUM_STEPS,
    last_rl_run_info,
    run_staged_learning,
    sample_molecules,
)
from pharmacophore import draw_with_features, extract_pharmacophore, signature_similarity
from scoring import GEOMETRY_STAGE_TOP_N, rank_candidates, to_dicts

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

app = FastAPI(title="Drug-Design Pipeline — Step 1 Preview")

TEMPLATES = Path(__file__).parent / "templates"


def _validate(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {smiles!r}")
    return mol


def _mol_to_svg(mol: Chem.Mol, size: int = 320) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _props(mol: Chem.Mol) -> dict:
    return {
        "name": Chem.MolToSmiles(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 1),
        "logp": round(Crippen.MolLogP(mol), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "lipinski_violations": sum(
            [
                Descriptors.MolWt(mol) > 500,
                Crippen.MolLogP(mol) > 5,
                Lipinski.NumHDonors(mol) > 5,
                Lipinski.NumHAcceptors(mol) > 10,
            ]
        ),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((TEMPLATES / "index.html").read_text())


@app.get("/api/molecule")
def molecule(
    smiles: str = Query(..., description="Molecule as a SMILES string"),
    size: int = Query(320, ge=100, le=800),
) -> JSONResponse:
    mol = _validate(smiles)
    return JSONResponse({"smiles": smiles, "svg": _mol_to_svg(mol, size), "props": _props(mol)})


@app.get("/api/pharmacophore")
def pharmacophore_endpoint(
    smiles: str = Query(..., description="Molecule as a SMILES string"),
    size: int = Query(320, ge=100, le=800),
) -> JSONResponse:
    """Step 2: the 'bumps & notches' pattern of the molecule."""
    try:
        ph = extract_pharmacophore(smiles, with_3d=True)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return JSONResponse(
        {
            "smiles": ph.smiles,
            "has_3d": ph.has_3d,
            "features": [
                {"family": f.family, "atoms": f.atoms, "centroid": f.centroid}
                for f in ph.features
            ],
            "signature": ph.signature,
            "distances": sorted(ph.distances, key=lambda d: d["angstrom"])[:8],
            "svg_highlighted": draw_with_features(ph.smiles, size),
        }
    )


@app.get("/api/generate")
def generate(
    reference: str = Query(..., description="Reference molecule SMILES to match"),
    num: int = Query(100, ge=10, le=500, description="How many molecules to invent"),
    seed: int | None = Query(None, description="Random seed for reproducibility"),
    top: int = Query(12, ge=1, le=50, description="How many top candidates to return"),
) -> JSONResponse:
    """Step 3: REINVENT4 invents molecules; we rank them against the reference.

    The reference may be a SMILES *or* a drug name (e.g. "sorafenib") —
    resolved server-side via PubChem so every UI path accepts both.
    """
    ref_mol = _resolve_reference(reference)
    ref_canon = Chem.MolToSmiles(ref_mol)
    # 3D-embed the reference ONCE; candidates are embedded only for the
    # geometry stage (top-N by count score) inside rank_candidates.
    ref_ph = extract_pharmacophore(ref_canon, with_3d=True)

    try:
        raw = sample_molecules(num, seed=seed)
    except GenerationError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    ranked = rank_candidates(
        raw,
        ref_ph.signature,
        ref_pharmacophore=ref_ph,
        geometry_top_n=max(GEOMETRY_STAGE_TOP_N, top),
    )[:top]
    results = to_dicts(ranked)
    for r in results:
        r["svg"] = draw_with_features(r["smiles"], 240)
    return JSONResponse(
        {
            "reference": ref_canon,
            "reference_signature": ref_ph.signature,
            "scoring_mode": "counts+geometry" if ref_ph.has_3d else "counts_only",
            "reference_has_3d": ref_ph.has_3d,
            "requested": num,
            "generated": len(raw),
            "returned": len(results),
            "candidates": results,
        }
    )


@app.get("/api/generate_rl")
def generate_rl(
    reference: str = Query(..., description="Reference molecule SMILES (or name) to steer toward"),
    num_steps: int = Query(30, ge=1, le=RL_MAX_NUM_STEPS, description="RL steps — CPU budget caps this at 45; 200 would take ~14 min"),
    batch_size: int = Query(64, ge=8, le=128, description="Molecules sampled per RL step"),
    seed: int | None = Query(None, description="Random seed for reproducibility"),
    top: int = Query(12, ge=1, le=50, description="How many top candidates to return"),
) -> JSONResponse:
    """Step 3, steered mode: actual RL fine-tuning against the pharmacophore reward.

    Same shape as /api/generate, but the reward is applied DURING generation
    (REINVENT4 staged_learning with a live scoring component) instead of only
    post-hoc. Costs ~2.5 min on this 1-core CPU (vs ~10 s for /api/generate);
    30 steps visibly steers the agent but is not a converged RL run.
    """
    ref_mol = _resolve_reference(reference)
    ref_canon = Chem.MolToSmiles(ref_mol)
    ref_ph = extract_pharmacophore(ref_canon, with_3d=False)

    try:
        raw = run_staged_learning(ref_ph.signature, num_steps=num_steps, batch_size=batch_size, seed=seed)
    except GenerationError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err

    ranked = rank_candidates(raw, ref_ph.signature)[:top]
    results = to_dicts(ranked)
    for r in results:
        r["svg"] = draw_with_features(r["smiles"], 240)
    return JSONResponse(
        {
            "reference": ref_canon,
            "reference_signature": ref_ph.signature,
            "mode": "rl_staged_learning",
            "requested_steps": num_steps,
            "batch_size": batch_size,
            "rl_run": dict(last_rl_run_info),
            "generated": len(raw),
            "returned": len(results),
            "candidates": results,
        }
    )


def _resolve_reference(text: str) -> Chem.Mol:
    """Accept a SMILES *or* a drug name (PubChem lookup); return a Mol."""
    mol = Chem.MolFromSmiles(text)
    if mol is not None:
        return mol
    smiles, _ = _pubchem_lookup(text.strip().lower())
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        raise HTTPException(status_code=502, detail="PubChem returned an unusable SMILES")
    raise HTTPException(
        status_code=422,
        detail=f"'{text}' is neither a valid SMILES nor a known compound name (PubChem)",
    )


class BuildabilityRequest(BaseModel):
    smiles: list[str]


@app.post("/api/buildability")
def buildability(req: BuildabilityRequest) -> JSONResponse:
    """Step 4: retrosynthesis check — can these molecules actually be made?

    Each check takes ~8-10 s (tree search against reaction rules on a single
    CPU); we cap the batch at 10 molecules.
    """
    canonical: list[str] = []
    for smi in req.smiles[:10]:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical.append(Chem.MolToSmiles(mol))
    if not canonical:
        raise HTTPException(status_code=422, detail="No valid SMILES in request")

    try:
        results = get_checker().check_smiles_batch(canonical)
    except BuildabilityError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    return JSONResponse({"checked": len(results), "results": results})


@app.get("/api/resolve")
def resolve(name: str = Query(..., description="Drug or compound name, e.g. sorafenib")) -> JSONResponse:
    """Look up a compound by (trade/INN/IUPAC) name via PubChem's free API."""
    q = name.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Empty name")

    smiles, label = _pubchem_lookup(q.lower())
    if smiles is None:
        raise HTTPException(status_code=404, detail=f"Could not find {q!r} in PubChem")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # PubChem returned something RDKit dislikes — extremely rare
        raise HTTPException(status_code=502, detail="PubChem returned an unusable SMILES")

    return JSONResponse(
        {
            "query": q,
            "label": label or q,
            "smiles": Chem.MolToSmiles(mol),
        }
    )


@lru_cache(maxsize=256)
def _pubchem_lookup(name: str) -> tuple[str | None, str | None]:
    """PubChem PUG REST name lookup; cached so repeat queries don't re-hit the API."""
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name)}/property/SMILES,ConnectivitySMILES,Title/JSON"
    )
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "silico-pipeline/0.1"})
    except requests.RequestException:
        return None, None
    if resp.status_code != 200:
        return None, None
    try:
        props = resp.json()["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError, ValueError):
        return None, None
    smiles = props.get("ConnectivitySMILES") or props.get("SMILES")
    return (smiles, props.get("Title")) if smiles else (None, None)


@app.get("/api/examples")
def examples() -> JSONResponse:
    return JSONResponse(
        [
            {"label": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"label": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
            {"label": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
            {"label": "Penicillin G", "smiles": "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"},
        ]
    )
