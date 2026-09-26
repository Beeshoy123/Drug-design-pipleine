"""Drug-design pipeline — Step 1 preview app.

A tiny FastAPI server that shows what our pipeline's UI will feel like:
type a molecule (SMILES), see it drawn by RDKit with its basic properties.
This is the first brick of the front-end that will later wrap REINVENT4
(Step 3) and AiZynthFinder (Step 4).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from generator import GenerationError, sample_molecules
from pharmacophore import draw_with_features, extract_pharmacophore, signature_similarity
from scoring import rank_candidates, to_dicts

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
    """Step 3: REINVENT4 invents molecules; we rank them against the reference."""
    ref_mol = _validate(reference)
    ref_canon = Chem.MolToSmiles(ref_mol)
    ref_ph = extract_pharmacophore(ref_canon, with_3d=False)

    try:
        raw = sample_molecules(num, seed=seed)
    except GenerationError as err:
        raise HTTPException(status_code=503, detail=str(err)) from err

    ranked = rank_candidates(raw, ref_ph.signature)[:top]
    results = to_dicts(ranked)
    for r in results:
        r["svg"] = draw_with_features(r["smiles"], 240)
    return JSONResponse(
        {
            "reference": ref_canon,
            "reference_signature": ref_ph.signature,
            "requested": num,
            "generated": len(raw),
            "returned": len(results),
            "candidates": results,
        }
    )


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
