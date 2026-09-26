"""Step 2 — Pharmacophore extraction (the "bumps & notches" pattern).

Turns a molecule into a list of *features* (donors, acceptors, aromatic
rings, hydrophobes, positive/negative spots) plus the distances between
them. Two molecules that share the same features in the same arrangement
can act on the body the same way — that is the whole trick of Step 2.

For the hobby pipeline we use 2D SMARTS feature detection (fast, robust)
plus an optional 3D embedding (ETKDG) for inter-feature distances. The
output also contains a compact "signature" (feature counts) that Step 3
will use as a quick similarity score for generated molecules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# Feature family -> list of SMARTS patterns. Kept small and understandable
# on purpose; these are pragmatic approximations, not a production rule set.
FEATURE_DEFS: dict[str, list[str]] = {
    "donor": ["[#7,#8;!H0]"],  # any N or O carrying at least one H (incl. aromatic NH)
    "acceptor": [
        "[#8;H0;!$([#8-])]",  # neutral O with no H (carbonyl, ester, ether, ...)
        "[nX2;H0]",  # pyridine-like aromatic N (lone pair not in the ring)
        "[NX3;H0;!$(N[C,S,P]=O)]",  # amine N, not amide/nitro
    ],
    "aromatic_ring": ["c1ccccc1"],
    "hydrophobe": ["[c]1[c][c][c][c]1~[#6;!R]", "[#6;!R;!H0]~[#6;!R;!H0]"],
    "positive": ["[+;!$([+1]~[-1])]"],
    "negative": ["[-;!$([-1]~[+1])]"],
}

FAMILY_COLORS = {
    "donor": (0.2, 0.8, 0.4),
    "acceptor": (0.9, 0.3, 0.3),
    "aromatic_ring": (0.3, 0.5, 0.95),
    "hydrophobe": (0.85, 0.65, 0.2),
    "positive": (0.55, 0.35, 0.9),
    "negative": (0.9, 0.5, 0.15),
}


@dataclass
class PharmacophoreFeature:
    family: str
    atoms: list[int]
    centroid: list[float] | None = None  # 3D coordinates, if embedding worked


@dataclass
class Pharmacophore:
    smiles: str
    features: list[PharmacophoreFeature] = field(default_factory=list)
    distances: list[dict] = field(default_factory=list)  # {"a","b","angstrom"}
    has_3d: bool = False
    signature: dict[str, int] = field(default_factory=dict)  # family -> count


def _embed(mol: Chem.Mol) -> Chem.Mol | None:
    """Try to add a sensible 3D conformer (ETKDG + MMFF relax)."""
    work = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D  # reproducible
    if AllChem.EmbedMolecule(work, params) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(work, maxIters=500)  # best-effort relax
    return work


def _feature_atoms(mol: Chem.Mol, family: str) -> list[list[int]]:
    """All unique atom-group matches for a family (deduped, sorted)."""
    found: set[tuple[int, ...]] = set()
    for smarts in FEATURE_DEFS[family]:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            found.add(tuple(match))
    return [list(t) for t in sorted(found)]


def extract_pharmacophore(smiles: str, with_3d: bool = True) -> Pharmacophore:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    canon = Chem.MolToSmiles(mol)
    work = _embed(mol) if with_3d else None

    ph = Pharmacophore(smiles=canon)
    for family in FEATURE_DEFS:
        for atoms in _feature_atoms(mol, family):
            feat = PharmacophoreFeature(family=family, atoms=atoms)
            if work is not None:
                conf = work.GetConformer()
                pts = [conf.GetAtomPosition(a) for a in atoms]
                feat.centroid = [
                    sum(p.x for p in pts) / len(pts),
                    sum(p.y for p in pts) / len(pts),
                    sum(p.z for p in pts) / len(pts),
                ]
            ph.features.append(feat)

    if work is not None:
        ph.has_3d = True
        for i, f1 in enumerate(ph.features):
            for j in range(i + 1, len(ph.features)):
                f2 = ph.features[j]
                if f1.centroid is None or f2.centroid is None:
                    continue
                ph.distances.append(
                    {
                        "a": f"{f1.family}#{i}",
                        "b": f"{f2.family}#{j}",
                        "angstrom": round(math.dist(f1.centroid, f2.centroid), 2),
                    }
                )

    counts: dict[str, int] = {}
    for f in ph.features:
        counts[f.family] = counts.get(f.family, 0) + 1
    ph.signature = counts
    return ph


def signature_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    """Quick 0-1 score: how well two feature-count signatures overlap."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    intersection = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return intersection / union


def draw_with_features(smiles: str, size: int = 320) -> str:
    """2D depiction with feature atoms highlighted (one color per family)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    highlight_atoms: list[int] = []
    highlight_colors: dict[int, tuple[float, float, float]] = {}
    for family, color in FAMILY_COLORS.items():
        for atoms in _feature_atoms(mol, family):
            for a in atoms:
                if a not in highlight_colors:
                    highlight_atoms.append(a)
                    highlight_colors[a] = color

    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
