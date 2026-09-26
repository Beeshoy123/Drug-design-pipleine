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


# Gaussian width (in Angstrom) for geometry_similarity.  Why a Gaussian instead
# of a hard 1-2 A cutoff:  feature positions come from centroids of SMARTS
# atom groups in ONE embedded conformer, so distances carry ~1 A of conformer
# noise even between perfect matches; a hard threshold would flip matches on
# tiny perturbations.  exp(-((d_ref-d_cand)/1.5)^2) gives ~0.64 credit at 1 A
# deviation and ~0.17 at 2 A — smooth, symmetric, and lenient exactly where
# the noise is.  Same spirit as the sigmoid transforms REINVENT4 uses.
GEOMETRY_SIGMA_A = 1.5


def _distance_pair_family(entry: dict) -> tuple[str, str]:
    """Family pair of a distances entry, order-normalized ("donor", "acceptor")."""
    fam_a = entry["a"].split("#")[0]
    fam_b = entry["b"].split("#")[0]
    return (fam_a, fam_b) if fam_a <= fam_b else (fam_b, fam_a)


def _distances_by_pair(ph: Pharmacophore) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for entry in ph.distances:
        grouped.setdefault(_distance_pair_family(entry), []).append(entry["angstrom"])
    return grouped


def geometry_similarity(ref: Pharmacophore, candidate: Pharmacophore) -> float:
    """0-1 score: how well the candidate's 3D inter-feature distances match the
    reference's, regardless of which specific atoms realize them.

    Method (soft symmetric matching, a smoothed Chamfer distance):
      For each feature-pair type present in EITHER molecule (e.g. donor-acceptor)
      every candidate distance of that type is credited by the Gaussian of its
      nearest reference distance, and symmetrically every reference distance by
      its nearest candidate distance; the two means are averaged.  This is
      order-free — no greedy pairing, so extra features on either side simply
      dilute their side's mean instead of crashing or being ignored.

    Unmatched feature-pair types (in the union) contribute 0.0: a molecule
    missing an entire interaction motif is materially different, so the penalty
    is proportional (1/n_types) rather than catastrophic — counts of remaining
    motifs can still be excellent.

    Requires both Pharmacophores to be 3D-embedded (has_3d=True); raises
    ValueError otherwise — never fall back silently, that would hide exactly
    the shape mismatch this function exists to catch.
    """
    if not (ref.has_3d and candidate.has_3d):
        raise ValueError(
            "geometry_similarity needs 3D-embedded pharmacophores "
            "(extract_pharmacophore(..., with_3d=True) on both sides)"
        )

    ref_groups = _distances_by_pair(ref)
    cand_groups = _distances_by_pair(candidate)
    if not ref_groups and not cand_groups:
        return 1.0  # fewer than two features each: no geometry to disagree about

    def group_score(ref_d: list[float] | None, cand_d: list[float] | None) -> float:
        if not ref_d or not cand_d:
            return 0.0  # type exists on one side only -> treated as full mismatch
        sigma2 = 2 * GEOMETRY_SIGMA_A**2

        def side_mean(sources: list[float], targets: list[float]) -> float:
            return sum(
                max(math.exp(-((s - t) ** 2) / sigma2) for t in targets)
                for s in sources
            ) / len(sources)

        return (side_mean(cand_d, ref_d) + side_mean(ref_d, cand_d)) / 2

    all_types = set(ref_groups) | set(cand_groups)
    total = sum(group_score(ref_groups.get(t), cand_groups.get(t)) for t in all_types)
    return round(total / len(all_types), 4)


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
