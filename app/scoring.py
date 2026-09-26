"""Step 3 — Scoring: rank candidate molecules against a reference molecule.

The score is a weighted mix of four explainable parts:

1. **Pattern match** (`sig_sim`) — how well the candidate's pharmacophore
   signature (feature *counts*, 2D) overlaps the reference's.
2. **Geometry match** (`geometry_sim`) — 3D refinement: how well the
   inter-feature *distances* agree with the reference (Gaussian soft matching,
   see pharmacophore.geometry_similarity).  ``None`` when 3D embedding was not
   computed for a candidate (the two-stage filter keeps this rare).
3. **Drug-likeness** (`qed`) — RDKit's QED estimate (0-1), a rough
   "does this look like a medicine" score learned from real drugs.
4. **Lipinski** (`lipinski`) — 1.0 with zero rule-of-5 violations, minus
   0.34 per violation (so one slip is forgivable, many are not).

All terms live in [0, 1] and combine by weighted geometric mean — the same
friendly aggregation REINVENT itself uses. Geometric mean punishes being bad
at one thing harder than arithmetic averaging, which is what we want: a
molecule that matches the pattern perfectly but looks poisonous should score
low.

**Why the weights are split the way they are.**  Pattern and geometry both
measure "is this the same *kind* of key", so pharmacophore evidence as a whole
keeps the original 60% share — but split 35/25.  Counts get more than geometry
because counts are cheaper, conformer-independent, and every candidate has
them, so they anchor the ranking; geometry is the arbiter among candidates the
counts already like, and it inherits exactly the 25% QED held before — the
pattern/geometry pair answers *what* the molecule does, QED/Lipinski answer
*whether it is a sensible drug*, and Lipinski stays at the original 15%.
Concretely: within the 60% pharmacophore credit, a candidate with perfect
counts but wrong 3D shape lands at 0.35/0.60 = 58% of that credit — worse than
a strong-but-not-perfect all-rounder, which is the intended behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from pharmacophore import (
    Pharmacophore,
    extract_pharmacophore,
    geometry_similarity,
    signature_similarity,
)

WEIGHTS = {"sig_sim": 0.35, "geometry_sim": 0.25, "qed": 0.25, "lipinski": 0.15}

# Two-stage ranking: after count-based scoring, only the top N candidates get
# the expensive 3D embedding + geometry_similarity.  24 comfortably covers any
# `top` the API can return (max 50 would allow up to 50; 24 keeps /api/generate
# snappy — benchmarked cost in runs notes: ~0.05 s per embedding).
GEOMETRY_STAGE_TOP_N = 24


@dataclass
class CandidateScore:
    smiles: str
    sig_sim: float
    qed: float
    lipinski: float
    score: float
    violations: int
    props: dict
    geometry_sim: float | None = None  # None when 3D geometry was not computed


def _lipinski_score(mol: Chem.Mol) -> tuple[float, int]:
    violations = sum(
        [
            Descriptors.MolWt(mol) > 500,
            Crippen.MolLogP(mol) > 5,
            Lipinski.NumHDonors(mol) > 5,
            Lipinski.NumHAcceptors(mol) > 10,
        ]
    )
    return max(0.0, 1.0 - 0.34 * violations), violations


def quick_props(mol: Chem.Mol) -> dict:
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 1),
        "logp": round(Crippen.MolLogP(mol), 2),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 1),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
    }


def _combine(parts: dict[str, float]) -> float:
    """Weighted geometric mean; missing/failed terms score ~0 (survivable)."""
    score = 1.0
    for name, value in parts.items():
        score *= max(value, 1e-6) ** WEIGHTS[name]
    return score


def score_candidate(
    smiles: str,
    ref_signature: dict[str, int],
    ref_pharmacophore: Pharmacophore | None = None,
) -> CandidateScore | None:
    """Score one candidate; returns None for molecules RDKit cannot parse.

    With ``ref_pharmacophore`` (3D-embedded) the candidate is also embedded and
    geometry_similarity is included in the score; without it the geometry term
    is left as None and the remaining weights act as a pure relative weight
    (they still sum to 1.0, so count-only scores stay comparable).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canon = Chem.MolToSmiles(mol)
    try:
        ph = extract_pharmacophore(canon, with_3d=False)
    except Exception:
        return None

    sig_sim = signature_similarity(ref_signature, ph.signature)
    qed = QED.qed(mol)
    lip, violations = _lipinski_score(mol)

    geometry_sim: float | None = None
    parts: dict[str, float] = {"sig_sim": sig_sim, "qed": qed, "lipinski": lip}
    if ref_pharmacophore is not None:
        try:
            cand_ph = extract_pharmacophore(canon, with_3d=True)
            geometry_sim = geometry_similarity(ref_pharmacophore, cand_ph)
            parts["geometry_sim"] = geometry_sim
        except Exception:
            pass  # embedding can fail for exotic molecules; keep the 2D score

    return CandidateScore(
        smiles=canon,
        sig_sim=round(sig_sim, 3),
        qed=round(qed, 3),
        lipinski=round(lip, 3),
        score=round(_combine(parts), 4),
        violations=violations,
        props=quick_props(mol),
        geometry_sim=geometry_sim,
    )


def rank_candidates(
    smiles_list: list[str],
    ref_signature: dict[str, int],
    ref_pharmacophore: Pharmacophore | None = None,
    geometry_top_n: int = GEOMETRY_STAGE_TOP_N,
) -> list[CandidateScore]:
    """Score every candidate, drop unparseable ones, best first.

    Two-stage filter: stage 1 scores everyone on cheap 2D evidence (counts +
    QED + Lipinski); if a 3D reference is supplied, stage 2 re-scores only the
    ``geometry_top_n`` best with the geometry term included and re-sorts.
    Note the two stages sit on slightly different scales (stage-1 scores miss
    the geometry exponent, which mildly inflates them), so callers should set
    ``geometry_top_n >= top`` whenever they slice ``[:top]`` afterwards — that
    way every molecule that can be shown to a user is scored on the full
    four-term scale and only never-shown tail entries carry the lenient score.
    """
    scored = []
    for smi in smiles_list:
        result = score_candidate(smi, ref_signature)
        if result is not None:
            scored.append(result)
    scored.sort(key=lambda c: c.score, reverse=True)

    if ref_pharmacophore is None or not scored:
        return scored

    head = scored[:geometry_top_n]
    tail = scored[geometry_top_n:]
    refined = []
    for candidate in head:
        result = score_candidate(candidate.smiles, ref_signature, ref_pharmacophore)
        if result is not None:
            refined.append(result)
    refined.sort(key=lambda c: c.score, reverse=True)
    return refined + tail


def to_dicts(candidates: list[CandidateScore]) -> list[dict]:
    return [asdict(c) for c in candidates]
