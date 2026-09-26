"""Step 3 — Scoring: rank candidate molecules against a reference molecule.

The score is a weighted mix of three explainable parts:

1. **Pattern match** (`sig_sim`) — how well the candidate's pharmacophore
   signature (bumps & notches counts, Step 2) overlaps the reference's.
2. **Drug-likeness** (`qed`) — RDKit's QED estimate (0-1), a rough
   "does this look like a medicine" score learned from real drugs.
3. **Lipinski** (`lipinski`) — 1.0 with zero rule-of-5 violations, minus
   0.34 per violation (so one slip is forgivable, many are not).

All three live in [0, 1] and combine by geometric mean — the same friendly
aggregation REINVENT itself uses. Geometric mean punishes being bad at one
thing harder than arithmetic averaging, which is what we want: a molecule
that matches the pattern perfectly but looks poisonous should score low.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

from pharmacophore import extract_pharmacophore, signature_similarity

WEIGHTS = {"sig_sim": 0.6, "qed": 0.25, "lipinski": 0.15}


@dataclass
class CandidateScore:
    smiles: str
    sig_sim: float
    qed: float
    lipinski: float
    score: float
    violations: int
    props: dict


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


def score_candidate(smiles: str, ref_signature: dict[str, int]) -> CandidateScore | None:
    """Score one candidate; returns None for molecules RDKit cannot parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        ph = extract_pharmacophore(Chem.MolToSmiles(mol), with_3d=False)
    except Exception:
        return None

    sig_sim = signature_similarity(ref_signature, ph.signature)
    qed = QED.qed(mol)
    lip, violations = _lipinski_score(mol)

    parts = {"sig_sim": sig_sim, "qed": qed, "lipinski": lip}
    score = 1.0
    for name, value in parts.items():
        score *= max(value, 1e-6) ** WEIGHTS[name]  # geometric mean, weighted

    return CandidateScore(
        smiles=Chem.MolToSmiles(mol),
        sig_sim=round(sig_sim, 3),
        qed=round(qed, 3),
        lipinski=round(lip, 3),
        score=round(score, 4),
        violations=violations,
        props=quick_props(mol),
    )


def rank_candidates(
    smiles_list: list[str], ref_signature: dict[str, int]
) -> list[CandidateScore]:
    """Score every candidate, drop unparseable ones, best first."""
    scored = []
    for smi in smiles_list:
        result = score_candidate(smi, ref_signature)
        if result is not None:
            scored.append(result)
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def to_dicts(candidates: list[CandidateScore]) -> list[dict]:
    return [asdict(c) for c in candidates]
