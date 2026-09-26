"""REINVENT4 scoring component: pharmacophore-signature similarity reward.

Registered by REINVENT4's plugin scanner (``reinvent/scoring/importer.py``),
which walks every ``reinvent_plugins/components/`` namespace-package portion on
``PYTHONPATH`` for modules named ``comp_*.py`` — that is why this file lives in
``app/reinvent_plugins/components/`` and why ``run_staged_learning`` puts
``app/`` on the subprocess ``PYTHONPATH``.

Reward for one molecule (2D only — fast enough for the RL loop):

    reward = min_score + (max_score - min_score) * similarity ** power

where ``similarity`` is ``app/pharmacophore.signature_similarity`` between the
fixed reference signature and the candidate's signature. ``power > 1`` sharpens
the gradient toward close matches (default 3.0). The reference is passed in at
config time — never hardcoded — encoded as ``["family=count", ...]`` by
``app/reward_plugin.build_stage_scoring_toml``.
"""

from __future__ import annotations

__all__ = ["PharmacophoreSimilarity", "Parameters"]

import sys
from pathlib import Path
from typing import List

import numpy as np
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag

# Make our own app package importable (pharmacophore.py lives two levels up).
_APP_DIR = Path(__file__).resolve().parents[2]
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from pharmacophore import extract_pharmacophore, signature_similarity  # noqa: E402


@add_tag("__parameters")
@dataclass
class Parameters:
    """Params as collected by ``reinvent.scoring.config.collect_params``:

    TOML scalars arrive wrapped once (``List[float]``), flat TOML lists arrive
    as ``List[List[str]]`` — one inner entry per endpoint (we define exactly
    one endpoint).
    """

    reference_signature: List[List[str]]
    min_score: List[float]
    max_score: List[float]
    power: List[float]


@add_tag("__component")
class PharmacophoreSimilarity:
    """Live pharmacophore-match reward for REINVENT4's RL loop."""

    def __init__(self, params: Parameters):
        self.reference_signature: dict[str, int] = {}
        for item in params.reference_signature[0]:
            if not item:
                continue
            family, _, count = item.partition("=")
            self.reference_signature[family.strip()] = int(count)
        self.min_score = float(params.min_score[0])
        self.max_score = float(params.max_score[0])
        self.power = float(params.power[0])
        self.number_of_endpoints = 1

    def __call__(self, smilies: List[str]) -> ComponentResults:
        # One score per input SMILES, same order; failures MUST be NaN, never
        # 0.0 (see reinvent_plugins/components/component_results.py).
        scores = np.full(len(smilies), np.nan, dtype=float)
        for i, smiles in enumerate(smilies):
            try:
                ph = extract_pharmacophore(smiles, with_3d=False)
            except Exception:
                continue  # unparseable SMILES -> NaN
            sim = signature_similarity(self.reference_signature, ph.signature)
            scores[i] = self.min_score + (self.max_score - self.min_score) * (
                sim ** self.power
            )
        return ComponentResults([scores])
