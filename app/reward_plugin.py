"""The pharmacophore-similarity reward, wired into REINVENT4 as a plugin.

REINVENT4 discovers user-supplied scoring components by scanning every
``reinvent_plugins/components/`` directory found on ``PYTHONPATH`` for modules
named ``comp_*.py`` (see ``reinvent/scoring/importer.py`` — classes tagged with
``@add_tag("__component")`` plus one ``@add_tag("__parameters")`` dataclass).
The component class therefore lives in
``app/reinvent_plugins/components/comp_pharmacophore_similarity.py``; this
module is its home: it owns the shared config glue (this process writes the
stage-scoring TOML, the REINVENT subprocess parses it back) and re-exports the
class for anyone who wants the object itself.

Everything here is deliberately dependency-light (stdlib only) so it can be
imported from both sides of the subprocess boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent  # app/
PLUGIN_PARENT = APP_DIR  # what run_staged_learning puts on the subprocess PYTHONPATH
COMPONENT_MODULE = "reinvent_plugins.components.comp_pharmacophore_similarity"

REWARD_NAME = "Pharmacophore similarity"  # becomes the CSV column name


def encode_signature(signature: dict[str, int]) -> list[str]:
    """Serialize a signature dict into TOML/JSON-compatible ``family=count`` items."""
    return [f"{family}={count}" for family, count in sorted(signature.items())]


def build_stage_scoring_toml(
    reference_signature: dict[str, int],
    min_score: float = 0.0,
    max_score: float = 1.0,
    power: float = 3.0,
) -> str:
    """Build the per-stage scoring file for REINVENT4's ``staged_learning`` mode.

    Note (verified against v4.8.24): this file's content is the *scoring
    section itself* — top-level ``type`` + ``[[component]]`` — because
    ``reinvent/scoring/scorer.py::setup_scoring`` merges it into the stage's
    scoring dict. Wrapping it in ``[scoring]`` fails validation.
    """
    if not reference_signature:
        raise ValueError("reference_signature is empty — no reward signal to steer with")
    # TOML string arrays are JSON-compatible; json.dumps gives safe quoting.
    import json

    items = json.dumps(encode_signature(reference_signature))
    return (
        'type = "geometric_mean"\n'
        "\n"
        "[[component]]\n"
        "[component.pharmacophore_similarity]\n"
        "\n"
        "[[component.pharmacophore_similarity.endpoint]]\n"
        f'name = "{REWARD_NAME}"\n'
        "weight = 1.0\n"
        f"params.reference_signature = {items}\n"
        f"params.min_score = {float(min_score)}\n"
        f"params.max_score = {float(max_score)}\n"
        f"params.power = {float(power)}\n"
    )


def __getattr__(name: str):
    """Lazily re-export the component class (keeps this module import-light)."""
    if name in ("PharmacophoreSimilarity", "Parameters"):
        import importlib

        return getattr(importlib.import_module(COMPONENT_MODULE), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
