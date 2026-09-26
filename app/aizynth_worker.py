"""Step 4 — Buildability worker: a long-lived AiZynthFinder subprocess.

Why a worker process? Loading AiZynthFinder's models takes ~1 s of setup and
the tree search takes 10-30 s per molecule. The web server must stay
responsive, so the heavy lifting happens in a separate process that speaks
JSON lines over stdin/stdout:

    -> {"smiles": "...", "id": 1}
    <- {"id": 1, "buildable": true, "n_routes": 3, "n_steps": 2, ...}

Also important on this 1.9 GB machine: the worker uses the light config
(ONNX models + the 11 MB molbloom "buyable?" filter), which fits in memory
happily where the full 1.3 GB ZINC stock would get the process OOM-killed.
"""

from __future__ import annotations

import json
import sys

CONFIG = "runs/aizynth_data/config.yml"
EXPANSION_TIME = 20  # seconds per molecule


def main() -> None:
    from aizynthfinder.aizynthfinder import AiZynthFinder

    finder = AiZynthFinder(configfile=CONFIG)
    finder.stock.select(["zinc"])
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")
    finder.expansion_time = EXPANSION_TIME

    def send(msg: dict) -> None:
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

    send({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        smiles = request.get("smiles", "")
        request_id = request.get("id")
        try:
            finder.target_smiles = smiles
            finder.tree_search()
            finder.build_routes()
            routes = finder.routes

            if not routes:
                send(
                    {
                        "id": request_id,
                        "smiles": smiles,
                        "buildable": False,
                        "n_routes": 0,
                        "n_steps": None,
                        "building_blocks": [],
                        "score": None,
                    }
                )
                continue

            tree = routes[0]["reaction_tree"].to_dict()

            def walk(node, stats):
                if node.get("is_reaction"):
                    stats["reactions"] += 1
                if node.get("type") == "mol" and node.get("in_stock"):
                    stats["blocks"].add(node["smiles"])
                for child in node.get("children", []):
                    walk(child, stats)

            stats = {"reactions": 0, "blocks": set()}
            walk(tree, stats)
            score = routes[0].get("score", {})
            send(
                {
                    "id": request_id,
                    "smiles": smiles,
                    "buildable": bool(routes[0]["route_metadata"].get("is_solved")),
                    "n_routes": len(routes),
                    "n_steps": stats["reactions"],
                    "building_blocks": sorted(stats["blocks"])[:6],
                    "score": round(float(next(iter(score.values()))), 3) if score else None,
                }
            )
        except Exception as err:  # noqa: BLE001 — worker must survive any molecule
            send({"id": request_id, "smiles": smiles, "error": str(err)[:200]})


if __name__ == "__main__":
    main()
