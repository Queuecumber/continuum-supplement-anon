#!/usr/bin/env python3
"""Report UniGenBench results the way the leaderboard does.

The official `calculate_scores` emits per-dimension breakdowns and no overall
score, while the leaderboard publishes one. Reproducing it from the published
rows shows it is the unweighted mean of the ten primary dimensions — that
formula recovers every one of the 63 published Overall values to within 0.005,
which is what makes it safe to quote ours beside theirs.

    ./report.py --results results --compare        # against the live leaderboard

Every dimension is asserted present. A missing one would quietly shrink the
mean and flatter whichever arm lost it, and the scorer's own keys carry
trailing spaces ("Attribute ") that make a silent miss easy.
"""

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LEADERBOARD_URL = (
    "https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard/raw/main/leaderboard_data.json"
)

# our primary dimension -> the leaderboard's column. Ten of them; the overall
# is their unweighted mean.
PRIMARY = {
    "Style": "Style",
    "World Knowledge": "World Knowledge",
    "Attribute": "Attribute-Overall",
    "Action": "Action-Overall",
    "Relationship": "Relationship-Overall",
    "Compound": "Compound-Overall",
    "Grammar": "Grammar-Overall",
    "Entity Layout": "Layout-Overall",
    "Logical Reasoning": "Logical Reasoning",
    "Text Generation": "Text",
}

SUB = {
    "Attribute - Quantity": "Quantity",
    "Attribute - Expression": "Expression",
    "Attribute - Material": "Material",
    "Attribute - Size": "Size",
    "Attribute - Shape": "Shape",
    "Attribute - Color": "Color",
    "Action - Hand (Character/Anthropomorphic)": "Hand",
    "Action - Full-body (Character/Anthropomorphic)": "Full body",
    "Action - Animal": "Animal",
    "Action - Non-contact Interaction": "Non Contact",
    "Action - Contact Interaction": "Contact",
    "Action - State": "State",
    "Relationship - Composition": "Composition",
    "Relationship - Similarity": "Similarity",
    "Relationship - Inclusion": "Inclusion",
    "Relationship - Comparison": "Comparison",
    "Compound - Imagination": "Imagination",
    "Compound - Feature Matching": "Feature matching",
    "Grammar - Pronoun Reference": "Pronoun Reference",
    "Grammar - Consistency": "Consistency",
    "Grammar - Negation": "Negation",
    "Entity Layout - Two-Dimensional Space": "2D",
    "Entity Layout - Three-Dimensional Space": "3D",
    "Logical Reasoning": "Logical Reasoning",
    "Style": "Style",
    "World Knowledge": "World Knowledge",
    "Text Generation": "Text",
}


def normalise(scores: Dict[str, dict]) -> Dict[str, dict]:
    """Strip the trailing spaces the scorer leaves on split dimension names."""
    return {key.strip(): value for key, value in scores.items()}


def percent(entry: dict) -> float:
    return float(entry["accuracy"]) * 100.0


def load(path: Path) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return normalise(data["primary_dimensions"]), normalise(data["sub_dimensions"])


def overall(primary: Dict[str, dict]) -> float:
    """The leaderboard's Overall: unweighted mean of the ten primary dimensions."""
    missing = [k for k in PRIMARY if k not in primary]
    if missing:
        raise KeyError(f"missing primary dimensions: {missing}")
    return statistics.mean(percent(primary[k]) for k in PRIMARY)


def fetch_leaderboard() -> Optional[List[dict]]:
    try:
        with urllib.request.urlopen(LEADERBOARD_URL, timeout=60) as response:
            return json.loads(response.read())["leaderboard"]
    except Exception as exc:  # noqa: BLE001 - comparison is optional
        print(f"  (leaderboard unavailable: {exc})", file=sys.stderr)
        return None


def verify_formula(board: List[dict]) -> float:
    """Largest error reproducing published Overall as the mean of ten primaries."""
    worst = 0.0
    for row in board:
        values = [row[c] for c in PRIMARY.values() if isinstance(row.get(c), (int, float))]
        if len(values) == len(PRIMARY) and isinstance(row.get("Overall"), (int, float)):
            worst = max(worst, abs(statistics.mean(values) - row["Overall"]))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results", help="directory holding <arm>_en.json")
    ap.add_argument("--arms", default="direct,agent")
    ap.add_argument("--compare", action="store_true", help="rank against the published leaderboard")
    ap.add_argument("--full", action="store_true", help="every leaderboard column, not just the ten primaries")
    args = ap.parse_args()

    root = Path(args.results)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    loaded = {}
    for arm in arms:
        path = root / f"{arm}_en.json"
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            return 2
        loaded[arm] = load(path)

    print(f"{'primary dimension':<22}" + "".join(f"{a:>10}" for a in arms) + f"{'delta':>9}{'n':>7}")
    print("-" * (22 + 10 * len(arms) + 16))
    for key in PRIMARY:
        cells = [percent(loaded[a][0][key]) for a in arms]
        n = loaded[arms[0]][0][key]["total"]
        delta = cells[-1] - cells[0] if len(cells) > 1 else 0.0
        print(f"{key:<22}" + "".join(f"{c:>9.2f}" for c in cells) + f"{delta:>+9.2f}{n:>7}")

    print("-" * (22 + 10 * len(arms) + 16))
    overalls = {a: overall(loaded[a][0]) for a in arms}
    print(f"{'OVERALL':<22}" + "".join(f"{overalls[a]:>9.2f}" for a in arms), end="")
    if len(arms) > 1:
        print(f"{overalls[arms[-1]] - overalls[arms[0]]:>+9.2f}")
    else:
        print()

    if args.full:
        board = fetch_leaderboard() if args.compare else None
        ranked_by = {}
        if board:
            for col in list(PRIMARY.values()) + list(SUB.values()):
                ranked_by[col] = sorted(
                    (r[col] for r in board if isinstance(r.get(col), (int, float))), reverse=True
                )
        print(f"\n{'leaderboard column':<22}" + "".join(f"{a:>10}" for a in arms) + f"{'delta':>9}{'rank':>7}")
        print("-" * (22 + 10 * len(arms) + 16))
        seen = set()
        for source, mapping in (("primary", PRIMARY), ("sub", SUB)):
            for ours_key, column in mapping.items():
                if column in seen:
                    continue
                seen.add(column)
                index = 0 if source == "primary" else 1
                cells = []
                for arm in arms:
                    entry = loaded[arm][index].get(ours_key)
                    if entry is None:
                        break
                    cells.append(percent(entry))
                if len(cells) != len(arms):
                    print(f"{column:<22}{'MISSING':>10}")
                    continue
                delta = cells[-1] - cells[0] if len(cells) > 1 else 0.0
                rank = ""
                if ranked_by.get(column):
                    rank = f"{1 + sum(1 for s in ranked_by[column] if s > cells[-1]):>5}/{len(ranked_by[column]) + 1}"
                print(f"{column:<22}" + "".join(f"{c:>9.2f}" for c in cells) + f"{delta:>+9.2f}{rank:>9}")

    if args.compare:
        board = fetch_leaderboard()
        if board:
            worst = verify_formula(board)
            print(f"\noverall formula reproduces {len(board)} published rows to within {worst:.3f} points")
            ranked = sorted(
                ((r["model"], r["Overall"]) for r in board if isinstance(r.get("Overall"), (int, float))),
                key=lambda x: -x[1],
            )
            for arm in arms:
                rank = 1 + sum(1 for _, s in ranked if s > overalls[arm])
                print(f"  {arm:<8}{overalls[arm]:>7.2f}   would rank {rank} of {len(ranked) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
