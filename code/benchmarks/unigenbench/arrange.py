#!/usr/bin/env python3
"""Lay a harness run out the way the UniGenBench evaluator expects to find it.

The evaluator builds each image path as `<data_path>/<index>_<j>.png`, where
`index` is the row in `data/test_prompts_en.csv` and `j` counts the samples per
prompt. Our runs name images by item id — `ugb-en-0000` — and a sharded run
scatters them across one directory per GPU.

Symlinks rather than copies: 600 images per arm, and the originals stay the
single source of truth if a run is later resumed or re-scored.

    ./arrange.py --run ../ugb-agent/out --out eval_data/en/agent

Refuses to guess. An id that does not parse, or two images claiming the same
slot, is reported rather than silently dropped — the evaluator treats a missing
image as a failed prompt, so a quiet mistake becomes a worse score with no
indication why.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ugb-en-0000
ITEM_ID = re.compile(r"^ugb-(en|zh)-(\d+)$")


def parse_item_id(stem: str) -> Optional[int]:
    """`ugb-en-0158` -> 158, the row index the evaluator matches on."""
    match = ITEM_ID.match(stem)
    # Leading zeros are ours; the evaluator formats the index as a bare integer.
    return int(match.group(2)) if match else None


def item_id_of(path: Path, sample: int) -> Optional[int]:
    """The manifest index for an image, whichever layout it came from."""
    # outputs/<id>.png carries the id in its own stem; outputs/<id>/<k>.png
    # carries it on the parent directory.
    return parse_item_id(path.stem if sample == 0 and not path.parent.name.startswith("ugb-") else path.parent.name)


def find_outputs(run: Path) -> List[Tuple[Path, int]]:
    """Every output image with the sample it belongs to, sharded or not.

    The harness writes outputs/<id>.png for a single sample and
    outputs/<id>/<k>.png once --samples exceeds one, so both layouts have to be
    read: a run started at one sample and extended to four contains neither
    shape exclusively.
    """
    found: List[Tuple[Path, int]] = []
    roots = sorted(run.glob("shard-*/outputs")) or [run / "outputs"]
    for root in roots:
        # Nested first: when a run is extended from one sample to four, an item
        # that had to be redone has both outputs/<id>.png from the original run
        # and outputs/<id>/0.png from the new one. The nested copy is the newer
        # generation and the one its siblings came from, so it wins.
        for item_dir in sorted(p for p in root.glob("*") if p.is_dir()):
            for path in sorted(item_dir.glob("*.png")):
                try:
                    found.append((path, int(path.stem)))
                except ValueError:
                    continue
        for path in sorted(root.glob("*.png")):
            found.append((path, 0))
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="harness output dir (contains outputs/ or shard-*/outputs/)")
    ap.add_argument("--out", required=True, help="data_path to build, e.g. eval_data/en/agent")
    ap.add_argument("--sample", type=int, default=0, help="the j in <index>_<j>.png (default 0)")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    args = ap.parse_args()

    run = Path(args.run).resolve()
    out = Path(args.out).resolve()
    images = find_outputs(run)
    if not images:
        print(f"no images under {run}", file=sys.stderr)
        return 2

    placed: Dict[Tuple[int, int], Path] = {}
    unparsed: List[str] = []
    collisions: List[str] = []

    for image, sample in images:
        index = item_id_of(image, sample)
        if index is None:
            unparsed.append(str(image))
            continue
        key = (index, sample)
        if key in placed:
            collisions.append(f"{image} and {placed[key]} both map to {key}")
            continue
        placed[key] = image

    out.mkdir(parents=True, exist_ok=True)
    for (index, sample), source in sorted(placed.items()):
        target = out / f"{index}_{sample}.png"
        if target.is_symlink() or target.exists():
            target.unlink()
        if args.copy:
            target.write_bytes(source.read_bytes())
        else:
            target.symlink_to(source)

    per_sample: Dict[int, int] = {}
    for _, sample in placed:
        per_sample[sample] = per_sample.get(sample, 0) + 1
    for sample in sorted(per_sample):
        print(f"  sample {sample}: {per_sample[sample]} images")
    print(f"placed {len(placed)} of {len(images)} images into {out}")
    # Collisions are expected and resolved: an item redone after the run was
    # extended to four samples has its sample 0 in both layouts, and the nested
    # copy wins. Reported so the count is visible, but not a failure — treating
    # them as one aborted the scoring script twice under `set -e`.
    if collisions:
        print(f"resolved {len(collisions)} duplicate(s) in favour of the nested layout")
        for problem in collisions[:3]:
            print(f"  {problem}")

    # Unparsed ids are different: those are images that were not placed at all,
    # and the evaluator raises on a missing slot rather than skipping it.
    if unparsed:
        print(f"unparsed ids: {len(unparsed)}", file=sys.stderr)
        for problem in unparsed[:5]:
            print(f"  {problem}", file=sys.stderr)
    return 1 if unparsed else 0


if __name__ == "__main__":
    raise SystemExit(main())
