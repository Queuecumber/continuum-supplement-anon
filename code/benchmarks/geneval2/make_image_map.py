#!/usr/bin/env python3
"""Map GenEval 2 prompts to generated images, one JSON per sample index.

    python make_image_map.py --run ../ugb-direct/out-ge2-klein --out maps/direct-klein

GenEval 2's evaluator takes {prompt: filepath} and scores one image per
prompt, so a 4-sample run is scored as four separate passes and averaged.
Keys are the prompt text itself, which is what the evaluator looks up.
"""
import argparse, glob, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=str(HERE.parent / "geneval2_prompts.jsonl"))
    ap.add_argument("--require-complete", action="store_true",
                    help="refuse to write maps until every prompt/sample is accepted")
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    if a.samples < 1:
        ap.error("--samples must be positive")

    prompt_of = {json.loads(l)["id"]: json.loads(l)["prompt"]
                 for l in open(a.manifest) if l.strip()}
    per_sample = {}
    for f in glob.glob(f"{a.run}/shard-*/manifest.jsonl") + glob.glob(f"{a.run}/manifest.jsonl"):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok" or not r.get("output_image"):
                continue
            if a.require_complete and r.get("stop_reason", "") not in ("", "completed", "direct"):
                continue
            if not os.path.exists(r["output_image"]):
                continue
            p = prompt_of.get(r["item_id"])
            if p:
                per_sample.setdefault(r["sample_index"], {})[p] = r["output_image"]

    if a.require_complete:
        expected = set(prompt_of.values())
        if not expected or len(expected) != len(prompt_of):
            raise SystemExit("manifest must contain nonempty, unique prompts")
        problems = []
        for s in range(a.samples):
            missing = expected - per_sample.get(s, {}).keys()
            if missing:
                problems.append(f"sample {s}: {len(missing)}/{len(expected)} prompts missing")
        extra = set(per_sample) - set(range(a.samples))
        if extra:
            problems.append(f"unexpected sample indices: {sorted(extra)}")
        if problems:
            raise SystemExit("run is incomplete; no maps written:\n  " + "\n  ".join(problems))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, mapping in sorted(per_sample.items()):
        dst = out / f"sample_{s:02d}.json"
        dst.write_text(json.dumps(mapping, ensure_ascii=False, indent=1))
        print(f"  sample {s}: {len(mapping)}/{len(prompt_of)} prompts -> {dst}")
    if not per_sample:
        raise SystemExit("  no images found -- is the run complete?")


if __name__ == "__main__":
    main()
