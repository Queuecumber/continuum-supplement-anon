"""Build measured main results and audited appendix tables.

Run from the repository root: python -m benchmarks.make_paper_tables
Only complete GenEval arms receive numerical scores. Saved traces are counted
once per prompt/sample, independently of the number of manifest retry rows.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path

from benchmarks.make_geneval2_table import DATA, GENERATORS, summarize
from benchmarks.unigenbench import report as ugb

HERE = Path(__file__).resolve().parent
AGENTS = (("direct", "Direct"), ("k3", "Kimi K3"), ("qwen38", "Qwen3.8-27B"))
IMAGE_CALLS = {"generate_image", "edit_image", "inpaint_image"}


def ugb_cell(generator, agent):
    if generator == "klein":
        return {"direct": "direct", "k3": "agent", "qwen38": "qwen38-klein"}[agent]
    return f"{agent}-{generator}"


def run_path(cell, direct=False, geneval=False):
    if geneval:
        name = cell.removeprefix("direct-") if direct else cell
        return HERE / ("ugb-direct" if direct else "ugb-agent") / f"out-ge2-{name}"
    if direct:
        return HERE / "ugb-direct" / ("out" if cell == "direct" else f"out-{cell.removeprefix('direct-')}")
    return HERE / "ugb-agent" / ("out" if cell == "agent" else f"out-{cell}")


def load_rows(ugb_dir, ge_dir, benchmark, ugb_dev_direct="direct-dev"):
    rows = []
    for generator, name in GENERATORS.items():
        for agent, label in AGENTS:
            cell = ugb_dev_direct if generator == "dev" and agent == "direct" else ugb_cell(generator, agent)
            primary, _ = ugb.load(ugb_dir / f"{cell}_en.json")
            arm = f"{agent}-{generator}"
            scores = ge_dir / arm
            complete = all((scores / f"sample_{i:02d}.json").is_file() for i in range(4))
            ge = summarize(scores, benchmark) if complete else None
            rows.append({"generator": name, "generator_key": generator, "agent": label,
                         "agent_key": agent, "ugb_cell": cell,
                         "ugb": ugb.overall(primary), "geneval2": ge})
    return rows


def trace_stats(run):
    latest = {}
    records = 0
    files = sorted(run.glob("shard-*/manifest.jsonl")) + sorted(run.glob("manifest.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                records += 1
                latest[(row["item_id"], row["sample_index"])] = row
    if not latest:
        raise ValueError(f"no recorded episodes in {run}")
    turns = images = reasoning = 0
    statuses = collections.Counter()
    for row in latest.values():
        messages = json.loads(Path(row["transcript"]).read_text())
        assistants = [m for m in messages if m.get("role") == "assistant"]
        turns += len(assistants)
        reasoning += sum(bool(str(m.get("reasoning_content") or "").strip()) for m in assistants)
        names = {c["id"]: c["function"]["name"] for m in assistants for c in (m.get("tool_calls") or [])}
        images += sum(m.get("role") == "tool" and names.get(m.get("tool_call_id")) in IMAGE_CALLS
                      and str(m.get("content", "")).startswith("Produced image ") for m in messages)
        statuses[row.get("stop_reason", row["status"])] += 1
    n = len(latest)
    return {"traces": n, "additional_attempts": records - n, "turns": turns,
            "image_calls": images, "reasoning_turns": reasoning,
            "mean_turns": turns / n, "mean_image_calls": images / n,
            "reasoning_percent": 100 * reasoning / turns if turns else 0, "stop_reasons": dict(statuses)}


def table(columns, rows, caption, label, wide=False, fit=False):
    env = "table*" if wide else "table"
    lines = [f"\\begin{{{env}}}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}"]
    if fit:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines += [r"\begin{tabular}{" + columns + "}", r"\toprule"]
    lines.extend(rows)
    lines += [r"\bottomrule", r"\end{tabular}"]
    if fit:
        lines.append("}")
    lines += [r"\caption{" + caption + "}", r"\label{" + label + "}", f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def main_table(rows):
    lines = [r"Base generator & Agent & UGB++ Overall $\uparrow$ & GenEval 2 GM $\uparrow$ & GenEval 2 AM $\uparrow$ \\", r"\midrule"]
    for i, row in enumerate(rows):
        if i and i % 3 == 0:
            lines.append(r"\midrule")
        ge = row["geneval2"]
        gm = f"{ge['gm']:.2f}" if ge else r"\textit{pending}"
        am = f"{ge['am']:.2f}" if ge else r"\textit{pending}"
        lines.append(f"{row['generator']} & {row['agent']} & {row['ugb']:.2f} & {gm} & {am}" + r" \\")
    caption = (r"\textbf{Our measured benchmark results.} All scores use a 0--100 scale. "
               r"UGB++ Overall is the unweighted mean of ten primary-dimension accuracies. "
               r"GenEval 2 reports Soft-TIFA geometric (GM) and arithmetic (AM) means. "
               r"Each configuration uses four samples per prompt: 600 prompts for UGB++ and "
               r"800 for GenEval 2. Direct baselines are our own measurements. "
               + (r"Pending cells have not completed scoring. " if any(r["geneval2"] is None for r in rows) else "")
               + r"Additional comparisons, settings, "
               r"and trace statistics are in Appendix~A.")
    return table("llrrr", lines, caption, "tab:results", wide=True)


def breakdown(rows, benchmark):
    skills = ("object", "attribute", "count", "position", "verb")
    skill_lines = [r"Generator / Agent & Object & Attribute & Count & Position & Verb \\", r"\midrule"]
    atom_lines = [r"Generator / Agent & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\", r"\midrule"]
    for row in rows:
        if row["geneval2"] is None:
            continue
        arm = row["agent_key"] + "-" + row["generator_key"]
        by_skill, by_atom = collections.defaultdict(list), collections.defaultdict(list)
        for path in sorted((DATA / "scores" / arm).glob("sample_*.json")):
            for prompt, scores in zip(benchmark, json.loads(path.read_text())):
                for skill, score in zip(prompt["skills"], scores):
                    by_skill[skill].append(score)
                by_atom[prompt["atom_count"]].append(0 if 0 in scores else statistics.geometric_mean(scores))
        name = row["generator"] + " / " + row["agent"]
        skill_lines.append(name + " & " + " & ".join(f"{100*statistics.mean(by_skill[s]):.2f}" for s in skills) + r" \\")
        atom_lines.append(name + " & " + " & ".join(f"{100*statistics.mean(by_atom[a]):.2f}" for a in range(3, 11)) + r" \\")
    return table("lrrrrr", skill_lines, r"\textbf{GenEval 2 by skill.} Mean per-question soft score, "
                 r"averaged over four samples per prompt. Only fully scored configurations are included.",
                 "tab:geneval2-skills") + table("lrrrrrrrr", atom_lines,
                 r"\textbf{GenEval 2 by compositionality.} Soft-TIFA GM for prompts containing 3--10 "
                 r"visual atoms, averaged over four samples. Only fully scored configurations are included.",
                 "tab:geneval2-atomicity")


def trace_table(traces):
    lines = [r"Benchmark & Generator & Agent & Traces & Extra attempts & Mean turns & Image calls & Reasoning (\%) \\", r"\midrule"]
    for row in traces:
        lines.append(f"{row['benchmark']} & {row['generator']} & {row['agent']} & {row['traces']:,} & "
                     f"{row['additional_attempts']:,} & {row['mean_turns']:.2f} & "
                     f"{row['mean_image_calls']:.2f} & {row['reasoning_percent']:.1f}" + r" \\")
    caption = (r"\textbf{Recorded traces and interaction cost.} Each prompt/sample contributes its "
               r"latest saved transcript once. Extra attempts count additional manifest records, "
               r"including retries, whose transcripts may have been overwritten. Mean turns and "
               r"image calls describe the retained transcripts: recorded assistant messages and "
               r"successful generation, editing, and inpainting responses, respectively. Reasoning "
               r"is the percentage of assistant messages with a nonempty returned reasoning field. "
               r"These counts exclude request-level API retries and do not establish downstream "
               r"training effectiveness.")
    return table("lllrrrrr", lines, caption, "tab:trace-cost", fit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE.parent / "paper/tab")
    parser.add_argument("--ugb-dev-direct", default="direct-dev",
                        help="explicit result cell for the FLUX.2-dev UGB++ direct baseline")
    args = parser.parse_args()
    benchmark = [json.loads(line) for line in (DATA / "geneval2_data.jsonl").read_text().splitlines() if line.strip()]
    rows = load_rows(HERE / "unigenbench/results", DATA / "scores", benchmark, args.ugb_dev_direct)
    traces = []
    for row in rows:
        if row["agent_key"] == "direct":
            continue
        gen, agent = row["generator_key"], row["agent_key"]
        sources = [("UGB++", run_path(ugb_cell(gen, agent)), 2400)]
        if row["geneval2"] is not None:
            sources.append(("GenEval 2", run_path(agent + "-" + gen, geneval=True), 3200))
        for name, run, expected in sources:
            stats = trace_stats(run)
            if stats["traces"] != expected:
                raise ValueError(f"{run}: expected {expected} traces, found {stats['traces']}")
            traces.append({"benchmark": name, "agent": row["agent"], "generator": row["generator"], **stats})
            print(name, row["generator"], row["agent"], stats, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "measured.tex").write_text(main_table(rows))
    (args.out / "trace_cost.tex").write_text(trace_table(traces))
    (args.out / "geneval2_breakdown.tex").write_text(breakdown(rows, benchmark))
    (args.out / "measurements.json").write_text(json.dumps({"results": rows, "traces": traces}, indent=2) + "\n")


if __name__ == "__main__":
    main()
