"""Build main-paper size-ladder and serial local-execution tables from saved results."""
import argparse
import collections
import csv
import hashlib
import json
import statistics
from pathlib import Path

from benchmarks.unigenbench import report as ugb

ROOT = Path(__file__).resolve().parents[1]
LADDER = [('q35-9b', '9B', 9), ('q35-35b-a3b', '35B-A3B', 3),
          ('q35-122b-a10b', '122B-A10B', 10), ('q35-397b-a17b', '397B-A17B', 17)]


def scored_cell(directory, cell, prompts=600, samples=4):
    path = directory / f'{cell}_en.csv'
    rows = list(csv.DictReader(path.open()))
    expected = {f'{i}_{s}.png' for i in range(prompts) for s in range(samples)}
    if len(rows) != len(expected) or {Path(r['img_path']).name for r in rows} != expected:
        raise ValueError(f'{cell}: incomplete or duplicated prompt/sample coverage')
    totals = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        verdict = json.loads(row['result_json'])
        if len(verdict['testpoint']) != len(verdict['score']):
            raise ValueError(f'{cell}: inconsistent judge verdict')
        for testpoint, score in zip(verdict['testpoint'], verdict['score']):
            if score not in (0, 1):
                raise ValueError(f'{cell}: nonbinary judge score')
            key = testpoint.split(' - ')[0].strip()
            totals[key][0] += score
            totals[key][1] += 1
    primary, _ = ugb.load(path.with_suffix('.json'))
    for key, value in primary.items():
        if totals[key] != [value['correct'], value['total']]:
            raise ValueError(f'{cell}: aggregate disagrees with raw verdicts')
    return {'cell': cell, 'score': ugb.overall(primary), 'samples': len(rows),
            'no_output': sum(r['output'] == 'no image produced' for r in rows),
            'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def local_rows(report):
    rows = []
    for case in report['cases']:
        row = {'case': case['id']}
        for phase in ('direct', 'agent'):
            trials = [t for t in report['trials'] if t['case'] == case['id'] and t['phase'] == phase]
            warmups = [t for t in trials if t['warmup']]
            measured = [t for t in trials if not t['warmup']]
            if (len(warmups) != report['warmups'] or not warmups
                    or len(measured) != report['repetitions'] or not measured
                    or not all(t['ok'] and t['wall_s'] > 0 for t in trials)):
                raise ValueError('local table requires complete successful warmup and measured trials')
            times = [t['wall_s'] for t in measured]
            row[phase + '_median_s'] = statistics.median(times)
            row[phase + '_serial_per_min'] = 60 * len(times) / sum(times)
        rows.append(row)
    return rows


def render(ladder, baseline, local, memory_gib):
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{minipage}[t]{0.49\textwidth}', r'\vspace{0pt}', r'\centering', r'\draftON',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Qwen3.5 agent & Active (B) & UGB++ & $\Delta$ \\', r'\midrule',
             f"Direct Klein & --- & {baseline['score']:.2f} & ---" + r' \\']
    for row in ladder:
        lines.append(f"{row['label']} & {row['active_b']} & {row['score']:.2f} & +{row['score']-baseline['score']:.2f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{\draft{\textbf{Qwen3.5 size ladder.} Hosted Qwen3.5 agents~\cite{qwen35} '
              r'with the same FLUX.2-klein-9B generator. Overall scores use 600 prompts '
              r'and four samples each; $\Delta$ is the gain over direct Klein. Active parameters '
              r'are per token. Appendix~A gives endpoint settings and the 122B repeat.}}',
              r'\label{tab:qwen35-ladder}', r'\end{minipage}\hfill',
              r'\begin{minipage}[t]{0.49\textwidth}', r'\vspace{0pt}', r'\centering', r'\draftON',
              r'\begin{tabular}{lrrr}', r'\toprule',
              r'Task & Fixed (s) & Agent (s) & Agent tasks/min \\', r'\midrule']
    labels = {'text': 'Rendered text', 'count': 'Count / color', 'layout': 'Spatial layout'}
    for row in local:
        lines.append(f"{labels[row['case']]} & {row['direct_median_s']:.2f} & {row['agent_median_s']:.2f} & {row['agent_serial_per_min']:.2f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{\draft{\textbf{Local execution on one H100 80GB.} Qwen3.8-27B-FP8 '
              r'and BF16 Klein at $1024^2$. Latencies are medians of five trials after one warmup. '
              r'The fixed path generates and edits once. Serial agent rate is $60/\text{mean seconds}$ '
              r'at concurrency one. All workflows completed; peak GPU memory was '
              + f'{memory_gib:.1f}' + r' GiB. Protocol and quality limitations are in Appendix~A.}}',
              r'\label{tab:local-main}', r'\end{minipage}', r'\end{table*}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'paper/tab/scaling_local.tex')
    args = parser.parse_args()
    results = ROOT / 'benchmarks/unigenbench/results'
    baseline = scored_cell(results, 'direct')
    ladder = [{**scored_cell(results, cell), 'label': label, 'active_b': active}
              for cell, label, active in LADDER]
    report = json.loads(args.hardware.read_text())
    if (report['agent_model'] != 'Qwen/Qwen3.8-27B-FP8'
            or report['profile'] != 'qwen38-fp8-klein-bf16'
            or 'H100' not in report['host']['gpu'] or report['size'] != [1024, 1024]
            or report['warmups'] != 1 or report['repetitions'] != 5):
        raise ValueError('hardware report does not match the table protocol')
    local = local_rows(report)
    args.out.write_text(render(ladder, baseline, local, report['memory']['gpu_peak_gib']))
    args.out.with_suffix('.json').write_text(json.dumps({
        'baseline': baseline, 'ladder': ladder,
        'repeat_122b': scored_cell(results, 'q35-122b-a10b-rep2'), 'local': local,
        'hardware_report_sha256': hashlib.sha256(args.hardware.read_bytes()).hexdigest(),
        'serial_rate_definition': '60 * measured successful trials / sum of measured wall seconds; no warmups',
        'excluded_0b8': 'Incomplete pilot: 25 recorded episodes; no aggregate score.',
        'endpoint_image_history': {'q35-397b-a17b': 32, 'other_completed_rungs': 'uncapped'},
    }, indent=2) + '\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
