"""Build compact UGB++ standings and GenEval 2 GM comparisons for the main paper."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_LABELS = {'FLUX.2-klein-9B': 'Klein 9B', 'FLUX.2-dev': 'FLUX.2-dev',
                    'Qwen-Image': 'Qwen-Image',
                    'GPT-4o': 'GPT Image', 'GPT-4o-1.5': 'GPT Image 1.5',
                    'Imagen-4.0-Ultra-preview-06-06': 'Imagen 4 Ultra'}


def tex(text):
    escaped = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '&': r'\&',
               '_': r'\_', '%': r'\%', '#': r'\#', '$': r'\$', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}'}
    return ''.join(escaped.get(char, char) for char in str(text))


def standings(measured, leaderboard, limit=5):
    def key(name):
        return re.sub('[^a-z0-9]', '', name.lower())

    published = sorted(leaderboard, key=lambda row: -row['Overall'])
    agents = [row for row in measured if row['agent_key'] != 'direct']
    generators = {key(row['generator']): row['generator'] for row in agents}
    missing = set(generators) - {key(row['model']) for row in published}
    if missing:
        raise ValueError(f'missing published leaderboard baselines: {[generators[k] for k in sorted(missing)]}')
    rows = []
    for rank, row in enumerate(published, 1):
        model_key = key(row['model'])
        if rank > limit and model_key not in generators:
            continue
        rows.append({'generator': generators.get(model_key, row['model']),
                     'agent': '---', 'score': row['Overall'], 'rank': rank,
                     'kind': 'direct' if model_key in generators else 'published'})
    for row in agents:
        rows.append({'generator': row['generator'], 'agent': row['agent'], 'score': row['ugb'],
                     'rank': None, 'kind': 'agent'})
    return sorted(rows, key=lambda row: -row['score'])


def improvements(measured, published=()):
    """Prefer a published direct baseline for the same generator, when available."""
    if any(r['geneval2'] is None for r in measured):
        raise ValueError('main comparisons require completed GenEval 2 scores')
    direct = {r['generator']: {'gm': r['geneval2']['gm'], 'source': 'ours'}
              for r in measured if r['agent_key'] == 'direct'}
    reported = {}
    for row in published:
        if row['method'] != 'Direct':
            continue
        previous = reported.get(row['generator'])
        if previous is not None and previous['gm'] != row['gm']:
            raise ValueError('conflicting published direct baselines; select a source explicitly')
        reported[row['generator']] = row
    direct.update(reported)
    rows = []
    for row in measured:
        baseline = direct[row['generator']]
        rows.append({**row, 'delta': None if row['agent_key'] == 'direct'
                     else row['geneval2']['gm'] - baseline['gm'],
                     'baseline_gm': baseline['gm'], 'baseline_source': baseline['source']})
    return rows


def published_improvements(published):
    """Keep reported deltas paired with the direct baseline from their source."""
    peers = [r for r in published if r.get('source') == 'epic' and r['generator'] == 'FLUX.2-klein-9B']
    direct = {r['generator']: r['gm'] for r in peers if r['method'] == 'Direct'}
    for row in peers:
        if row['method'] == 'Direct':
            if row.get('gm_delta') is not None:
                raise ValueError('published direct baseline must not have a delta')
        elif (row['generator'] not in direct or row.get('gm_delta') is None
              or abs(row['gm_delta'] - (row['gm'] - direct[row['generator']])) > 0.005):
            raise ValueError('published delta must match its published direct baseline')
    return sorted(peers, key=lambda r: -r['gm'])


def render(measured, leaderboard, published):
    ugb = standings(measured, leaderboard)
    ge = improvements(measured, published)
    lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{minipage}[t]{0.49\textwidth}', r'\vspace{0pt}', r'\centering', r'\draftON',
             r'\begin{tabular}{llrr}', r'\toprule', r'Generator & Agent & Overall $\uparrow$ & Rank \\', r'\midrule']
    for row in ugb:
        gen = tex(GENERATOR_LABELS.get(row['generator'], row['generator']))
        agent, score = tex(row['agent']), f"{row['score']:.2f}"
        if row['kind'] == 'agent':
            gen, agent, score = (r'\textbf{' + x + '}' for x in (gen, agent, score))
        elif row['kind'] == 'direct':
            gen, agent = r'\textit{' + gen + '}', r'\textit{' + agent + '}'
        rank = str(row['rank']) if row['rank'] is not None else '---'
        lines.append(f'{gen} & {agent} & {score} & {rank}' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
        r'\caption{\draft{\textbf{UGB++ standings.} Overall scores on a 0--100 scale. '
        r'Bold: our agentic runs; italics: published direct baselines. Other rows are the top five '
        r'published entries~\cite{ugb_leaderboard_20260914}. Rank refers only to the published '
        r'leaderboard snapshot (September 14, 2026). Klein 9B denotes FLUX.2-klein-9B. '
        r'Execution settings and additional comparisons are in Appendix~A.}}',
        r'\label{tab:results}', r'\end{minipage}\hfill',
        r'\begin{minipage}[t]{0.49\textwidth}', r'\vspace{0pt}', r'\centering', r'\draftON',
        r'\begin{tabular}{llrr}', r'\toprule', r'Generator & Agent / method & GM $\uparrow$ & $\Delta$ \\', r'\midrule']
    for i, row in enumerate(ge):
        if i and row['generator_key'] != ge[i - 1]['generator_key']:
            lines.append(r'\midrule')
        gen = tex(GENERATOR_LABELS[row['generator']])
        agent = tex(row['agent'])
        score = f"{row['baseline_gm'] if row['agent_key'] == 'direct' else row['geneval2']['gm']:.2f}"
        delta = '---' if row['delta'] is None else f"{row['delta']:+.2f}"
        if row['agent_key'] != 'direct':
            agent, score, delta = (r'\textbf{' + x + '}' for x in (agent, score, delta))
        lines.append(f'{gen} & {agent} & {score} & {delta}' + r' \\')
    lines += [r'\midrule', r'\multicolumn{4}{l}{\textit{Published comparison: Klein 9B}$^{\dagger}$} \\']
    peers = published_improvements(published)
    for row in peers:
        delta = '---' if row.get('gm_delta') is None else f"{row['gm_delta']:+.2f}"
        lines.append(r'\multicolumn{2}{l}{' + tex(row['method']) + '} & '
                     + f"{row['gm']:.2f} & {delta}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
        r'\caption{\draft{\textbf{GenEval 2 improvement and comparison.} Soft-TIFA GM on a '
        r'0--100 scale. $\Delta$ uses the published direct baselines shown: FLUX.2-dev and '
        r'Klein 9B from EPIC, Tables~6 and~1~\cite{mun2026epic}, and Qwen-Image from '
        r'GenEval~2, Table~4~\cite{kamath2025geneval2}. Our runs use 800 prompts '
        r'and four samples each. $\dagger$: reported in EPIC, Table~1 '
        r'($T=0$)~\cite{mun2026epic}; agent and compute settings differ. AM scores and '
        r'broader published comparisons are in Appendix~A.}}',
        r'\label{tab:geneval2-main}', r'\end{minipage}', r'\end{table*}', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leaderboard', type=Path, required=True)
    parser.add_argument('--measurements', type=Path, default=ROOT / 'paper/tab/measurements.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'paper/tab/main_comparisons.tex')
    args = parser.parse_args()
    measured = json.loads(args.measurements.read_text())['results']
    board = json.loads(args.leaderboard.read_text())['leaderboard']
    published = json.loads((ROOT / 'benchmarks/geneval2/published_results.json').read_text())['rows']
    args.out.write_text(render(measured, board, published))
    args.out.with_suffix('.json').write_text(json.dumps({
        'ugb': standings(measured, board), 'geneval2': improvements(measured, published),
        'published_geneval2': published_improvements(published),
        'ugb_source': 'https://huggingface.co/spaces/CodeGoat24/UniGenBench_Leaderboard/blob/main/leaderboard_data.json',
        'snapshot_date': '2026-09-14',
    }, indent=2) + '\n')
    print(f'Wrote compact standings and GM comparisons to {args.out}')


if __name__ == '__main__':
    main()
