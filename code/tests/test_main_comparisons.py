import pytest

from benchmarks.make_main_comparisons import improvements, published_improvements, render, standings


def rows():
    return [
        {'generator': 'FLUX.2-klein-9B', 'generator_key': 'klein', 'agent': 'Direct',
         'agent_key': 'direct', 'ugb': 78.94, 'geneval2': {'gm': 34.57, 'am': 79.25}},
        {'generator': 'FLUX.2-klein-9B', 'generator_key': 'klein', 'agent': 'Kimi K3',
         'agent_key': 'k3', 'ugb': 93.43, 'geneval2': {'gm': 75.39, 'am': 91.62}},
        {'generator': 'Qwen-Image', 'generator_key': 'qwenimage', 'agent': 'Direct',
         'agent_key': 'direct', 'ugb': 79.41, 'geneval2': {'gm': 40.25, 'am': 82.89}},
        {'generator': 'Qwen-Image', 'generator_key': 'qwenimage', 'agent': 'Kimi K3',
         'agent_key': 'k3', 'ugb': 94.60, 'geneval2': {'gm': 79.76, 'am': 93.25}},
    ]


def test_gains_use_each_generators_own_baseline():
    result = improvements(rows())
    assert result[0]['delta'] is None
    assert result[1]['delta'] == pytest.approx(40.82)
    assert result[3]['delta'] == pytest.approx(39.51)
    with pytest.raises(ValueError, match='completed GenEval'):
        improvements([{**rows()[0], 'geneval2': None}])


def test_standings_do_not_invent_official_ranks_for_our_runs():
    result = standings(rows(), leaderboard())
    assert result[0]['generator'] == 'Public A' and result[0]['rank'] == 1
    assert all(r['rank'] is None for r in result if r['kind'] == 'agent')
    assert next(r for r in result if r['generator'] == 'Public B')['rank'] == 2


def leaderboard():
    return [{'model': 'Public B', 'Overall': 92.0},
            {'model': 'FLUX.2-klein-9b', 'Overall': 78.28},
            {'model': 'Public A', 'Overall': 95.0},
            {'model': 'Qwen-Image', 'Overall': 78.36}]


def test_published_baselines_outside_top_rows_keep_their_full_leaderboard_rank():
    result = standings(rows(), leaderboard(), limit=1)
    direct = {r['generator']: r for r in result if r['kind'] == 'direct'}
    assert (direct['FLUX.2-klein-9B']['score'], direct['FLUX.2-klein-9B']['rank']) == (78.28, 4)
    assert (direct['Qwen-Image']['score'], direct['Qwen-Image']['rank']) == (78.36, 3)
    assert len(direct) == 2
    assert not any(r['generator'] == 'Public B' for r in result)
    assert not any(r['score'] in (78.94, 79.41) for r in result)


def test_direct_baseline_in_the_top_rows_is_not_duplicated():
    result = standings(rows(), leaderboard(), limit=5)
    assert len([r for r in result if r['kind'] == 'direct']) == 2


def test_missing_published_baseline_is_not_replaced_with_our_measurement():
    with pytest.raises(ValueError, match='missing published leaderboard baselines'):
        standings(rows(), [{'model': 'Public A', 'Overall': 95.0}])


def published_rows():
    return [{'method': 'EPIC', 'generator': 'FLUX.2-klein-9B', 'source': 'epic',
             'gm': 71.46, 'am': 90.25, 'gm_delta': 37.30},
            {'method': 'Direct', 'generator': 'FLUX.2-klein-9B', 'source': 'epic',
             'gm': 34.16, 'am': 79.06, 'gm_delta': None},
            {'method': 'Direct', 'generator': 'Qwen-Image', 'source': 'geneval2',
             'gm': 33.8, 'am': 80.8}]


def test_gains_prefer_published_baselines_and_keep_measured_scores_intact():
    result = improvements(rows(), published_rows())
    assert result[1]['delta'] == pytest.approx(41.23)
    assert result[3]['delta'] == pytest.approx(45.96)
    assert result[0]['geneval2']['gm'] == 34.57
    assert result[0]['baseline_gm'] == 34.16
    assert result[0]['baseline_source'] == 'epic'
    assert result[2]['baseline_source'] == 'geneval2'


def test_missing_published_baseline_falls_back_per_generator():
    result = improvements(rows(), published_rows()[:2])
    assert result[1]['delta'] == pytest.approx(41.23)
    assert result[3]['delta'] == pytest.approx(39.51)
    assert result[3]['baseline_source'] == 'ours'


def test_conflicting_published_baselines_are_not_cherry_picked():
    published = published_rows() + [{**published_rows()[1], 'gm': 35.0}]
    with pytest.raises(ValueError, match='conflicting published'):
        improvements(rows(), published)


def test_main_table_keeps_am_in_appendix_and_uses_each_sources_deltas():
    text = render(rows(), leaderboard(),
                  published_rows())
    assert 'GM $' in text and 'AM scores' in text
    assert '91.62' not in text and '90.25' not in text
    assert r'\multicolumn{2}{l}{EPIC} & 71.46 & +37.30' in text
    assert r'\multicolumn{2}{l}{Direct} & 34.16 & ---' in text
    assert '+41.23' in text and '+45.96' in text
    assert '33.80' in text and '40.25' not in text
    assert text.count(r'\caption{') == 2


def test_published_deltas_require_the_matching_published_baseline():
    peers = published_rows()
    with pytest.raises(ValueError, match='published direct baseline'):
        published_improvements(peers[:1])
    peers[1]['gm'] = 34.57
    with pytest.raises(ValueError, match='published direct baseline'):
        published_improvements(peers)


def test_flux_dev_published_baseline_is_not_shown_as_a_klein_result():
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).resolve().parents[1] /
                       'benchmarks/geneval2/published_results.json').read_text())['rows']
    dev = next(row for row in data if row['generator'] == 'FLUX.2-dev')
    assert dev['gm'] == 42.09 and dev['am'] == 83.34 and dev['table'] == '6'
    assert all(row['generator'] == 'FLUX.2-klein-9B' for row in published_improvements(data))
