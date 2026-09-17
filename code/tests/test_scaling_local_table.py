import csv

import pytest

from benchmarks.make_scaling_local_table import local_rows, scored_cell


def hardware_report():
    return {'cases': [{'id': 'text'}], 'warmups': 1, 'repetitions': 3,
            'trials': [{'case': 'text', 'phase': phase, 'warmup': i == 0,
                        'ok': True, 'wall_s': seconds}
                       for phase in ('direct', 'agent')
                       for i, seconds in enumerate([1000, 10, 10, 40])]}


def test_serial_rate_uses_total_measured_time_and_excludes_warmup():
    row = local_rows(hardware_report())[0]
    assert row['agent_median_s'] == 10
    assert row['agent_serial_per_min'] == 3
    assert row['direct_serial_per_min'] == 3


@pytest.mark.parametrize('change', ['failed', 'missing'])
def test_local_table_does_not_hide_unsuccessful_or_missing_trials(change):
    report = hardware_report()
    if change == 'failed':
        report['trials'][-1]['ok'] = False
    else:
        report['trials'].pop()
    with pytest.raises(ValueError, match='complete successful'):
        local_rows(report)


def test_scored_cell_rejects_duplicate_sample_coverage(tmp_path):
    with (tmp_path / 'cell_en.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['img_path'])
        writer.writeheader()
        writer.writerows([{'img_path': '0_0.png'}, {'img_path': '0_0.png'}])
    with pytest.raises(ValueError, match='coverage'):
        scored_cell(tmp_path, 'cell', prompts=1, samples=2)
