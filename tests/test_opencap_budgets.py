import numpy as np
import pytest

from analysis.compare_opencap_budgets import paired_scores, mean_scores


def row(joint, error, samples=75):
    return dict(joint=joint, mae_deg=abs(error), rmse_deg=abs(error), bias_deg=error, samples=samples)


def test_paired_comparison_matches_names_and_keeps_worsening_joint():
    names = ('knee_angle_r', 'knee_angle_l')
    baseline = [row(names[0], 20), row(names[1], 5)]
    candidate = [row(names[1], -7), row(names[0], 4)]
    pairs = paired_scores(baseline, candidate, names)
    assert [r['delta_mae_deg'] for r in pairs] == [-16, 2]
    assert pairs[1]['candidate_bias_deg'] == -7
    result = mean_scores(pairs)
    assert result['baseline_mae_deg'] == 12.5
    assert result['candidate_mae_deg'] == 5.5
    assert result['delta_mae_deg'] == -7
    assert result['candidate_rmse_deg'] == pytest.approx(np.sqrt((4**2+7**2)/2))


def test_pair_rejects_missing_duplicate_or_differently_sampled_joints():
    names = ('a', 'b')
    baseline = [row('a', 1), row('b', 2)]
    with pytest.raises(ValueError, match='declared joint'):
        paired_scores(baseline, [row('a', 1)], names)
    with pytest.raises(ValueError, match='Duplicate'):
        paired_scores(baseline, baseline+[row('a', 1)], names)
    with pytest.raises(ValueError, match='same number'):
        paired_scores(baseline, [row('a', 1, 74), row('b', 2)], names)
