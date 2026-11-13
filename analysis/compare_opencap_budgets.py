"""Run predeclared longer fits and verify paired joint errors against saved baselines."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from analysis.evaluate_opencap import KNEES, read_mot, score_angles
from analysis.score_opencap_joints import JOINTS, coordinate_signs, score_saved_fit
from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics


def paired_scores(baseline, candidate, joints=JOINTS):
    before = {r['joint']: r for r in baseline}
    after = {r['joint']: r for r in candidate}
    if set(before) != set(joints) or set(after) != set(joints):
        raise ValueError('Both fits must contain exactly the declared joint coordinates.')
    if len(baseline) != len(joints) or len(candidate) != len(joints):
        raise ValueError('Duplicate joint coordinates in paired comparison.')
    rows = []
    for joint in joints:
        b, a = before[joint], after[joint]
        if b['samples'] != a['samples']:
            raise ValueError('Paired joint scores must cover the same number of samples.')
        rows.append(dict(joint=joint, samples=b['samples'],
                         baseline_mae_deg=b['mae_deg'], candidate_mae_deg=a['mae_deg'],
                         delta_mae_deg=a['mae_deg']-b['mae_deg'],
                         baseline_rmse_deg=b['rmse_deg'], candidate_rmse_deg=a['rmse_deg'],
                         baseline_bias_deg=b['bias_deg'], candidate_bias_deg=a['bias_deg']))
    return rows


def verify_pair(clip, directory, iterations, model):
    baseline = Path(clip['baseline'])
    metadata = json.loads((baseline/'metrics.json').read_text())
    result = json.loads((directory/'metrics.json').read_text())
    assert metadata['iterations'] == result['baseline_iterations'] == 2000, 'Wrong baseline budget'
    assert result['status'] == 'complete' and result['iterations'] == iterations, 'Incomplete longer fit'
    assert Path(result['clip']).resolve() == baseline.resolve(), 'Wrong baseline clip'
    assert result['detection_sha256'] == hashlib.sha256((baseline/'torch_keypoints.npz').read_bytes()).hexdigest(), 'Changed detections'
    checkpoint = next(c for c in result['checkpoints'] if c['step'] == iterations)
    signs = coordinate_signs(metadata['reference_model'], model)
    rt, reference = read_mot(metadata['mocap_ik'], JOINTS)
    with np.load(directory/f'checkpoint_{iterations}.npz') as candidate, np.load(baseline/'trajectories.npz') as original:
        times, qpos = candidate['reference_time'], candidate['qpos']
        np.testing.assert_array_equal(times, original['reference_time'])
        assert qpos.shape == original['torch_qpos'].shape == (75, model.nq), 'Wrong trajectory shape'
        assert np.isfinite(qpos).all(), 'Nonfinite fitted coordinates'
        assert times[0] >= rt[0] and times[-1] <= rt[-1], 'Reference extrapolation'
        prediction = np.rad2deg(qpos[:, [int(model.joint(n).qposadr[0]) for n in JOINTS]]) * signs
        recomputed, _, _ = score_angles(times, prediction, rt, reference, JOINTS)
    for actual, stored in zip(recomputed, checkpoint['metrics']):
        assert actual['joint'] == stored['joint'], 'Wrong coordinate order'
        for metric in ('mae_deg', 'rmse_deg', 'bias_deg'):
            np.testing.assert_allclose(actual[metric], stored[metric], atol=1e-8)
    before, _ = score_saved_fit(baseline, model)
    return paired_scores([r for r in before if r['pipeline'] == 'torch'], recomputed)


def mean_scores(rows):
    return dict(baseline_mae_deg=float(np.mean([r['baseline_mae_deg'] for r in rows])),
                candidate_mae_deg=float(np.mean([r['candidate_mae_deg'] for r in rows])),
                delta_mae_deg=float(np.mean([r['delta_mae_deg'] for r in rows])),
                baseline_rmse_deg=float(np.sqrt(np.mean([r['baseline_rmse_deg']**2 for r in rows]))),
                candidate_rmse_deg=float(np.sqrt(np.mean([r['candidate_rmse_deg']**2 for r in rows]))))


def summarize(output, manifest):
    model = ForwardKinematics().model
    records, participants = [], []
    for clip in manifest['clips']:
        directory = output/clip['id']
        if not (directory/'metrics.json').exists():
            continue
        result = json.loads((directory/'metrics.json').read_text())
        if result['status'] != 'complete':
            continue
        rows = verify_pair(clip, directory, manifest['iterations'], model)
        records.extend(dict(clip_id=clip['id'], subject=clip['subject'], activity=clip['activity'], **r) for r in rows)
        participants.append(dict(subject=clip['subject'], activity=clip['activity'], clip_id=clip['id'],
                                 all_joints=mean_scores(rows),
                                 knees=mean_scores([r for r in rows if r['joint'] in KNEES]),
                                 joints_improved=sum(r['delta_mae_deg'] < 0 for r in rows)))
    knees = [r for r in records if r['joint'] in KNEES]
    report = dict(status='complete' if len(participants) == len(manifest['clips']) else 'partial',
                  generated_utc=datetime.now(timezone.utc).isoformat(),
                  selected=len(manifest['clips']), completed=len(participants),
                  baseline_iterations=manifest['baseline_iterations'], candidate_iterations=manifest['iterations'],
                  participants=participants, limitations=manifest['limitations'],
                  verification='Matching saved clocks, detections, budgets and 75-frame trajectories; scores independently recomputed from qpos and original marker-based IK.')
    if records:
        report.update(all_joints=mean_scores(records), knees=mean_scores(knees),
                      participants_with_better_knees=sum(p['knees']['delta_mae_deg'] < 0 for p in participants),
                      participants_with_better_all_joint_average=sum(p['all_joints']['delta_mae_deg'] < 0 for p in participants),
                      individual_knees_improved=sum(r['delta_mae_deg'] < 0 for r in knees),
                      joints={j: mean_scores([r for r in records if r['joint'] == j]) for j in JOINTS})
        partial = output/'joint_comparison.partial.csv'
        with partial.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        partial.replace(output/'joint_comparison.csv')
    partial = output/'summary.partial.json'
    partial.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    partial.replace(output/'summary.json')
    return report


def run_batch(output):
    manifest = json.loads((output/'manifest.json').read_text())
    if len({c['subject'] for c in manifest['clips']}) != len(manifest['clips']):
        raise ValueError('Use one clip per participant for this paired comparison.')
    for clip in manifest['clips']:
        directory = output/clip['id']
        metrics = directory/'metrics.json'
        if not metrics.exists() or json.loads(metrics.read_text())['status'] != 'complete':
            print('Fitting', clip['id'], flush=True)
            subprocess.run([sys.executable, '-u', '-m', 'analysis.evaluate_opencap_convergence',
                            '--clip', clip['baseline'], '--output', str(directory),
                            '--iterations', str(manifest['iterations'])], check=True)
        report = summarize(output, manifest)
        print(json.dumps(dict(completed=report['completed'], knees=report.get('knees'),
                              all_joints=report.get('all_joints'))), flush=True)
    return summarize(output, manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation/opencap_5000steps'))
    args = parser.parse_args()
    run_batch(args.output)
