"""Score all comparable relative joint coordinates from saved OpenCap fits."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from analysis.evaluate_opencap import KNEES, read_mot, score_angles, validate_reference_model
from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics

JOINTS = tuple(f'{name}_{side}' for side in ('r', 'l') for name in
               ('hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle',
                'ankle_angle', 'subtalar_angle', 'mtp_angle')) + (
               'lumbar_extension', 'lumbar_bending', 'lumbar_rotation') + tuple(
               f'{name}_{side}' for side in ('r', 'l') for name in
               ('arm_flex', 'arm_add', 'arm_rot', 'elbow_flex', 'pro_sup'))
EXCLUDED = {
    'neck_extension, neck_bending, neck_rotation': 'No neck coordinates in reference model or IK.',
    'wrist_flex_r/l, wrist_dev_r/l': 'Reference wrists are welded; no measured wrist coordinates.',
    'pelvis_tilt, pelvis_list, pelvis_rotation, pelvis_tx/ty/tz':
        'Global camera-to-mocap frame mapping is not verified; these are not relative joint angles.',
    'knee_angle_r_beta, knee_angle_l_beta': 'Reference patellar coupling coordinates have no Torch counterpart.',
}


def coordinate_signs(path, model, joints=JOINTS):
    """Validate axes/order against the actual OpenSim model, never infer signs from data."""
    root = ET.parse(path).getroot()
    if set(KNEES).issubset(joints):
        validate_reference_model(path)
    signs = []
    for name in joints:
        joint = next((j for j in root.findall('.//JointSet/objects/*')
                      if j.find(f'./coordinates/Coordinate[@name="{name}"]') is not None), None)
        if joint is None:
            raise ValueError(f'Missing reference joint coordinate: {name}')
        coordinate = joint.find(f'./coordinates/Coordinate[@name="{name}"]')
        if coordinate.findtext('locked', 'false').lower() == 'true':
            raise ValueError(f'Reference coordinate is locked: {name}')
        if name in KNEES:
            # The reference knee includes coupled rotations/translations; compare flexion coordinates only.
            signs.append(-1.)
            continue
        frames = {f.attrib['name']: f for f in joint.findall('./frames/*')}
        parent = frames[joint.findtext('socket_parent_frame')]
        child = frames[joint.findtext('socket_child_frame')]
        rotations = [Rotation.from_euler('XYZ', np.fromstring(f.findtext('orientation'), sep=' '))
                     for f in (parent, child)]
        if not np.allclose(rotations[0].as_matrix(), rotations[1].as_matrix(), atol=1e-6):
            raise ValueError(f'Nonzero reference neutral rotation: {name}')
        if joint.tag == 'PinJoint':
            axis = rotations[0].apply([0., 0., 1.])
        elif joint.tag == 'CustomJoint':
            axes = joint.findall('./SpatialTransform/TransformAxis')[:3]
            order = [a.findtext('coordinates', '').strip() for a in axes]
            addresses = [int(model.joint(n).qposadr[0]) for n in order]
            if addresses != sorted(addresses):
                raise ValueError(f'Different reference rotation order: {name}')
            transform = axes[order.index(name)]
            coefficients = np.fromstring(transform.findtext('./LinearFunction/coefficients', ''), sep=' ')
            if coefficients.shape != (2,) or abs(coefficients[1]) > 1e-8 or abs(abs(coefficients[0])-1) > 1e-8:
                raise ValueError(f'Nonlinear or offset reference rotation: {name}')
            axis = rotations[0].apply(np.fromstring(transform.findtext('axis'), sep=' ')) * coefficients[0]
        else:
            raise ValueError(f'Unsupported reference joint: {name}')
        native = model.joint(name)
        if not np.allclose(model.body_quat[native.bodyid[0]], [1, 0, 0, 0], atol=1e-6):
            raise ValueError(f'Nonzero prediction neutral rotation: {name}')
        dot = float(np.dot(axis / np.linalg.norm(axis), native.axis))
        if abs(abs(dot)-1) > 1e-5:
            raise ValueError(f'Nonparallel rotation axes: {name}')
        signs.append(float(np.sign(dot)))
    return np.asarray(signs)


def score_saved_fit(directory, model):
    report = json.loads((directory / 'metrics.json').read_text())
    signs = coordinate_signs(report['reference_model'], model)
    reference_time, reference = read_mot(report['mocap_ik'], JOINTS)
    addresses = [int(model.joint(n).qposadr[0]) for n in JOINTS]
    records = []
    with np.load(directory / 'trajectories.npz') as arrays:
        times = arrays['reference_time']
        np.testing.assert_allclose(times, arrays['source_time'] + report['time_offset_seconds'], atol=1e-10)
        if times[0] < reference_time[0] or times[-1] > reference_time[-1]:
            raise ValueError(f'Incomplete reference coverage: {directory.name}')
        for pipeline in ('original', 'torch'):
            prediction = np.rad2deg(arrays[pipeline + '_qpos'][:, addresses]) * signs
            if not np.isfinite(prediction).all() or not np.isfinite(reference).all():
                raise ValueError(f'Non-finite joint coordinates: {directory.name}')
            rows, truth, _ = score_angles(times, prediction, reference_time, reference, JOINTS)
            for j, row in enumerate(rows):
                records.append(dict(pipeline=pipeline, **row, reference_rom_deg=float(np.ptp(truth[:, j]))))
    return records, dict(zip(JOINTS, signs.tolist()))


def aggregate(rows):
    trials, subjects = defaultdict(list), defaultdict(list)
    for r in rows:
        trials[(r['subject'], r['trial'])].append(r['mae_deg'])
    for (subject, _), values in trials.items():
        subjects[subject].append(float(np.mean(values)))
    samples = sum(r['samples'] for r in rows)
    return dict(mae_deg=sum(r['mae_deg']*r['samples'] for r in rows)/samples,
                rmse_deg=float(np.sqrt(sum(r['rmse_deg']**2*r['samples'] for r in rows)/samples)),
                bias_deg=sum(r['bias_deg']*r['samples'] for r in rows)/samples,
                subject_balanced_mae_deg=float(np.mean([np.mean(v) for v in subjects.values()])),
                samples=samples)


def score_batch(output, manifest):
    model = ForwardKinematics().model
    records, completed, skipped, signs = [], [], [], {}
    for clip in manifest['clips']:
        directory = output / clip['id']
        if (directory / 'skipped.json').exists():
            skipped.append(clip['id'])
            continue
        if not (directory / 'comparison.mp4').exists() or not (directory / 'metrics.json').exists():
            continue
        rows, signs = score_saved_fit(directory, model)
        records.extend(dict(clip_id=clip['id'], subject=clip['subject'], trial=clip['trial'],
                            activity=clip['activity'], camera=clip['camera'], **r) for r in rows)
        completed.append(clip)
    pipelines = {}
    for pipeline in ('original', 'torch'):
        rows = [r for r in records if r['pipeline'] == pipeline]
        if rows:
            joint_scores = {}
            for joint in JOINTS:
                selected = [r for r in rows if r['joint'] == joint]
                joint_scores[joint] = dict(aggregate(selected),
                    mean_reference_rom_deg=float(np.mean([r['reference_rom_deg'] for r in selected])),
                    low_motion_clips=sum(r['reference_rom_deg'] < 1 for r in selected),
                    static_reference=all(r['reference_rom_deg'] < .01 for r in selected))
            pipelines[pipeline] = dict(overall=aggregate(rows), joints=joint_scores)
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                  status=('complete_with_skips' if skipped else 'complete')
                         if len(completed)+len(skipped) == len(manifest['clips']) else 'partial',
                  selected=len(manifest['clips']), completed=len(completed), skipped=skipped,
                  distinct_trials=len({(c['subject'], c['trial']) for c in completed}),
                  participants=len({c['subject'] for c in completed}), iterations=manifest['settings']['iterations'],
                  joints=list(JOINTS), prediction_to_reference_signs=signs, excluded=EXCLUDED,
                  static_reference_threshold_rom_deg=.01, low_motion_threshold_rom_deg=1.,
                  pipelines=pipelines, limitations=[
                      'Marker-based OpenSim IK is a model-derived reference, not direct joint-angle measurement.',
                      'Knee flexion coordinates compare models with different knee kinematics.',
                      'Low-motion coordinates can have low error without demonstrating motion-tracking accuracy.',
                      'No fitted time lag, angle offset, or ground-truth-driven alignment.',
                      'Repeated camera views are correlated; participant-balanced averages group views by trial.'])
    partial = output / f'all_joint_summary.{os.getpid()}.tmp'
    partial.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    partial.replace(output / 'all_joint_summary.json')
    if records:
        partial = output / f'all_joint_metrics.{os.getpid()}.tmp'
        with partial.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        partial.replace(output / 'all_joint_metrics.csv')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation/opencap_100'))
    args = parser.parse_args()
    print(json.dumps(score_batch(args.output, json.loads((args.output/'manifest.json').read_text())), indent=2))
