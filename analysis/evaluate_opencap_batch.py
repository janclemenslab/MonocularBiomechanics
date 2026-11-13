"""Evaluate a predeclared OpenCap sample and render each synchronized comparison."""
import argparse
from collections import defaultdict, deque
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import struct
import sys
import subprocess
import zipfile

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import numpy as np

from analysis.evaluate_opencap import read_mot
from analysis.extract_opencap_pilot import extract_subset
from analysis.opencap_videos import comparison_video
from analysis.score_opencap_joints import score_batch


def write_text_atomic(path, text):
    partial = path.with_name(path.name + f'.{os.getpid()}.tmp')
    partial.write_text(text)
    partial.replace(path)


def catalogue(archive):
    """Join synchronized videos to marker-based references without looking at predictions."""
    result = []
    with zipfile.ZipFile(archive) as zipped:
        names = set(zipped.namelist())
        for name in sorted(names):
            match = re.fullmatch(r'LabValidation_withVideos/(subject\d+)/VideoData/(Session\d+)/(Cam\d+)/(\w+)/\4_syncdWithMocap\.avi', name)
            if match is None:
                continue
            subject, session, camera, trial = match.groups()
            root = f'LabValidation_withVideos/{subject}/'
            reference = root + f'OpenSimData/Mocap/IK/{trial}.mot'
            markers = root + f'MarkerData/Mocap/{trial}.trc'
            models = sorted(n for n in names if n.startswith(root + 'OpenSimData/Mocap/Model/') and n.endswith('_scaled.osim'))
            calibration = root + f'VideoData/{session}/{camera}/cameraIntrinsicsExtrinsics.pickle'
            if not {reference, markers, calibration}.issubset(names):
                continue
            if len(models) != 1:
                raise ValueError(f'Expected one scaled marker-based model for {subject}, found {models}.')
            with zipped.open(name) as handle:
                header = handle.read(65536)
            pos = header.index(b'avih') + 8
            info = struct.unpack_from('<10I', header, pos)
            pos = header.index(b'strh') + 8
            if header[pos:pos + 4] != b'vids':
                raise ValueError(f'Expected the first AVI stream to be video: {name}')
            scale, rate = struct.unpack_from('<II', header, pos + 20)
            fps, frames = rate / scale, info[4]
            mot = zipped.read(reference).decode().splitlines()
            end = next(i for i, line in enumerate(mot) if line.strip().lower() == 'endheader')
            rows = [line for line in mot[end + 2:] if line.strip()]
            first_time, last_time = float(rows[0].split()[0]), float(rows[-1].split()[0])
            first_time = max(0, first_time)
            last_time = min((frames - 1) / fps, last_time)
            duration = last_time - first_time
            if duration < .2:
                continue
            result.append(dict(id=f'{subject}_{trial}_{session}_{camera}', subject=subject, trial=trial,
                               activity=re.sub(r'\d+$', '', trial), session=session, camera=camera,
                               video=name, mocap_ik=reference, mocap_trc=markers, camera_parameters=calibration,
                               reference_model=models[0], available_start=first_time,
                               available_seconds=duration, source_fps=fps,
                               source_frames=frames))
    return result


def select_clips(candidates, count, *, allow_short=False, multiple_views=False, seed=20261106):
    candidates = [dict(c) for c in candidates if allow_short or c['available_seconds'] >= 5]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    groups = defaultdict(deque)
    for candidate in candidates:
        groups[(candidate['subject'], candidate['activity'])].append(candidate)
    keys = sorted(groups)
    selected, seen = [], set()
    while len(selected) < count:
        before = len(selected)
        for key in keys:
            while groups[key]:
                candidate = groups[key].popleft()
                trial_key = candidate['subject'], candidate['trial']
                if not multiple_views and trial_key in seen:
                    continue
                seen.add(trial_key)
                duration = min(5., candidate['available_seconds'])
                start = candidate['available_start'] + max(0, (candidate['available_seconds'] - duration) / 2)
                start = np.ceil(start * candidate['source_fps']) / candidate['source_fps']
                selected.append(dict(candidate, start=start, duration=duration))
                break
            if len(selected) == count:
                break
        if before == len(selected):
            raise ValueError(f'Only {len(selected)} eligible clips for the requested selection of {count}.')
    return selected


def fit_ready(directory, iterations, fps, duration):
    expected = int(np.ceil(duration * fps - 1e-9))
    try:
        report = json.loads((directory / 'metrics.json').read_text())
        if report['iterations'] != iterations or report['sampled_frames'] != expected:
            return False
        with np.load(directory / 'trajectories.npz') as data:
            return all(data[name + '_sites_camera_mm'].shape == (expected, 87, 3)
                       for name in ('original', 'torch'))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False


def verify_outputs(output, manifest):
    """Independently check the saved clocks, reference scores, and playable video endpoints."""
    import cv2
    verified, errors = [], []
    fps = manifest['settings']['sample_fps']
    skipped = [c['id'] for c in manifest['clips'] if (output / c['id'] / 'skipped.json').exists()]
    for clip in manifest['clips']:
        if clip['id'] in skipped:
            continue
        directory = output / clip['id']
        expected = int(np.ceil(clip['duration'] * fps - 1e-9))
        try:
            report = json.loads((directory / 'metrics.json').read_text())
            assert fit_ready(directory, manifest['settings']['iterations'], fps, clip['duration']), 'Incomplete fitted trajectories'
            assert report['overlapping_frames'] == expected, 'Incomplete ground-truth coverage'
            assert report['video'].endswith(clip['video']), 'Wrong source video'
            with np.load(directory / 'trajectories.npz') as arrays:
                times = arrays['reference_time']
                assert times.shape == (expected,) and np.all(np.diff(times) > 0), 'Invalid reference clock'
                np.testing.assert_allclose(times, arrays['source_time'], atol=1e-12)
                truth_time, truth = read_mot(report['mocap_ik'])
                assert times[0] >= truth_time[0] and times[-1] <= truth_time[-1], 'Reference extrapolation'
                reference = np.column_stack([np.interp(times, truth_time, truth[:, j]) for j in range(2)])
                np.testing.assert_allclose(arrays['reference_knees_deg'], reference, atol=1e-9)
                for pipeline in ('original', 'torch'):
                    prediction = arrays[pipeline + '_knees_deg']
                    assert prediction.shape == (expected, 2) and np.isfinite(prediction).all(), 'Invalid knee predictions'
                    assert np.isfinite(arrays[pipeline + '_sites_camera_mm']).all(), 'Invalid fitted markers'
                    rows = [r for r in report['metrics'] if r['pipeline'] == pipeline]
                    assert len(rows) == 2, 'Missing knee scores'
                    for j, row in enumerate(rows):
                        error = prediction[:, j] - reference[:, j]
                        assert row['samples'] == expected, 'Wrong sample count'
                        np.testing.assert_allclose(row['mae_deg'], np.abs(error).mean(), atol=1e-5)
                        np.testing.assert_allclose(row['rmse_deg'], np.sqrt(np.mean(error**2)), atol=1e-5)
            for filename in ('prepared.mp4', 'comparison.mp4'):
                capture = cv2.VideoCapture(str(directory / filename))
                try:
                    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected, filename + ' frame count'
                    assert abs(capture.get(cv2.CAP_PROP_FPS) - fps) < 1e-4, filename + ' frame rate'
                    ok, _ = capture.read()
                    assert ok, filename + ' first frame'
                    capture.set(cv2.CAP_PROP_POS_FRAMES, expected - 1)
                    ok, _ = capture.read()
                    assert ok, filename + ' last frame'
                finally:
                    capture.release()
            verified.append(clip['id'])
        except Exception as error:
            errors.append(dict(clip_id=clip['id'], error=repr(error)))
    expected_count = len(manifest['clips']) - len(skipped)
    result = dict(status='passed' if len(verified) == expected_count else 'failed',
                  checked_clips=len(verified), expected_clips=expected_count,
                  selected_clips=len(manifest['clips']), skipped=skipped, errors=errors)
    write_text_atomic(output / 'verification.json', json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def summarize(output, manifest):
    records, completed, failures, skipped = [], [], [], []
    for clip in manifest['clips']:
        directory = output / clip['id']
        metrics = directory / 'metrics.json'
        if (directory / 'skipped.json').exists():
            skipped.append(dict(id=clip['id'], **json.loads((directory / 'skipped.json').read_text())))
            continue
        if metrics.exists() and (directory / 'comparison.mp4').exists():
            report = json.loads(metrics.read_text())
            completed.append(clip)
            records.extend(dict(clip_id=clip['id'], subject=clip['subject'], trial=clip['trial'],
                                activity=clip['activity'], camera=clip['camera'], **row) for row in report['metrics'])
        elif (directory / 'error.txt').exists():
            failures.append(dict(id=clip['id'], error=(directory / 'error.txt').read_text()))
    if records:
        partial = output / f'metrics.{os.getpid()}.tmp'
        with open(partial, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
        partial.replace(output / 'metrics.csv')
    aggregates = {}
    for pipeline in ('original', 'torch'):
        rows = [r for r in records if r['pipeline'] == pipeline]
        if not rows:
            continue
        by_trial, by_subject = defaultdict(list), defaultdict(list)
        for row in rows:
            by_trial[(row['subject'], row['trial'], row['joint'])].append(row['mae_deg'])
        for (subject, trial, joint), values in by_trial.items():
            by_subject[subject].append(float(np.mean(values)))
        aggregates[pipeline] = dict(clip_joint_macro_mae_deg=float(np.mean([r['mae_deg'] for r in rows])),
                                    trial_joint_macro_mae_deg=float(np.mean([np.mean(v) for v in by_trial.values()])),
                                    subject_macro_mae_deg=float(np.mean([np.mean(v) for v in by_subject.values()])))
    finished = len(completed) + len(skipped) == len(manifest['clips'])
    report = dict(status=('complete_with_skips' if skipped else 'complete') if finished else 'running',
                  requested=len(manifest['clips']), completed=len(completed), failures=failures, skipped=skipped,
                  distinct_trials=len({(c['subject'], c['trial']) for c in completed}),
                  participants=len({c['subject'] for c in completed}), aggregates=aggregates,
                  limitations=['Bilateral knee flexion only.', 'Fixed optimization budget; convergence is not established.',
                               'Camera views of a trial are correlated; aggregate trial and participant scores are reported.',
                               '3D views are pelvis centered with initial pelvis headings aligned for display only.'])
    all_joints = score_batch(output, manifest)
    report['all_joint_scores'] = 'all_joint_summary.json'
    write_text_atomic(output / 'summary.json', json.dumps(report, indent=2, allow_nan=False) + '\n')
    if (output / 'torch_5000/gallery.json').exists():
        return report  # Keep the published 5k gallery when resuming the earlier evaluation.
    cards = []
    activities = {'STS': 'Sit-to-stand', 'STSweakLegs': 'Weak-leg sit-to-stand',
                  'squats': 'Squats', 'squatsAsym': 'Asymmetric squats'}
    for clip in completed:
        title = html.escape(f"Participant {clip['subject'].removeprefix('subject')} · "
                            f"{activities.get(clip['activity'], clip['activity'])} · "
                            f"Camera {clip['camera'].removeprefix('Cam')}")
        source = html.escape(clip['id'] + '/comparison.mp4', quote=True)
        cards.append(f'<article><h2>{title}</h2><video controls preload="none" src="{source}"></video></article>')
    rows = ''.join(f'<tr><td>{"Original" if name == "original" else "Torch"}</td>'
                   f'<td>{values["clip_joint_macro_mae_deg"]:.2f}°</td>'
                   f'<td>{values["trial_joint_macro_mae_deg"]:.2f}°</td>'
                   f'<td>{values["subject_macro_mae_deg"]:.2f}°</td></tr>'
                   for name, values in aggregates.items())
    results_title = 'Knee-angle error' if finished else 'Partial knee-angle results'
    joint_rows = ''.join(f'<tr><td>{html.escape(joint)}'
        f'{" (nearly static reference)" if all_joints["pipelines"]["torch"]["joints"][joint]["static_reference"] else ""}</td>' + ''.join(
        f'<td>{all_joints["pipelines"][pipeline]["joints"][joint]["mae_deg"]:.2f}°</td>'
        for pipeline in ('original', 'torch')) + '</tr>' for joint in all_joints['joints']) if completed else ''
    text = f'''<!doctype html><meta charset="utf-8"><title>OpenCap evaluation</title>
<style>body{{font:16px system-ui;margin:28px;background:#f5f5f5;color:#202020}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,600px),1fr));gap:24px}}article{{background:white;padding:16px;border-radius:12px}}h2{{font-size:17px}}video{{width:100%}}table{{border-collapse:collapse;background:white;margin-bottom:24px}}td,th{{padding:12px 18px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>OpenCap: ground truth and predictions</h1><p>{len(completed)} videos ready · {len(skipped)} skipped · {len(manifest['clips'])} selected clips.</p>
<p>Each video: footage with detections, motion-capture marker skeleton, original fitted skeleton, Torch fitted skeleton. All 3D views share scale; pelvis translation and initial heading are removed for display only.</p>
<h2>{results_title}</h2><p>Mean absolute error in bilateral knee flexion. Repeated camera views are grouped when averaging by trial and participant.</p>
<table><tr><th>Pipeline</th><th>By camera clip</th><th>By movement trial</th><th>By participant</th></tr>{rows}</table>
<p><a href="metrics.csv">Download all scores</a> · <a href="summary.json">Evaluation details</a></p>
<h2>All comparable joint angles</h2><p>Mean absolute error for 27 relative joint coordinates. Neck and wrist angles have no ground truth; global pelvis coordinates require a verified camera-to-mocap frame mapping. Toe and knee signs are converted to the reference convention. Nearly static means reference motion below 0.01° in every scored clip; these coordinates do not establish motion-tracking accuracy.</p>
<table><tr><th>Joint coordinate</th><th>Original MAE</th><th>Torch MAE</th></tr>{joint_rows}</table>
<p><a href="all_joint_metrics.csv">All joint scores by clip</a> · <a href="all_joint_summary.json">Joint averages, exclusions and limitations</a></p>
<main>{''.join(cards)}</main>'''
    write_text_atomic(output / 'index.html', text)
    return report


def run_batch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = dict(count=args.count, allow_short=args.allow_short_clips, multiple_views=args.multiple_views,
                        iterations=args.iterations, sample_fps=args.sample_fps, detector_device=args.detector_device)
        if manifest['settings'] != expected:
            raise ValueError('Existing manifest has different settings. Use its original settings or another output directory.')
    else:
        clips = select_clips(catalogue(args.archive), args.count, allow_short=args.allow_short_clips, multiple_views=args.multiple_views)
        manifest = dict(dataset='OpenCap LabValidation_withVideos v0.1', archive=str(args.archive.resolve()), seed=20261106,
                        settings=dict(count=args.count, allow_short=args.allow_short_clips, multiple_views=args.multiple_views,
                                      iterations=args.iterations, sample_fps=args.sample_fps, detector_device=args.detector_device), clips=clips)
        write_text_atomic(manifest_path, json.dumps(manifest, indent=2) + '\n')
    print('Selected',len(manifest['clips']),'clips;',len({(c['subject'], c['trial']) for c in manifest['clips']}),'distinct trials', flush=True)
    if args.prepare_only:
        return
    write_text_atomic(args.output / 'verification.json', json.dumps(dict(status='running', expected_clips=len(manifest['clips'])), indent=2)+'\n')
    for index, clip in enumerate(manifest['clips']):
        directory = args.output / clip['id']
        if (directory / 'skipped.json').exists():
            continue
        if (directory / 'comparison.mp4').exists() and fit_ready(directory, args.iterations, args.sample_fps, clip['duration']):
            continue
        directory.mkdir(exist_ok=True)
        print(f'[{index + 1}/{len(manifest["clips"])}] {clip["id"]}', flush=True)
        try:
            members = [clip[k] for k in ('video', 'mocap_ik', 'mocap_trc', 'camera_parameters', 'reference_model')]
            paths = extract_subset(args.archive, args.cache, members)
            local = dict(zip(('video', 'mocap_ik', 'mocap_trc', 'camera_parameters', 'reference_model'), paths))
            provenance = [dict(member=n, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in zip(members,paths)]
            write_text_atomic(directory / 'input_provenance.json', json.dumps(provenance, indent=2)+'\n')
            if not fit_ready(directory, args.iterations, args.sample_fps, clip['duration']):
                command = [sys.executable, '-u', '-m', 'analysis.evaluate_opencap',
                           '--reference-python', str(args.reference_python), '--skip-shared-fit',
                           '--time-offset', '0', '--sync-source', 'Dataset-provided *_syncdWithMocap.avi; shared source-frame clock',
                           '--start', str(clip['start']), '--duration', str(clip['duration']),
                           '--sample-fps', str(args.sample_fps), '--iterations', str(args.iterations),
                           '--detector-device', args.detector_device, '--output', str(directory)]
                for option in ('video','mocap_ik','camera_parameters','reference_model'):
                    command.extend(['--'+option.replace('_','-'),str(local[option])])
                # Reclaim TensorFlow/Metal and JAX allocations between clips on this 32 GB Mac.
                subprocess.run(command, check=True)
            comparison_video(directory, local['mocap_trc'])
            (directory / 'error.txt').unlink(missing_ok=True)
        except Exception:
            import traceback
            write_text_atomic(directory / 'error.txt', traceback.format_exc())
            if args.skip_reference_errors and (directory / 'tensorflow_keypoints.error.json').exists():
                reason = json.loads((directory / 'tensorflow_keypoints.error.json').read_text())
                write_text_atomic(directory / 'skipped.json', json.dumps(reason) + '\n')
                print('Skipping reference GPU failure as requested.', flush=True)
            else:
                print('Clip failed; recorded error and continuing.', flush=True)
        print(json.dumps(summarize(args.output, manifest)), flush=True)
    report = summarize(args.output, manifest)
    verification = verify_outputs(args.output, manifest)
    if report['completed'] + len(report['skipped']) != len(manifest['clips']) or verification['status'] != 'passed':
        report['status']='incomplete'
        write_text_atomic(args.output / 'summary.json', json.dumps(report,indent=2)+'\n')
        raise RuntimeError(f'Only {verification["checked_clips"]} verified clips. Review summary.json and verification.json.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path.home()/'Downloads/LabValidation_withVideos.zip')
    parser.add_argument('--reference-python', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/monocular-biomechanics/opencap')
    parser.add_argument('--output', type=Path, default=Path('validation/opencap_100'))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--allow-short-clips', action='store_true', help='Distinct trials, each up to five seconds.')
    selection.add_argument('--multiple-views', action='store_true', help='Five-second clips with correlated camera views of longer trials.')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--sample-fps', type=float, default=15)
    parser.add_argument('--detector-device', choices=['cpu','cuda','mps'], default='cpu')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--skip-reference-errors', action='store_true', help='Record and skip clips where TensorFlow GPU detection aborts.')
    args = parser.parse_args()
    if args.count < 1 or args.iterations < 10 or not np.isfinite(args.sample_fps) or args.sample_fps <= 0:
        parser.error('Count/FPS must be positive and iterations at least ten.')
    run_batch(args)


if __name__ == '__main__':
    main()
