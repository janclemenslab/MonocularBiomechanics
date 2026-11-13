"""Pilot evaluation of monocular knee flexion against OpenCap marker-based IK.

Inputs must be one camera video, its calibration, marker-based OpenSim IK, and
an independently verified video-to-IK time offset. The original and Torch fits
start from identical weights. All reference scoring uses the same timestamps.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

KNEES = ('knee_angle_r', 'knee_angle_l')


def detect_reference(python, video, output, calibration):
    Path(output).with_suffix('.error.json').unlink(missing_ok=True)
    command = [str(python), str(Path(__file__).with_name('detect_reference.py')),
               '--video', str(video), '--output', str(output), '--calibration', str(calibration)]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        if error.returncode == -signal.SIGABRT:
            Path(output).with_suffix('.error.json').write_text(json.dumps(
                dict(reason='TensorFlow GPU backend aborted', returncode=error.returncode)) + '\n')
        raise


def read_mot(path, joints=KNEES):
    """Read OpenSim storage with explicit angle units and strictly increasing time."""
    lines = Path(path).read_text().splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip().lower() == 'endheader'), None)
    if end is None:
        raise ValueError('OpenSim IK file must contain an endheader line.')
    metadata = {}
    for line in lines[:end]:
        if '=' in line:
            key, value = line.split('=', 1)
            metadata[key.strip().lower()] = value.strip().lower()
    units = metadata.get('indegrees')
    if units not in ('yes', 'no'):
        raise ValueError('Reference IK must declare inDegrees=yes or inDegrees=no.')
    table = [line for line in lines[end + 1:] if line.strip()]
    if len(table) < 3:
        raise ValueError('Reference IK must contain a header and at least two samples.')
    names = table[0].split()
    values = np.loadtxt(table[1:], ndmin=2)
    if values.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError('Reference IK columns must be unique and match the data.')
    for name in ('time', *joints):
        if name not in names:
            raise ValueError(f'Reference IK is missing {name}.')
    timestamps = values[:, names.index('time')]
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Reference timestamps must be finite and strictly increasing.')
    angles = values[:, [names.index(k) for k in joints]]
    if units == 'no':
        angles = np.rad2deg(angles)
    return timestamps, angles


def knee_flexion(qpos, model):
    """MuJoCo knee coordinates are negative in flexion, OpenCap's are positive."""
    addresses = [model.joint(name).qposadr[0] for name in KNEES]
    return -np.rad2deg(np.asarray(qpos)[:, addresses])


def score_knees(timestamps, prediction, reference_time, reference_angles):
    return score_angles(timestamps, prediction, reference_time, reference_angles, KNEES)


def score_angles(timestamps, prediction, reference_time, reference_angles, joints):
    timestamps, prediction = np.asarray(timestamps), np.asarray(prediction)
    reference_time, reference_angles = np.asarray(reference_time), np.asarray(reference_angles)
    if prediction.shape != (len(timestamps), len(joints)) or reference_angles.shape != (len(reference_time), len(joints)):
        raise ValueError('Angles must have one row per timestamp and one column per joint.')
    if reference_time.ndim != 1 or len(reference_time) < 2 or not np.isfinite(reference_time).all() or np.any(np.diff(reference_time) <= 0):
        raise ValueError('Reference timestamps must be finite and strictly increasing.')
    if timestamps.ndim != 1 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Evaluation timestamps must be finite and strictly increasing.')
    overlap = (timestamps >= reference_time[0]) & (timestamps <= reference_time[-1])
    interpolated = np.column_stack([np.interp(timestamps, reference_time, reference_angles[:, j]) for j in range(len(joints))])
    interpolated[~overlap] = np.nan
    rows = []
    for j, name in enumerate(joints):
        valid = overlap & np.isfinite(prediction[:, j]) & np.isfinite(interpolated[:, j])
        if valid.sum() < 3:
            raise ValueError(f'Insufficient overlapping valid samples for {name}. Check synchronization.')
        observed, predicted = interpolated[valid, j], prediction[valid, j]
        error = predicted - observed
        correlation = None
        if np.std(predicted) > 1e-8 and np.std(observed) > 1e-8:
            correlation = float(np.corrcoef(predicted, observed)[0, 1])
        rows.append(dict(joint=name, samples=int(valid.sum()), mae_deg=float(np.abs(error).mean()),
                         median_absolute_error_deg=float(np.median(np.abs(error))),
                         rmse_deg=float(np.sqrt(np.mean(error**2))), bias_deg=float(error.mean()),
                         correlation=correlation))
    return rows, interpolated, overlap


def validate_reference_model(path):
    """Require OpenCap's positive-flexion, zero-extension knee coordinates."""
    root = ET.parse(path).getroot()
    for name in KNEES:
        coordinate = root.find(f'.//Coordinate[@name="{name}"]')
        if coordinate is None:
            raise ValueError(f'Reference OpenSim model is missing {name}.')
        limits = np.fromstring(coordinate.findtext('range', ''), sep=' ')
        rest = float(coordinate.findtext('default_value', 'nan'))
        if limits.shape != (2,) or limits[0] < -0.25 or limits[0] > 0 or limits[1] < 1 or abs(rest) > 1e-6:
            raise ValueError(f'Unexpected reference knee convention for {name}. Verify the model mapping.')


def load_calibration(path, frame_shape):
    with open(path, 'rb') as handle:
        parameters = pickle.load(handle)
    intrinsic = np.asarray(parameters['intrinsicMat'], dtype=float)
    distortion = np.asarray(parameters['distortion'], dtype=float).ravel()
    size = tuple(np.asarray(parameters['imageSize']).ravel().astype(int))
    if size != tuple(frame_shape[:2]):
        raise ValueError(f'Calibration imageSize {size} differs from video {frame_shape[:2]}. Resolve orientation/resizing first.')
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or not np.isfinite(distortion).all():
        raise ValueError('Camera calibration must contain finite intrinsics and distortion.')
    return intrinsic, distortion


def prepare_video(video, calibration_path, destination, start, duration, sample_fps):
    """Undistort a short sample with true intrinsics and preserve source-frame times."""
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ok, first = cap.read()
        if not ok or fps <= 0:
            raise ValueError(f'Cannot read video and frame rate from {video}.')
        intrinsic, distortion = load_calibration(calibration_path, first.shape)
        height, width = first.shape[:2]
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = min(start + duration, count / fps)
        # Count samples before rounding so floating clip endpoints cannot add a frame.
        samples = max(0, int(np.ceil((end - start) * sample_fps - 1e-9)))
        indices = np.unique(np.rint((start + np.arange(samples) / sample_fps) * fps).astype(int))
        indices = indices[(indices < count) & (indices / fps < end - 1e-9)]
        if len(indices) < 3:
            raise ValueError('Requested clip contains fewer than three frames.')
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), sample_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create prepared video {destination}.')
        mapx, mapy = cv2.initUndistortRectifyMap(intrinsic, distortion, None, intrinsic, (width, height), cv2.CV_32FC1)
        try:
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode source frame {index}.')
                writer.write(cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR))
        finally:
            writer.release()
    finally:
        cap.release()
    # Fit in the camera frame. Knee angles are independent of world extrinsics.
    calibration = dict(mtx=(intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]] / 1000)[None],
                       dist=np.zeros((1, 5)), rvec=np.zeros((1, 3)), tvec=np.zeros((1, 3)))
    np.savez(destination.with_suffix('.calibration.npz'), **calibration)
    return indices / fps, calibration, intrinsic


def detect_torch(video, intrinsic, output, device=None):
    from monocular_demos.utils import load_metrabs, video_reader
    detector = load_metrabs(device=device)
    results = {name: [] for name in ('keypoints2d', 'keypoints3d', 'confs')}
    frames, _ = video_reader(str(video), batch_size=8 if device == 'mps' else 2)
    for batch in frames:
        prediction = detector.detect_poses_batched(batch, intrinsic_matrix=intrinsic[None])
        for i in range(len(batch)):
            found = len(prediction['boxes'][i]) > 0
            results['keypoints2d'].append(prediction['poses2d'][i][0].numpy() if found else np.zeros((87, 2)))
            results['keypoints3d'].append(prediction['poses3d'][i][0].numpy() if found else np.zeros((87, 3)))
            results['confs'].append(np.ones(87) if found else np.zeros(87))
    np.savez(output, **results)


def run_evaluation(args):
    import jax
    import torch
    from monocular_demos.dataset import MonocularDataset as JaxDataset
    from monocular_demos.torch_dataset import MonocularDataset
    from monocular_demos.biomechanics_mjx.monocular_trajectory import get_model as original_model, fit_model as original_fit
    from monocular_demos.biomechanics_mjx.torch_monocular_trajectory import get_model, fit_model
    from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics
    from monocular_demos.utils import joint_names
    from analysis.validate_torch_port import transfer_weights

    jax.config.update('jax_enable_x64', True)
    torch.set_num_threads(4)
    validate_reference_model(args.reference_model)
    reference_time, reference_angles = read_mot(args.mocap_ik)
    args.output.mkdir(parents=True, exist_ok=True)
    video = args.output / 'prepared.mp4'
    source_times, calibration, intrinsic = prepare_video(args.video, args.camera_parameters, video, args.start, args.duration, args.sample_fps)
    evaluation_time = source_times + args.time_offset
    # Validate time coverage before detector downloads or model fitting.
    score_knees(evaluation_time, np.zeros((len(source_times), 2)), reference_time, reference_angles)
    fingerprint = hashlib.sha256(video.read_bytes() + intrinsic.tobytes()).hexdigest()
    fingerprint_path = args.output / 'detection_input.sha256'
    cache_valid = fingerprint_path.exists() and fingerprint_path.read_text().strip() == fingerprint
    tf_keypoints, torch_keypoints = args.output / 'tensorflow_keypoints.npz', args.output / 'torch_keypoints.npz'
    if not cache_valid or not tf_keypoints.exists():
        detect_reference(args.reference_python, video, tf_keypoints, video.with_suffix('.calibration.npz'))
    if not cache_valid or not torch_keypoints.exists():
        detect_torch(video, intrinsic, torch_keypoints, getattr(args, 'detector_device', None))
    fingerprint_path.write_text(fingerprint + '\n')
    # Detection weights and Metal caches are not needed by the fitting stage.
    from monocular_demos.metrabs_torch import load_metrabs
    import gc
    load_metrabs.cache_clear()
    gc.collect()
    if getattr(args, 'detector_device', None) == 'mps':
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    def make_dataset(cls, path):
        with np.load(path) as data:
            if len(data['confs']) != len(source_times):
                raise ValueError('Detection count differs from prepared video frames.')
            return cls(timestamps=[source_times.copy()], keypoints_2d=[data['keypoints2d'][None]],
                       keypoints_3d=[data['keypoints3d'][None]], keypoint_confidence=[data['confs'][None]],
                       camera_params=deepcopy(calibration))

    original_data = make_dataset(JaxDataset, tf_keypoints)
    shared_data, torch_data = make_dataset(MonocularDataset, tf_keypoints), make_dataset(MonocularDataset, torch_keypoints)
    options = dict(joint_names=joint_names, jmp_policy='params=float32,compute=float32,output=float32')
    original = original_model(original_data, **options)
    shared = None if getattr(args, 'skip_shared_fit', False) else get_model(shared_data, **options)
    complete = get_model(torch_data, **options)
    if shared is not None:
        transfer_weights(original, shared)
    transfer_weights(original, complete)
    times = source_times - source_times[0]
    trajectories, positions, timings = {}, {}, {}
    start = time.perf_counter()
    original, _ = original_fit(original, original_data, max_iters=args.iterations, lr_init_value=1e-3)
    original_state = original(times, skip_vel=True, skip_action=True, steps=0, check_constraints=False)
    trajectories['original'] = np.asarray(original_state[1][0])
    positions['original'] = np.asarray(original_state[0][0].site_xpos) * 1000
    timings['original'] = time.perf_counter() - start
    for name, model, data in (('torch_shared_detections', shared, shared_data), ('torch', complete, torch_data)):
        if model is None:
            continue
        start = time.perf_counter()
        model, _ = fit_model(model, data, max_iters=args.iterations, lr_init_value=1e-3)
        with torch.no_grad():
            state = model(times, skip_vel=True)
            trajectories[name] = state[1][0].cpu().numpy()
            positions[name] = state[0][0].site_xpos.cpu().numpy() * 1000
        timings[name] = time.perf_counter() - start
    model = ForwardKinematics().model
    table, predictions = [], {}
    for name, qpos in trajectories.items():
        predictions[name] = knee_flexion(qpos, model)
        rows, interpolated, overlap = score_knees(evaluation_time, predictions[name], reference_time, reference_angles)
        table.extend(dict(pipeline=name, **row) for row in rows)
    agreement = []
    for name in ('torch_shared_detections', 'torch'):
        if name not in predictions:
            continue
        rows, _, _ = score_knees(evaluation_time, predictions[name], evaluation_time, predictions['original'])
        agreement.extend(dict(pipeline=name, **row) for row in rows)
    report = dict(status='evaluated', video=str(args.video.resolve()), mocap_ik=str(args.mocap_ik.resolve()),
                  camera_parameters=str(args.camera_parameters.resolve()), reference_model=str(args.reference_model.resolve()),
                  reference='marker-based OpenSim inverse kinematics', joints=list(KNEES),
                  knee_sign_mapping=-1, time_offset_seconds=args.time_offset, sync_source=args.sync_source,
                  start_seconds=args.start, duration_seconds=args.duration, sampled_frames=len(times),
                  overlapping_frames=int(overlap.sum()), iterations=args.iterations,
                  initialization='identical JAX weights copied to each Torch fit', compute='float32',
                  detector_device=getattr(args, 'detector_device', None) or 'cpu',
                  reference_detector=(json.loads(tf_keypoints.with_suffix('.metadata.json').read_text())
                                      if tf_keypoints.with_suffix('.metadata.json').exists() else {'device': 'cached, unrecorded'}),
                  timings_seconds=timings, metrics=table, agreement_with_original=agreement,
                  versions=dict(torch=torch.__version__, jax=jax.__version__),
                  limitations=[f'Pilot on one camera/trial at a fixed {args.iterations}-step fitting budget; convergence is not established.',
                               'Different musculoskeletal models define knee coordinates.',
                               'Synchronization relies on the supplied sync_source; no lag or angle offset is fitted.'])
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with open(args.output / 'metrics.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    np.savez(args.output / 'trajectories.npz', source_time=source_times, reference_time=evaluation_time,
             reference_knees_deg=interpolated, overlap=overlap, **{k + '_knees_deg': v for k, v in predictions.items()},
             **{k + '_qpos': v for k, v in trajectories.items()},
             **{k + '_sites_camera_mm': v for k, v in positions.items()})
    plot_results(args.output, evaluation_time, interpolated, predictions)
    print(json.dumps(report, indent=2))
    return report


def plot_results(destination, timestamps, reference, predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6), constrained_layout=True)
    labels = {'original': 'Original', 'torch_shared_detections': 'Torch (same detections)',
              'torch': 'Torch (own detections)'}
    for j, axis in enumerate(axes):
        axis.plot(timestamps, reference[:, j], color='black', label='Marker-based IK', linewidth=2)
        for name, values in predictions.items():
            axis.plot(timestamps, values[:, j], label=labels[name],
                      linestyle='--' if name == 'torch_shared_detections' else '-')
        axis.set(ylabel='Flexion (degrees)', title=('Right knee', 'Left knee')[j])
        axis.legend(frameon=False)
    axes[-1].set_xlabel('Reference time (seconds)')
    figure.savefig(destination / 'knee_flexion.png', dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--mocap-ik', type=Path, required=True, help='Marker-based IK .mot file, selected from the Mocap branch.')
    parser.add_argument('--reference-model', type=Path, required=True, help='Scaled OpenSim .osim model used for marker-based IK.')
    parser.add_argument('--camera-parameters', type=Path, required=True, help='Official cameraIntrinsicsExtrinsics.pickle for this camera/session.')
    parser.add_argument('--reference-python', type=Path, required=True, help='Python executable in the TensorFlow detector environment.')
    parser.add_argument('--time-offset', type=float, required=True, help='Reference IK time corresponding to source video frame zero, in seconds.')
    parser.add_argument('--sync-source', required=True, help='Metadata file or independently verified event used to establish time offset.')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--duration', type=float, default=5)
    parser.add_argument('--sample-fps', type=float, default=15)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--skip-shared-fit', action='store_true', help='Compare complete original and Torch pipelines only.')
    parser.add_argument('--detector-device', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--output', type=Path, default=Path('validation/opencap'))
    args = parser.parse_args()
    if not np.isfinite([args.start, args.duration, args.sample_fps, args.time_offset]).all() or args.start < 0 or args.duration <= 0 or args.sample_fps <= 0 or args.iterations < 10:
        parser.error('Start must be nonnegative. Duration/FPS must be positive and iterations at least 10.')
    for path in (args.video, args.mocap_ik, args.reference_model, args.camera_parameters, args.reference_python):
        if not path.is_file():
            parser.error(f'Required input does not exist: {path}')
    run_evaluation(args)


if __name__ == '__main__':
    main()
