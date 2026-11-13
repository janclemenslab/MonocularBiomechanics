"""Annotated detection and synchronized ground-truth comparison videos."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from monocular_demos.utils import joint_names

BONES = [('Head', 'Neck'), ('Neck', 'LShoulder'), ('Neck', 'RShoulder'),
         ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
         ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
         ('Neck', 'mhip'), ('LHip', 'RHip'), ('LHip', 'mhip'), ('RHip', 'mhip'),
         ('LHip', 'LKnee'), ('LKnee', 'LAnkle'), ('LAnkle', 'LBigToe'),
         ('RHip', 'RKnee'), ('RKnee', 'RAnkle'), ('RAnkle', 'RBigToe')]


def draw_points(frame, points, names, color, confidence=None):
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=-1) & np.any(points != 0, axis=-1)
    if confidence is not None:
        valid &= np.asarray(confidence) > 0
    lookup = {name: i for i, name in enumerate(names)}
    for a, b in BONES:
        if a in lookup and b in lookup and valid[lookup[a]] and valid[lookup[b]]:
            pa, pb = points[[lookup[a], lookup[b]]]
            # Clip lines before integer conversion; projections behind the camera are excluded upstream.
            pa, pb = tuple(np.clip(pa, -10000, 10000).astype(int)), tuple(np.clip(pb, -10000, 10000).astype(int))
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
    for point in points[valid]:
        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
            cv2.circle(frame, tuple(point.astype(int)), 2, color, -1, cv2.LINE_AA)
    return frame


def label(frame, text, row=0, color=(255, 255, 255), outline=True):
    y = 28 + row * 23
    if outline:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 1, cv2.LINE_AA)


def write_video(path, frames, fps):
    """Stream frames to H.264 for playback in the app and ordinary browsers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    with tempfile.TemporaryDirectory() as temporary:
        raw = Path(temporary) / 'video.avi'
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot create video for {path}.')
        try:
            writer.write(first)
            for frame in iterator:
                if frame.shape != first.shape:
                    raise ValueError('Video frames must have identical dimensions.')
                writer.write(frame)
        finally:
            writer.release()
        ffmpeg = Path(sys.executable).with_name('ffmpeg')
        executable = str(ffmpeg) if ffmpeg.exists() else (shutil.which('ffmpeg') or 'ffmpeg')
        partial = path.with_name(path.stem + '.partial.mp4')
        try:
            subprocess.run([executable, '-v', 'error', '-y', '-i', str(raw),
                            '-c:v', 'libx264', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            str(partial)], check=True)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)


def detection_video(directory):
    directory = Path(directory)
    data = {name: np.load(directory / f'{name}_keypoints.npz') for name in ('tensorflow', 'torch')}
    cap = cv2.VideoCapture(str(directory / 'prepared.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)

    def frames():
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                panels = []
                scale = 480 / frame.shape[1]
                for name, description in [('tensorflow', 'Original TensorFlow detections'), ('torch', 'Torch detections')]:
                    panel = cv2.resize(frame, (480, int(round(frame.shape[0] * scale / 2)) * 2))
                    draw_points(panel, data[name]['keypoints2d'][i] * scale, joint_names,
                                (60, 240, 100) if name == 'torch' else (0, 200, 255), data[name]['confs'][i])
                    label(panel, description)
                    count = int((data[name]['confs'][i] > 0).sum())
                    label(panel, f'{i / fps:.2f}s | {count} predicted markers', 1)
                    panels.append(panel)
                yield np.hstack(panels)
                i += 1
        finally:
            cap.release()
            for array in data.values():
                array.close()
    output = directory / 'detections.mp4'
    write_video(output, frames(), fps)
    return output


def read_trc(path):
    lines = Path(path).read_text().splitlines()
    metadata = dict(zip(lines[1].split(), lines[2].split()))
    units = metadata['Units'].lower()
    if units not in ('mm', 'm'):
        raise ValueError(f'Unsupported motion-capture units: {units}')
    names = lines[3].rstrip('\t').split('\t')[2::3]
    values = np.loadtxt([line.strip() for line in lines[6:] if line.strip()], ndmin=2)
    if values.shape != (int(metadata['NumFrames']), 2 + 3 * len(names)):
        raise ValueError('Motion-capture marker columns or frame count do not match the TRC header.')
    time = values[:, 1]
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('Motion-capture timestamps must be finite and strictly increasing.')
    return time, names, values[:, 2:].reshape(len(time), len(names), 3) / (1000 if units == 'mm' else 1)


def reference_skeleton(path, timestamps):
    time, names, markers = read_trc(path)
    if np.any(timestamps < time[0]) or np.any(timestamps > time[-1]):
        raise ValueError('Video frames must lie within the motion-capture reference clock.')
    points = np.stack([np.column_stack([np.interp(timestamps, time, markers[:, j, d])
                                       for d in range(3)]) for j in range(len(names))], axis=1)
    lookup = {name: points[:, i] for i, name in enumerate(names)}
    aliases = {'r_knee': 'RKnee', 'L_knee': 'LKnee', 'r_ankle': 'RAnkle', 'L_ankle': 'LAnkle',
               'r_toe': 'RBigToe', 'L_toe': 'LBigToe', 'r_calc': 'RHeel', 'L_calc': 'LHeel',
               'R_HJC': 'RHip', 'L_HJC': 'LHip', 'C7': 'Neck',
               'R_Shoulder': 'RShoulder', 'L_Shoulder': 'LShoulder',
               'r.ASIS': 'rasis', 'L.ASIS': 'lasis', 'r.PSIS': 'rpsis', 'L.PSIS': 'lpsis'}
    result = np.full((len(timestamps), len(joint_names), 3), np.nan)
    for source, target in aliases.items():
        result[:, joint_names.index(target)] = lookup[source]
    for side, prefix in [('R', 'R'), ('L', 'L')]:
        result[:, joint_names.index(side + 'Elbow')] = (lookup[prefix + '_elbow_med'] + lookup[prefix + '_elbow_lat']) / 2
        result[:, joint_names.index(side + 'Wrist')] = (lookup[prefix + '_wrist_radius'] + lookup[prefix + '_wrist_ulna']) / 2
    result[:, joint_names.index('mhip')] = (lookup['R_HJC'] + lookup['L_HJC']) / 2
    return result


def canonical_skeleton(points):
    """Center each pelvis and use one initial pelvis frame for display, preserving motion/scale."""
    p = {name: np.asarray(points)[:, joint_names.index(name)] for name in ('RHip', 'LHip', 'rasis', 'lasis', 'rpsis', 'lpsis')}
    right = p['RHip'] - p['LHip']
    forward = (p['rasis'] + p['lasis'] - p['rpsis'] - p['lpsis']) / 2
    valid = np.isfinite(right).all(-1) & np.isfinite(forward).all(-1)
    valid &= (np.linalg.norm(right, axis=-1) > 1e-6) & (np.linalg.norm(np.cross(right, forward), axis=-1) > 1e-6)
    if not valid.any():
        raise ValueError('No valid pelvis frame is available for the 3D skeleton display.')
    first = np.flatnonzero(valid)[0]
    z = right[first] / np.linalg.norm(right[first])
    y = np.cross(z, forward[first]); y /= np.linalg.norm(y)
    x = np.cross(y, z)
    center = (p['RHip'] + p['LHip']) / 2
    return (np.asarray(points) - center[:, None]) @ np.column_stack((x, y, z))


def skeleton_panel(points, title, color, scale, center):
    panel = np.full((640, 360, 3), 245, dtype=np.uint8)
    projected = np.column_stack((.8 * points[:, 0] + .6 * points[:, 2],
                                 points[:, 1] - .15 * (.6 * points[:, 0] - .8 * points[:, 2])))
    pixels = (projected - center) * [scale, -scale] + [180, 335]
    draw_points(panel, pixels, joint_names, color)
    label(panel, title, color=color, outline=False)
    return panel


def skeleton_views(skeletons):
    canonical = {key: canonical_skeleton(value) for key, value in skeletons.items()}
    all_points = np.concatenate(list(canonical.values()), axis=1)
    horizontal = .8 * all_points[..., 0] + .6 * all_points[..., 2]
    vertical = all_points[..., 1] - .15 * (.6 * all_points[..., 0] - .8 * all_points[..., 2])
    low, high = np.array([np.nanmin(horizontal), np.nanmin(vertical)]), np.array([np.nanmax(horizontal), np.nanmax(vertical)])
    scale = min(310 / max(high[0] - low[0], .1), 510 / max(high[1] - low[1], .1))
    return canonical, scale, (low + high) / 2


def torch_comparison_frames(timestamps, reference, prediction, reference_knees, prediction_knees, raw_video):
    """Input footage, measured skeleton and fitted sites on one synchronized clock."""
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 1 or len(timestamps) < 3 or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError('Comparison timestamps must be finite and strictly increasing.')
    expected = (len(timestamps), len(joint_names), 3)
    if np.shape(reference) != expected or np.shape(prediction) != expected or not np.isfinite(prediction).all():
        raise ValueError('Comparison skeletons must match the timestamps and marker layout.')
    if any(np.shape(v) != (len(timestamps), 2) or not np.isfinite(v).all()
           for v in (reference_knees, prediction_knees)):
        raise ValueError('Comparison knee angles must be finite and match the timestamps.')
    canonical, scale, center = skeleton_views(dict(reference=reference, torch=prediction))
    cap = cv2.VideoCapture(str(raw_video))
    try:
        if (not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != len(timestamps)
                or not np.allclose(np.diff(timestamps) * cap.get(cv2.CAP_PROP_FPS), 1, atol=1e-6)):
            raise ValueError('Input video frame count and rate must match the comparison timestamps.')
        for i, timestamp in enumerate(timestamps):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode input video frame {i}.')
            h, w = frame.shape[:2]
            factor = min(360 / w, 640 / h)
            small = cv2.resize(frame, (round(w * factor), round(h * factor)))
            video = np.zeros((640, 360, 3), dtype=np.uint8)
            top, left = (640-small.shape[0])//2, (360-small.shape[1])//2
            video[top:top+small.shape[0], left:left+small.shape[1]] = small
            label(video, 'Raw video')
            panels = [video]
            for key, title, color, angles in [('reference', 'Ground truth: mocap markers', (50, 50, 50), reference_knees),
                                               ('torch', 'Torch prediction: 5,000 steps', (40, 150, 40), prediction_knees)]:
                panel = skeleton_panel(canonical[key][i], title, color, scale, center)
                label(panel, f'Knee R {angles[i, 0]:.1f} / L {angles[i, 1]:.1f} deg', 1, color, outline=False)
                panels.append(panel)
            output = np.vstack((np.hstack(panels), np.full((96, 1080, 3), 255, np.uint8)))
            footer = [f'Synchronized time: {timestamp-timestamps[0]:.2f} s | Raw video / Ground truth / Torch 5k',
                      'Skeletons: pelvis centered; initial heading aligned for display. Same scale.',
                      'Footage: lens-corrected input, no detection overlays. Ground truth: measured mocap markers and hip centers.',
                      'Knee angles: marker-based IK versus fitted joint angles.']
            for row, text in enumerate(footer):
                cv2.putText(output, text, (12, 660+row*22), cv2.FONT_HERSHEY_SIMPLEX, .48, (40, 40, 40), 1, cv2.LINE_AA)
            yield output
    finally:
        cap.release()


def comparison_video(directory, mocap_trc, detections=False):
    """Show the input, measured mocap marker skeleton, and both predicted skeletons."""
    directory = Path(directory)
    with np.load(directory / 'trajectories.npz') as arrays:
        data = {k: arrays[k] for k in arrays.files}
    with np.load(directory / 'torch_keypoints.npz') as arrays:
        detected = arrays['keypoints2d'].copy(), arrays['confs'].copy()
    time = data['reference_time']
    reference = reference_skeleton(mocap_trc, time)
    if detections:
        prediction = {}
        for key, filename in [('original', 'tensorflow'), ('torch', 'torch')]:
            with np.load(directory / f'{filename}_keypoints.npz') as arrays:
                prediction[key] = arrays['keypoints3d'] / 1000
                prediction[key][arrays['confs'] <= 0] = np.nan
        suffix, description = 'detection_skeletons', 'detected'
    else:
        prediction = {key: data[key + '_sites_camera_mm'] / 1000 for key in ('original', 'torch')}
        suffix, description = 'comparison', 'fitted'
    canonical, scale, center = skeleton_views(dict(reference=reference, **prediction))
    cap = cv2.VideoCapture(str(directory / 'prepared.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)

    def frames():
        try:
            for i, timestamp in enumerate(time):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError('Prepared video is shorter than the evaluation trajectories.')
                h, w = frame.shape[:2]
                factor = min(360 / w, 640 / h)
                small = cv2.resize(frame, (round(w * factor), round(h * factor)))
                video = np.zeros((640, 360, 3), dtype=np.uint8)
                top, left = (640 - small.shape[0]) // 2, (360 - small.shape[1]) // 2
                video[top:top + small.shape[0], left:left + small.shape[1]] = small
                draw_points(video, detected[0][i] * factor + [left, top], joint_names, (60, 240, 100), detected[1][i])
                label(video, 'Video + Torch detections')
                label(video, f'Reference time: {timestamp:.2f}s', 1)
                panels = [video]
                for key, title, color in [('reference', 'Ground truth: mocap markers', (50, 50, 50)),
                                           ('original', f'Original: {description} skeleton', (220, 100, 35)),
                                           ('torch', f'Torch: {description} skeleton', (40, 150, 40))]:
                    panel = skeleton_panel(canonical[key][i], title, color, scale, center)
                    if not detections:
                        angles = data['reference_knees_deg'][i] if key == 'reference' else data[key + '_knees_deg'][i]
                        label(panel, f'Knee R {angles[0]:.1f} / L {angles[1]:.1f} deg', 1, color, outline=False)
                    panels.append(panel)
                output = np.vstack((np.hstack(panels), np.full((64, 1440, 3), 255, np.uint8)))
                cv2.putText(output, '3D views: pelvis centered; initial pelvis headings aligned for display only. Same spatial scale.',
                            (12, 664), cv2.FONT_HERSHEY_SIMPLEX, .58, (40, 40, 40), 1, cv2.LINE_AA)
                cv2.putText(output, 'Ground truth uses measured markers / hip centers. Numerical knee angles come from marker-based IK.',
                            (12, 688), cv2.FONT_HERSHEY_SIMPLEX, .55, (40, 40, 40), 1, cv2.LINE_AA)
                yield output
        finally:
            cap.release()
    path = directory / f'{suffix}.mp4'
    write_video(path, frames(), fps)
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mocap-trc', type=Path)
    parser.add_argument('--detections-3d', action='store_true')
    args = parser.parse_args()
    print(comparison_video(args.directory, args.mocap_trc, args.detections_3d)
          if args.mocap_trc else detection_video(args.directory))
