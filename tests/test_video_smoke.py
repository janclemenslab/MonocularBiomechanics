"""Bounded real-video check; downloads pretrained detection weights on first use."""
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl' if sys.platform == 'linux' else 'glfw')
if sys.platform == 'linux':
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import numpy as np
import torch

from analysis.opencap_videos import draw_points, write_video
from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics, scale_model
from monocular_demos.biomechanics_mjx.torch_monocular_trajectory import get_model, fit_model
from monocular_demos.metrabs_torch import load_metrabs
from monocular_demos.torch_dataset import MonocularDataset, get_samsung_calibration
from monocular_demos.utils import joint_names


def test_short_video_detection_fitting_and_rendering(tmp_path):
    torch.set_num_threads(2)
    cap = cv2.VideoCapture(str(Path(__file__).resolve().parents[1]/'data/running_sagittal.mp4'))
    frames = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        for _ in range(4):
            ok, frame = cap.read()
            assert ok, 'Bundled video must contain four decodable frames'
            frames.append(cv2.resize(frame, (480, 270)))
    finally:
        cap.release()
    assert fps > 0
    write_video(tmp_path/'input.mp4', frames, fps)
    calibration = get_samsung_calibration()
    focal = 480/(2*np.tan(np.deg2rad(55)/2))
    intrinsic = np.array([[focal, 0, 240], [0, focal, 135], [0, 0, 1]], dtype=np.float32)
    calibration['mtx'] = intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]][None]/1000
    start = time.perf_counter()
    detector = load_metrabs(device='cpu')
    detected = detector.detect_poses_batched(
        np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]),
        intrinsic_matrix=torch.tensor(intrinsic)[None], num_aug=1)
    assert all(len(boxes) > 0 for boxes in detected['boxes']), 'A person must be detected in every frame'
    kp2d = np.stack([p[0].numpy() for p in detected['poses2d']])
    kp3d = np.stack([p[0].numpy() for p in detected['poses3d']])
    assert kp2d.shape == (4, 87, 2) and kp3d.shape == (4, 87, 3)
    assert np.isfinite(kp2d).all() and np.isfinite(kp3d).all()
    detection_seconds = time.perf_counter()-start
    annotated = [draw_points(frame.copy(), points, joint_names, (60, 240, 100)) for frame, points in zip(frames, kp2d)]
    write_video(tmp_path/'detections.mp4', annotated, fps)
    times = np.arange(4)/fps
    dataset = MonocularDataset(timestamps=[times], keypoints_2d=[kp2d[None]],
        keypoints_3d=[kp3d[None]], keypoint_confidence=[np.ones((1, 4, 87))],
        camera_params=calibration, phone_attitude=None)
    model = get_model(dataset, joint_names=joint_names, device='cpu', features=(16, 32), concatenate_layers=1)
    metadata = ForwardKinematics()
    # Put the tiny test fit at the observed pelvis; twenty steps are not a convergence study.
    with torch.no_grad():
        pelvis = joint_names.index('mhip')
        delta = torch.tensor(kp3d[:, pelvis].mean(0)/1000)-model(times, skip_vel=True)[0][0].site_xpos[:, pelvis].mean(0)
        for name, shift in [('pelvis_tx', delta[0]), ('pelvis_tz', -delta[1]), ('pelvis_ty', delta[2])]:
            model.trajectory[0].final.bias[int(metadata.model.joint(name).qposadr[0])].add_(shift)
    start = time.perf_counter()
    model, metrics = fit_model(model, dataset, max_iters=20, lr_init_value=.01)
    assert torch.isfinite(metrics['total']).all()
    fitting_seconds = time.perf_counter()-start
    with torch.no_grad():
        qpos = model(times, skip_vel=True)[1][0].numpy()
    assert qpos.shape == (4, 40) and np.isfinite(qpos).all()
    start = time.perf_counter()
    from monocular_demos.biomechanics_mjx.visualize import get_overlay_monocular, render_trajectory, video_overlay
    scaled = scale_model(metadata.model, 1+metadata.build_default_scale_mixer()@model.body_scale.detach().numpy())
    overlay = get_overlay_monocular(calibration, np.zeros((4, 3)), scaled, qpos,
        np.arange(4), np.ones(4, dtype=bool), width=480, height=270)
    from OpenGL.GL import glGetString, GL_RENDERER, GL_VERSION
    graphics = {name: glGetString(key).decode() for name, key in [('renderer', GL_RENDERER), ('version', GL_VERSION)]}
    rendered = [overlay(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), i) for i, frame in enumerate(frames)]
    changed = [int(np.any(image != cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), axis=-1).sum()) for image, frame in zip(rendered, frames)]
    assert min(changed) > 10, 'The fitted mesh must be visible in every overlay frame'
    video_overlay(str(tmp_path/'input.mp4'), str(tmp_path/'overlay.mp4'), overlay, downsample=1, max_frames=4)
    cv2.imwrite(str(tmp_path/'preview.jpg'), np.hstack((frames[0], annotated[0], cv2.cvtColor(rendered[0], cv2.COLOR_RGB2BGR))))
    images = render_trajectory(qpos, body_scale=model.body_scale.detach().numpy(), width=320, height=180)
    assert len(images) == 4 and all(image.std() > 5 for image in images)
    write_video(tmp_path/'fitted_motion.mp4', [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images], fps)
    for name, size in [('input', (480, 270)), ('detections', (480, 270)), ('fitted_motion', (320, 180)), ('overlay', (480, 270))]:
        video = cv2.VideoCapture(str(tmp_path/f'{name}.mp4'))
        try:
            decoded = []
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                decoded.append(frame)
            assert len(decoded) == 4 and all(frame.shape[:2] == size[::-1] for frame in decoded)
            assert abs(video.get(cv2.CAP_PROP_FPS)-fps) < 1e-3
        finally:
            video.release()
    report = dict(frames=4, detection_seconds=detection_seconds, fitting_seconds=fitting_seconds,
        rendering_seconds=time.perf_counter()-start, fit_steps=20, overlay_changed_pixels=changed, graphics=graphics)
    (tmp_path/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
