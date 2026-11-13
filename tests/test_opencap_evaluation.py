import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.evaluate_opencap import read_mot, score_knees, knee_flexion, prepare_video, validate_reference_model
from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics


def write_mot(path, units='yes'):
    path.write_text(f'test\nnRows=5\nnColumns=3\ninDegrees={units}\nendheader\ntime\tknee_angle_r\tknee_angle_l\n'
                    + '\n'.join(f'{i / 10}\t{i * 10}\t{i * 20}' for i in range(5)))


def test_mot_units_and_sign_mapping(tmp_path):
    path = tmp_path / 'motion.mot'
    write_mot(path)
    time, angles = read_mot(path)
    model = ForwardKinematics().model
    qpos = np.repeat(model.qpos0[None], len(time), axis=0)
    for j, name in enumerate(('knee_angle_r', 'knee_angle_l')):
        qpos[:, model.joint(name).qposadr[0]] = -np.deg2rad(angles[:, j])
    np.testing.assert_allclose(knee_flexion(qpos, model), angles)
    write_mot(path, units='no')
    _, radians_converted = read_mot(path)
    np.testing.assert_allclose(radians_converted, np.rad2deg(angles))
    write_mot(path, units='unknown')
    with pytest.raises(ValueError, match='inDegrees'):
        read_mot(path)


def test_known_time_offset_and_no_extrapolation():
    reference_time = np.linspace(10, 12, 201)
    reference = np.column_stack((reference_time * 20, reference_time * 30))
    video_time = np.arange(-0.1, 2.11, 0.1)
    clock = video_time + 10
    prediction = np.column_stack((clock * 20 + 3, clock * 30 - 4))
    rows, interpolated, overlap = score_knees(clock, prediction, reference_time, reference)
    assert rows[0]['mae_deg'] == pytest.approx(3)
    assert rows[1]['mae_deg'] == pytest.approx(4)
    assert rows[0]['bias_deg'] == pytest.approx(3)
    assert rows[1]['bias_deg'] == pytest.approx(-4)
    assert rows[0]['correlation'] == pytest.approx(1)
    assert np.isnan(interpolated[~overlap]).all()
    assert rows[0]['samples'] == overlap.sum()
    with pytest.raises(ValueError, match='Insufficient'):
        score_knees(clock + 50, prediction, reference_time, reference)


def test_missing_reference_samples_are_excluded():
    time = np.arange(6.)
    reference = np.column_stack((time, time))
    reference[2, 0] = np.nan
    rows, _, _ = score_knees(time, np.zeros_like(reference), time, reference)
    assert rows[0]['samples'] == 5
    assert rows[1]['samples'] == 6


def test_explicit_reference_model_convention(tmp_path):
    path = tmp_path / 'reference.osim'
    coordinates = ''.join(f'<Coordinate name="knee_angle_{side}"><range>0 2.4</range><default_value>0</default_value></Coordinate>'
                          for side in ('r', 'l'))
    path.write_text(f'<OpenSimDocument><Model>{coordinates}</Model></OpenSimDocument>')
    validate_reference_model(path)
    path.write_text(path.read_text().replace('0 2.4', '-2.4 0'))
    with pytest.raises(ValueError, match='convention'):
        validate_reference_model(path)


def test_clip_preparation_preserves_source_clock_and_intrinsics(tmp_path):
    video = tmp_path / 'input.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (64, 48))
    for i in range(60):
        writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
    writer.release()
    calibration_path = tmp_path / 'camera.pickle'
    intrinsic = np.array([[60, 0, 32], [0, 60, 24], [0, 0, 1]])
    with open(calibration_path, 'wb') as handle:
        pickle.dump(dict(intrinsicMat=intrinsic, distortion=np.zeros((1, 5)), imageSize=(48, 64)), handle)
    times, calibration, actual_intrinsic = prepare_video(video, calibration_path, tmp_path/'prepared.mp4', .5, 1, 10)
    np.testing.assert_allclose(times, np.arange(.5, 1.5, .1))
    np.testing.assert_allclose(actual_intrinsic, intrinsic)
    np.testing.assert_allclose(calibration['mtx'], [[.06, .06, .032, .024]])
    with open(calibration_path, 'wb') as handle:
        pickle.dump(dict(intrinsicMat=intrinsic, distortion=np.zeros(5), imageSize=(48, 100)), handle)
    with pytest.raises(ValueError, match='imageSize'):
        prepare_video(video, calibration_path, tmp_path/'wrong.mp4', 0, 1, 10)


def test_malformed_reference_clock(tmp_path):
    path = tmp_path/'bad.mot'
    write_mot(path)
    path.write_text(path.read_text().replace('0.1\t10\t20', '0.0\t10\t20'))
    with pytest.raises(ValueError, match='strictly increasing'):
        read_mot(path)


def test_selective_archive_extraction_and_existing_file_protection(tmp_path):
    import zipfile
    from analysis.extract_opencap_pilot import SUBSET_FILES, extract_subset
    archive = tmp_path/'data.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        for name in SUBSET_FILES:
            handle.writestr(name, b'test data')
        handle.writestr('unrelated/video.avi', b'unselected')
    destination = tmp_path/'subset'
    paths = extract_subset(archive, destination)
    assert len(paths) == 4
    assert all(path.read_bytes() == b'test data' for path in paths)
    assert not (destination/'unrelated').exists()
    extract_subset(archive, destination)
    paths[0].write_bytes(b'different')
    with pytest.raises(FileExistsError):
        extract_subset(archive, destination)


def test_five_second_clip_has_no_extra_frame_from_float_endpoint(tmp_path):
    video = tmp_path/'rounding.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 60, (64, 48))
    for _ in range(600):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    camera = tmp_path/'camera.pickle'
    with camera.open('wb') as handle:
        pickle.dump(dict(intrinsicMat=np.array([[60,0,32],[0,60,24],[0,0,1]]),
                         distortion=np.zeros(5),imageSize=(48,64)),handle)
    start = 265 / 60
    assert (start + 5) * 60 > 565  # The old ceil/arange path produced 76 frames.
    output = tmp_path/'prepared.mp4'
    times, _, _ = prepare_video(video,camera,output,start,5,15)
    assert len(times) == 75
    np.testing.assert_allclose(times,start+np.arange(75)/15)
    cap = cv2.VideoCapture(str(output))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 75
    cap.release()


def test_reference_gpu_abort_is_recorded_without_retry(monkeypatch, tmp_path):
    import signal
    import subprocess
    from analysis.evaluate_opencap import detect_reference
    calls = []
    def gpu_abort(command, check):
        calls.append(command)
        raise subprocess.CalledProcessError(-signal.SIGABRT, command)
    monkeypatch.setattr(subprocess, 'run', gpu_abort)
    output = tmp_path/'poses.npz'
    with pytest.raises(subprocess.CalledProcessError):
        detect_reference('/python', 'video.mp4', output, 'camera.npz')
    assert len(calls) == 1
    assert output.with_suffix('.error.json').exists()
    def ordinary_failure(command, check):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, 'run', ordinary_failure)
    with pytest.raises(subprocess.CalledProcessError):
        detect_reference('/python', 'video.mp4', output, 'camera.npz')
    assert not output.with_suffix('.error.json').exists()
