from copy import deepcopy
import numpy as np
import pytest
import torch

from monocular_demos.torch_dataset import MonocularDataset, get_samsung_calibration
from monocular_demos.torch_camera import project_distortion, rotation_matrix
from monocular_demos.biomechanics_mjx.torch_monocular_trajectory import get_model, fit_model, loss


def make_data(phone=False):
    rng = np.random.default_rng(2)
    times = [np.arange(n) / 30 + 2 for n in (7, 4)]
    kp2d = [rng.normal(500, 10, (1, len(t), 87, 2)) for t in times]
    kp3d = [rng.normal(1000, 100, (1, len(t), 87, 3)) for t in times]
    confidence = [np.ones((1, len(t), 87)) for t in times]
    attitude = [np.column_stack((t, rng.normal(0, .1, (len(t), 3)))) for t in times] if phone else None
    return dict(timestamps=times, keypoints_2d=kp2d, keypoints_3d=kp3d, keypoint_confidence=confidence,
                camera_params=get_samsung_calibration(), phone_attitude=attitude, sample_length=3)


@pytest.mark.parametrize('phone', [False, True])
def test_dataset_matches_reference(phone):
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from monocular_demos.dataset import MonocularDataset as OriginalDataset
    args = make_data(phone)
    original, current = OriginalDataset(**deepcopy(args)), MonocularDataset(**deepcopy(args))
    for index in (0, 1, 2, 5, 11):
        for old_tuple, new_tuple in zip(original[index], current[index]):
            for a, b in zip(old_tuple, new_tuple):
                if a is None:
                    assert b is None
                else:
                    np.testing.assert_allclose(np.asarray(a), b, atol=1e-12)
    assert current[0][0][1].shape == (3,)
    assert current[np.array([0, 1])][1][0].shape == (2, 1, 3, 87, 2)


@pytest.mark.parametrize('phone', [False, True])
@pytest.mark.parametrize('trajectory_type', ['implicit', 'cubic'])
def test_fitting_and_velocities(phone, trajectory_type):
    dataset = MonocularDataset(**make_data(phone))
    options = dict(features=(16, 32), concatenate_layers=1, jmp_policy='params=float64,compute=float64,output=float64', non_linearity='silu')
    if trajectory_type == 'cubic':
        options = dict(cubic_sequence_length=12)
    model = get_model(dataset, trajectory_type=trajectory_type, **options)
    time = dataset.get_all_timestamps(0)
    state, (pose, velocity, action), extra = model(time)
    assert pose.shape == (7, 40) and velocity.shape == (7, 40)
    assert state[0].site_xpos.shape == (7, 87, 3)
    assert extra.shape == (7, 3 if phone else 0)
    h = 1e-5
    plus = model(time + h, trajectory_only=True, skip_vel=True)[0]
    minus = model(time - h, trajectory_only=True, skip_vel=True)[0]
    # Cubic endpoints clamp time, so use interior samples for a central difference.
    torch.testing.assert_close(velocity[1:-1], ((plus - minus) / (2*h))[1:-1], atol=5e-4, rtol=2e-2)
    settings = dict(lambda_reprojection=0.0, camera_calibration=dataset.camera_params, static=not phone)
    initial = loss(model, *dataset[0], **settings)[0].item()
    model, metrics = fit_model(model, dataset, max_iters=20, lr_init_value=0.001, lambda_reprojection=0.0)
    assert torch.isfinite(metrics['total']).all()
    assert loss(model, *dataset[0], **settings)[0].item() < initial


def test_camera_matches_reference_and_has_finite_zero_rotation_gradient():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    from monocular_demos.camera import project_distortion as reference
    calibration = get_samsung_calibration()
    calibration['rvec'] = np.array([[0.1, -0.3, 0.2]])
    calibration['tvec'] = np.array([[0.1, 0.2, 0.3]])
    calibration['dist'] = np.array([[0.04, 0, 0, 0, 0]])
    points = np.random.default_rng(1).normal(size=(3, 10, 3)) * 1000
    np.testing.assert_allclose(project_distortion(calibration, 0, points).numpy(), np.asarray(reference(calibration, 0, points)), atol=1e-9)
    zero = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    assert torch.isfinite(torch.autograd.grad(rotation_matrix(zero).sum(), zero)[0]).all()


def test_invalid_dataset_and_unsupported_dynamics():
    with pytest.raises(ValueError):
        args = make_data()
        args['timestamps'][0] = np.array([])
        MonocularDataset(**args)
    model = get_model(MonocularDataset(**make_data()), features=(8, 8), concatenate_layers=1)
    with pytest.raises(ValueError, match='kinematics'):
        model(torch.tensor([0.]), skip_action=False)


def test_missing_detections_are_masked_and_empty_recording_is_rejected():
    args = make_data()
    for kp2d, kp3d, confidence in zip(args['keypoints_2d'], args['keypoints_3d'], args['keypoint_confidence']):
        kp2d[:] = np.nan
        kp3d[:] = np.nan
        confidence[:] = 0
    dataset = MonocularDataset(**args)
    assert np.isfinite(dataset.keypoints_3d).all()
    model = get_model(dataset, features=(8, 8), concatenate_layers=1)
    with pytest.raises(ValueError, match='confident keypoints'):
        fit_model(model, dataset, max_iters=1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU required')
def test_mps_fitting_loss_and_gradients_match_cpu_float32():
    dataset = MonocularDataset(**make_data())
    options = dict(features=(16, 32), concatenate_layers=1,
                   jmp_policy='params=float32,compute=float32,output=float32')
    cpu = get_model(dataset, dtype=torch.float32, device='cpu', **options)
    gpu = get_model(dataset, device='mps', **options)
    gpu.load_state_dict(cpu.state_dict())
    assert gpu.body_scale.dtype == torch.float32 and gpu.body_scale.device.type == 'mps'
    settings = dict(lambda_reprojection=.1, camera_calibration=dataset.camera_params, static=True)
    before, _ = loss(cpu, *dataset[0], **settings)
    actual, _ = loss(gpu, *dataset[0], **settings)
    torch.testing.assert_close(actual.cpu(), before, rtol=5e-4, atol=1e-3)
    before.backward()
    actual.backward()
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        if a.grad is not None:
            assert b.grad is not None and torch.isfinite(b.grad).all()
            torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=5e-3, atol=5e-3)
    gpu, metrics = fit_model(gpu, dataset, max_iters=3, lr_init_value=.001)
    assert metrics['total'].device.type == 'mps' and torch.isfinite(metrics['total']).all()
    with pytest.raises(ValueError, match='requires dtype'):
        get_model(dataset, device='mps', dtype=torch.float64, **options)
