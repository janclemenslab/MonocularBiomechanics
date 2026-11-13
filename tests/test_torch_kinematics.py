"""Prototype parity checks. Run with pytest -q -s tests/test_torch_kinematics.py."""

from copy import deepcopy
from pathlib import Path

import mujoco
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from monocular_demos.biomechanics_mjx.torch_forward_kinematics import TorchForwardKinematics


JOINT_TEST_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <site pos="0.1 0.2 0.3"/>
    <body pos="0.2 -0.3 0.8" quat="0.9 0.1 0.2 0.3">
      <freejoint/>
      <geom size="0.1"/>
      <site pos="0.12 0.15 -0.13"/>
      <body pos="0.2 -0.1 0.4" euler="0.1 0.2 0.3">
        <joint type="ball" pos="0.03 -0.04 0.02"/>
        <geom size="0.1"/>
        <site pos="0.2 0.3 0.1"/>
        <body pos="0.1 0.2 0.3" euler="-0.3 0.1 0.2">
          <joint type="slide" axis="1 2 3" ref="0.12"/>
          <joint type="hinge" axis="2 1 -1" pos="0.04 0.05 -0.02" ref="0.31"/>
          <joint type="hinge" axis="1 -1 2" pos="-0.02 0.01 0.03" ref="-0.2"/>
          <geom size="0.1"/>
          <site pos="-0.1 0.2 0.3"/>
          <body pos="0.2 0.1 -0.1"><site pos="0.3 -0.1 0.2"/></body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module", params=["humanoid", "joint_types"])
def model(request):
    if request.param == "humanoid":
        path = Path(__file__).resolve().parents[1] / "monocular_demos/biomechanics_mjx/data/humanoid/humanoid_torque.xml"
        return mujoco.MjModel.from_xml_path(str(path))
    return mujoco.MjModel.from_xml_string(JOINT_TEST_XML)


def inputs(model, seed=42):
    rng = np.random.default_rng(seed)
    # Keep quaternion inputs non-unit to exercise normalization and its gradients.
    pose = model.qpos0 + rng.normal(0, 0.15, model.nq)
    scale = rng.uniform(0.8, 1.2, (model.nbody, 1))
    offsets = rng.normal(0, 0.01, (model.nsite, 3))
    return tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (pose, scale, offsets))


def native_positions(model, pose, scale, offsets):
    scaled = deepcopy(model)
    scale = np.broadcast_to(scale, (model.nbody, 1))
    scaled.body_pos[:] *= scale[model.body_parentid]
    scaled.site_pos[:] = model.site_pos * scale[model.site_bodyid] + offsets
    # Compiled frame shortcuts become stale when site coordinates are edited.
    scaled.site_sameframe[:] = 0
    data = mujoco.MjData(scaled)
    data.qpos[:] = pose
    mujoco.mj_kinematics(scaled, data)
    return data.xpos.copy(), data.site_xpos.copy()


@pytest.mark.parametrize("seed", [0, 42, 123])
def test_native_positions_and_batching(model, seed):
    fk = TorchForwardKinematics(model)
    pose, scale, offsets = inputs(model, seed)
    expected = native_positions(model, *(v.detach().numpy() for v in (pose, scale, offsets)))
    actual = fk(pose, scale, offsets)
    error = max(np.max(np.abs(a.detach().numpy() - b)) for a, b in zip(actual, expected))
    print(f"native position max error, nq={model.nq}, seed={seed}: {error:.3e} m")
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.detach().numpy(), b, atol=2e-12, rtol=2e-12)
    poses = torch.stack((pose, pose + 0.03, pose - 0.07)).reshape(1, 3, model.nq)
    batched = fk(poses, scale, offsets)
    for i in range(3):
        reference = native_positions(model, poses[0, i].detach().numpy(), scale.detach().numpy(), offsets.detach().numpy())
        for a, b in zip(batched, reference):
            np.testing.assert_allclose(a[0, i].detach().numpy(), b, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize('parallel', [False, True])
def test_rest_pose_scalar_scale_and_site_order(model, parallel):
    order = [model.nsite - 1, 0, model.nsite - 1]
    fk = TorchForwardKinematics(model, order, parallel=parallel)
    pose = torch.tensor(model.qpos0)
    for scale in (1.0, 1.17):
        expected = native_positions(model, model.qpos0, scale, np.zeros((model.nsite, 3)))
        actual = fk(pose, scale)
        np.testing.assert_allclose(actual.xpos.numpy(), expected[0], atol=2e-12)
        np.testing.assert_allclose(actual.site_xpos.numpy(), expected[1][order], atol=2e-12)


def test_autograd_against_native_finite_differences(model):
    fk = TorchForwardKinematics(model)
    values = inputs(model)
    rng = np.random.default_rng(123)
    weights = (rng.normal(size=(model.nbody, 3)), rng.normal(size=(model.nsite, 3)))
    loss = sum((position * torch.tensor(weight)).sum() for position, weight in zip(fk(*values), weights))
    gradients = torch.autograd.grad(loss, values)
    arrays = [v.detach().numpy().copy() for v in values]
    epsilon = 1e-6
    for name, array, gradient in zip(("pose", "scale", "offsets"), arrays, gradients):
        numerical = np.empty_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + epsilon
            positive = sum((p * w).sum() for p, w in zip(native_positions(model, *arrays), weights))
            array[index] = original - epsilon
            negative = sum((p * w).sum() for p, w in zip(native_positions(model, *arrays), weights))
            array[index] = original
            numerical[index] = (positive - negative) / (2 * epsilon)
        error = np.max(np.abs(gradient.numpy() - numerical))
        print(f"native gradient max error, nq={model.nq}, {name}: {error:.3e}")
        np.testing.assert_allclose(gradient.numpy(), numerical, atol=5e-8, rtol=5e-7)
    assert torch.autograd.gradcheck(fk, values, fast_mode=True, atol=1e-6, rtol=1e-4)


@pytest.fixture(scope="module")
def jax_reference(model):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    from mujoco import mjx
    from monocular_demos.biomechanics_mjx.forward_kinematics import offset_sites, scale_model

    reference = mjx.put_model(model)
    data = mjx.make_data(reference)

    @jax.jit
    def positions(pose, scale, offsets):
        scaled = offset_sites(scale_model(reference, scale), offsets)
        result = mjx.kinematics(scaled, data.replace(qpos=pose))
        return result.xpos, result.site_xpos

    return jax, positions


@pytest.mark.parametrize("seed", [7, 19])
@pytest.mark.parametrize('parallel', [False, True])
def test_jax_positions_and_gradients(model, jax_reference, seed, parallel):
    jax, positions = jax_reference
    import jax.numpy as jnp

    fk = TorchForwardKinematics(model, parallel=parallel)
    values = inputs(model, seed)
    arrays = tuple(v.detach().numpy() for v in values)
    actual = fk(*values)
    expected = positions(*arrays)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a.detach().numpy(), np.asarray(b), atol=2e-12, rtol=2e-12)
    rng = np.random.default_rng(seed)
    weights = tuple(rng.normal(size=p.shape) for p in actual)

    def objective(*args):
        return sum(jnp.sum(p * w) for p, w in zip(positions(*args), weights))

    expected_gradients = jax.jit(jax.grad(objective, argnums=(0, 1, 2)))(*arrays)
    objective_torch = sum((p * torch.tensor(w)).sum() for p, w in zip(actual, weights))
    actual_gradients = torch.autograd.grad(objective_torch, values)
    for name, a, b in zip(("pose", "scale", "offsets"), actual_gradients, expected_gradients):
        error = np.max(np.abs(a.numpy() - np.asarray(b)))
        print(f"JAX gradient max error, nq={model.nq}, seed={seed}, {name}: {error:.3e}")
        np.testing.assert_allclose(a.numpy(), np.asarray(b), atol=2e-11, rtol=2e-11)


def test_synthetic_marker_fitting(model):
    fk = TorchForwardKinematics(model)
    pose, scale, offsets = inputs(model)
    with torch.no_grad():
        target = fk(pose, scale, offsets).site_xpos.clone()
    pose = (pose.detach() + 0.1).requires_grad_()
    scale = torch.ones_like(scale, requires_grad=True)
    offsets = torch.zeros_like(offsets, requires_grad=True)
    optimizer = torch.optim.Adam((pose, scale, offsets), lr=0.01)
    initial = (fk(pose, scale, offsets).site_xpos - target).square().mean().item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = (fk(pose, scale, offsets).site_xpos - target).square().mean()
        loss.backward()
        assert all(torch.isfinite(v.grad).all() for v in (pose, scale, offsets))
        optimizer.step()
    final = (fk(pose, scale, offsets).site_xpos - target).square().mean().item()
    print(f"synthetic fitting, nq={model.nq}: {initial:.3e} -> {final:.3e} m^2")
    assert final < initial * 0.01


def test_float32_positions_and_gradients(model):
    values = inputs(model)
    expected = TorchForwardKinematics(model)(*values)
    float_values = tuple(v.detach().float().requires_grad_() for v in values)
    result = TorchForwardKinematics(model, dtype=torch.float32)(*float_values)
    for a, b in zip(result, expected):
        torch.testing.assert_close(a.double(), b, atol=2e-6, rtol=2e-6)
    gradients = torch.autograd.grad(sum(p.square().sum() for p in result), float_values)
    reference = torch.autograd.grad(sum(p.square().sum() for p in expected), values)
    for a, b in zip(gradients, reference):
        torch.testing.assert_close(a.double(), b, atol=2e-5, rtol=2e-5)


def test_fitting_scale_parameters(model, jax_reference):
    jax, positions = jax_reference
    from monocular_demos.biomechanics_mjx.forward_kinematics import create_custom_scale_mix, create_default_scale_mix

    body_names = [model.body(i).name for i in range(1, model.nbody)]
    if model.nq == 40:
        mixer = np.asarray(create_default_scale_mix(body_names))
    else:
        mixer = np.asarray(create_custom_scale_mix(body_names, {"overall": body_names}))
    coefficients = np.random.default_rng(71).normal(0, 0.03, (mixer.shape[1], 1))
    pose, _, offsets = inputs(model)
    coefficients_torch = torch.tensor(coefficients, requires_grad=True)
    scale = 1 + torch.tensor(mixer) @ coefficients_torch
    result = TorchForwardKinematics(model)(pose, scale, offsets).site_xpos
    gradient = torch.autograd.grad(result.square().sum(), coefficients_torch)[0]

    def objective(c):
        return (positions(pose.detach().numpy(), 1 + mixer @ c, offsets.detach().numpy())[1] ** 2).sum()

    expected = jax.jit(jax.grad(objective))(coefficients)
    np.testing.assert_allclose(gradient.numpy(), np.asarray(expected), atol=2e-11, rtol=2e-11)


def test_native_render_model_scaling_matches_torch(model):
    from monocular_demos.biomechanics_mjx.native_model import scale_model, offset_sites
    pose, scale, offsets = inputs(model)
    scaled = offset_sites(scale_model(model, scale.detach().numpy()), offsets.detach().numpy())
    data = mujoco.MjData(scaled)
    data.qpos[:] = pose.detach().numpy()
    mujoco.mj_kinematics(scaled, data)
    expected = TorchForwardKinematics(model)(pose, scale, offsets)
    np.testing.assert_allclose(data.site_xpos, expected.site_xpos.detach().numpy(), atol=2e-12)


def test_parallel_positions_and_gradients_match_serial_and_mujoco(model):
    pose, scale, offsets = inputs(model)
    poses = torch.stack([pose + i*.01 for i in range(6)]).reshape(2, 3, model.nq).detach().requires_grad_()
    values = poses, scale, offsets
    serial = TorchForwardKinematics(model, parallel=False)(*values)
    parallel = TorchForwardKinematics(model, parallel=True)(*values)
    for actual, expected in zip(parallel, serial):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    native = native_positions(model, poses[0, 0].detach().numpy(), scale.detach().numpy(), offsets.detach().numpy())
    for actual, expected in zip(parallel, native):
        np.testing.assert_allclose(actual[0, 0].detach().numpy(), expected, atol=2e-12, rtol=2e-12)
    a = torch.autograd.grad(sum(x.square().sum() for x in serial), values)
    b = torch.autograd.grad(sum(x.square().sum() for x in parallel), values)
    for actual, expected in zip(b, a):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='Apple GPU required')
def test_mps_parallel_positions_and_gradients(model):
    original = inputs(model)
    cpu = tuple(v.detach().float().requires_grad_() for v in original)
    gpu = tuple(v.detach().to(device='mps', dtype=torch.float32).requires_grad_() for v in original)
    expected = TorchForwardKinematics(model, dtype=torch.float32, parallel=False)(*cpu)
    actual = TorchForwardKinematics(model, dtype=torch.float32).to('mps')(*gpu)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, atol=2e-6, rtol=2e-6)
    cpu_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), cpu)
    gpu_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), gpu)
    for a, b in zip(gpu_grad, cpu_grad):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.cpu(), b, atol=2e-4, rtol=2e-4)
