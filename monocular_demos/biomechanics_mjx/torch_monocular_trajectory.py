"""PyTorch fitting for new monocular videos using differentiable kinematics."""
import math
from typing import NamedTuple

import mujoco
import numpy as np
import torch
from torch import nn
from tqdm import trange

from monocular_demos.torch_camera import (
    get_extrinsic, get_extrinsic_dynamic, project_distortion, project_dynamic,
    rotation_matrix, rotation_quaternion, transform_points,
)
from monocular_demos.torch_implicit_representation import ImplicitTrajectory, CubicSplineTrajectory, calculate_encoding_length
from .native_model import ForwardKinematics, get_dof_qpos_jntid, movi_joint_names
from .torch_forward_kinematics import TorchForwardKinematics, _quat_mul


class State(NamedTuple):
    qpos: torch.Tensor
    qvel: torch.Tensor
    xpos: torch.Tensor
    site_xpos: torch.Tensor
    efc_force: None = None


def normalize_velocity_from_quaternions(pose, derivative, freeroot):
    if not freeroot:
        return derivative
    quaternion = torch.nn.functional.normalize(pose[..., 3:7], dim=-1)
    conjugate = quaternion * quaternion.new_tensor([1, -1, -1, -1])
    omega = 2 * _quat_mul(derivative[..., 3:7], conjugate)[..., 1:]
    return torch.cat((derivative[..., :3], omega, derivative[..., 7:]), -1)


class KineticsWrapper(nn.Module):
    def __init__(self, dataset, xml_path=None, joint_names=None, scale_mappings=None,
                 trajectory_type='implicit', encoding_length=None, cubic_sequence_length=None,
                 min_freq=80, nextra=None, key=0, dtype=None, device=None, **kwargs):
        super().__init__()
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = dtype or (torch.float32 if torch.device(device).type == 'mps' else torch.float64)
        if torch.device(device).type == 'mps' and dtype != torch.float32:
            raise ValueError('MPS fitting requires dtype=torch.float32.')
        metadata = ForwardKinematics(xml_path)
        self.xml_path = metadata.xml_path
        m = metadata.model
        self.njoints, self.nvjoints, self.nactions = m.nq, m.nv, 0
        self.freeroot = m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.nextra = (3 if dataset.phone_attitude is not None else 0) if nextra is None else nextra
        if dataset.phone_attitude is not None and self.nextra != 3:
            raise ValueError('Phone-attitude fitting requires three camera rotation outputs.')
        names = movi_joint_names if joint_names is None else joint_names
        order = [metadata.site_names.index(name) for name in names]
        self.kinematics = TorchForwardKinematics(m, order, dtype=dtype)
        mixer = metadata.build_default_scale_mixer() if scale_mappings is None else metadata.build_custom_scale_mixer(scale_mappings)
        self.register_buffer('scale_mixer', torch.tensor(mixer, dtype=dtype))
        self.body_scale = nn.Parameter(torch.zeros(mixer.shape[1], 1, dtype=dtype))
        self.site_offsets = nn.Parameter(torch.zeros(m.nsite, 3, dtype=dtype))
        joint_ids = get_dof_qpos_jntid(m)
        self.register_buffer('lower_limit', torch.tensor(np.r_[m.jnt_range[joint_ids, 0], -np.ones(self.nextra)], dtype=dtype))
        self.register_buffer('upper_limit', torch.tensor(np.r_[m.jnt_range[joint_ids, 1], np.ones(self.nextra)], dtype=dtype))
        self.register_buffer('apply_limits', torch.tensor(np.r_[m.jnt_limited[joint_ids], np.zeros(self.nextra)] >= 1))
        self.trajectory = nn.ModuleList()
        max_time = float(dataset.max_time) + 1
        for i in range(len(dataset)):
            dims = self.njoints + self.nextra
            if trajectory_type == 'implicit':
                options = dict(kwargs)
                if str(device).startswith('cuda'):
                    options.setdefault('jmp_policy', 'params=float32,compute=float16,output=float32')
                if self.freeroot or dataset.phone_attitude is not None:
                    bias = np.full(dims, np.nan)
                    if self.freeroot:
                        bias[3:5] = 0.7071
                    if dataset.phone_attitude is not None:
                        rvec = np.median(dataset.get_all_attitude(i)[1], axis=0)
                        bias[-3:] = rvec
                        direction = rotation_matrix(torch.tensor(rvec)).numpy()[:, 2].copy()
                        direction[2] = 0
                        direction /= np.linalg.norm(direction)
                        bias[:3] = [direction[0] * 1.5, direction[1] * (1.5 if self.freeroot else -1.5), -1.5]
                    options['custom_bias'] = bias
                trajectory = ImplicitTrajectory(joints=dims, spatial_dims=1,
                    encoding_length=encoding_length if encoding_length is not None else calculate_encoding_length(max_time, min_freq),
                    max_time=max_time, key=int(key) + i, **options)
            elif trajectory_type == 'cubic':
                trajectory = CubicSplineTrajectory(cubic_sequence_length or int(max_time * min_freq), dims, 1, max_time, key=int(key) + i)
            else:
                raise ValueError(f'Unknown trajectory type {trajectory_type!r}.')
            self.trajectory.append(trajectory)
        self.to(device=device)

    def _pose(self, time, trajectory_selection):
        x = self.trajectory[int(trajectory_selection)](time)[..., 0].to(self.body_scale)
        half_range = ((self.upper_limit - self.lower_limit) / 2).clamp_min(0.01)
        middle = (self.upper_limit + self.lower_limit) / 2
        return torch.where(self.apply_limits, x.tanh() * half_range + middle, x)

    def forward(self, timepoint, trajectory_selection=0, trajectory_only=False, skip_action=True,
                skip_vel=False, steps=0, dt=1/30, fast_inference=False, freeze_scale=False,
                margin=None, action_repeats=0, check_constraints=False):
        if not skip_action or steps or check_constraints:
            raise ValueError('Torch video fitting supports kinematics with skip_action=True, steps=0, check_constraints=False.')
        time = torch.as_tensor(timepoint, dtype=self.body_scale.dtype, device=self.body_scale.device)
        output = self._pose(time, trajectory_selection)
        pose, extra = output[..., :self.njoints], output[..., self.njoints:]
        if skip_vel:
            velocity = pose.new_zeros((*pose.shape[:-1], self.nvjoints))
        else:
            derivative = torch.func.jvp(lambda t: self._pose(t, trajectory_selection), (time,), (torch.ones_like(time),))[1]
            velocity = normalize_velocity_from_quaternions(pose, derivative[..., :self.njoints], self.freeroot)
        action = pose.new_empty((*pose.shape[:-1], 0))
        if trajectory_only:
            return pose, velocity, action, extra
        scale, offsets = 1 + self.scale_mixer @ self.body_scale, self.site_offsets
        if freeze_scale:
            scale, offsets = scale.detach(), offsets.detach()
        positions = self.kinematics(pose, scale, offsets)
        state = State(pose, velocity, positions.xpos, positions.site_xpos)
        return (state, None, None), (pose, velocity, action), extra


def get_model(dataset, xml_path=None, joint_names=None, scale_mappings=None, **kwargs):
    return KineticsWrapper(dataset, xml_path, joint_names, scale_mappings, **kwargs)


def huber(x, delta=5.0):
    return torch.where(x.abs() < delta, 0.5 * x.square(), delta * (x.abs() - 0.5 * delta))


def loss(model, x, y, lambda_reprojection, camera_calibration, site_offset_regularization=1e2,
         static=False, center_3d=True, quat_regularization=100.0, ang_pred_regularization=0.0,
         ang_pred_vel_scale=10.0, dt=1/30):
    if np.ndim(x[0]) == 1:
        results = [loss(model, tuple(None if v is None else v[i] for v in x),
                        tuple(None if v is None else v[i] for v in y), lambda_reprojection,
                        camera_calibration, site_offset_regularization, static, center_3d,
                        quat_regularization, ang_pred_regularization, ang_pred_vel_scale, dt) for i in range(len(x[0]))]
        return sum(result[0] for result in results), {k: torch.stack([r[1][k] for r in results]) for k in results[0][1]}
    trajectory, video_times, attitude_times = x
    kp2d, kp3d, confidence, attitude = (None if v is None else torch.as_tensor(v).to(model.body_scale) for v in y)
    confidence = confidence[0]
    (state, _, _), (angles, velocity, _), rvec = model(video_times, trajectory_selection=trajectory, skip_vel=ang_pred_regularization <= 0)
    points = state.site_xpos * 1000
    attitude_loss = points.new_zeros(())
    if static:
        camera_points = transform_points(points, get_extrinsic(camera_calibration, 0))
        projection = project_distortion(camera_calibration, 0, points) if lambda_reprojection > 0 else None
    else:
        camera_points = transform_points(points, get_extrinsic_dynamic(camera_calibration, 0, rvec))
        projection = project_dynamic(points, camera_calibration, rvec) if lambda_reprojection > 0 else None
        predicted_attitude = model(attitude_times, trajectory_selection=trajectory, trajectory_only=True, skip_vel=True)[3]
        inverse = rotation_quaternion(attitude) * attitude.new_tensor([1, -1, -1, -1])
        difference = _quat_mul(inverse, rotation_quaternion(predicted_attitude))
        attitude_loss = (2 * torch.atan2(torch.linalg.vector_norm(difference[..., 1:], dim=-1), difference[..., 0].abs())).mean() * 180 / torch.pi
    observed = kp3d[0]
    if center_3d:
        camera_points = camera_points - camera_points.mean(dim=1, keepdim=True)
        observed = observed - observed.mean(dim=1, keepdim=True)
    norm = confidence.sum() + 1e-7
    error3d = huber(torch.linalg.vector_norm(camera_points / 10 - observed / 10, dim=-1) * confidence, 10).sum() / norm
    error2d = points.new_zeros(()) if projection is None else huber(torch.linalg.vector_norm(projection - kp2d[0], dim=-1) * confidence, 50).nansum() / norm
    offset_loss = model.site_offsets.square().sum()
    metrics = dict(mean_3d_raw=error3d, site_offset_raw=offset_loss, mean_reprojection_raw=error2d, attitude_loss_raw=attitude_loss)
    metrics.update(mean_3d_scaled=error3d, site_offset_scaled=offset_loss * site_offset_regularization,
                   mean_reprojection_scaled=error2d * lambda_reprojection, attitude_loss_scaled=attitude_loss)
    total = error3d + offset_loss * site_offset_regularization + error2d * lambda_reprojection + attitude_loss
    if model.freeroot:
        metrics['quat_error'] = (1 - torch.linalg.vector_norm(angles[..., 3:7], dim=-1)).square().mean()
        total = total + metrics['quat_error'] * quat_regularization
    if ang_pred_regularization > 0:
        future = model(torch.as_tensor(video_times).to(angles) + dt, trajectory_selection=trajectory, trajectory_only=True, skip_vel=True)[0].detach()
        if model.freeroot:
            qdot = 0.5 * _quat_mul(angles[..., 3:7], torch.cat((torch.zeros_like(velocity[..., :1]), velocity[..., 3:6]), -1))
            velocity = torch.cat((velocity[..., :3], qdot, velocity[..., 6:]), -1)
        error = angles + velocity * dt - future
        error = torch.cat((error[..., :3] * ang_pred_vel_scale, error[..., 3:]), -1)
        metrics['ang_pred'] = error.square().mean()
        total = total + metrics['ang_pred'] * ang_pred_regularization
    return total, {'total': total, **metrics}


def fit_model(model, dataset, lr_end_value=1e-7, lr_init_value=1e-4, warmup_steps=0,
              transition_begin=0, max_iters=25000, site_offset_regularization=1e2,
              lambda_reprojection=0.1, clip_by_global_norm=0.1, optimizer_zero_nans=False,
              center_3d=True, step_callback=None, **kwargs):
    if max_iters < 1 or not 0 <= warmup_steps < max_iters or lr_init_value <= 0 or lr_end_value <= 0:
        raise ValueError('Iterations and learning rates must be positive, with warmup shorter than training.')
    if dataset.keypoint_confidence.sum() <= 0:
        raise ValueError('No confident keypoints were detected in the videos.')
    static = kwargs.pop('static', dataset.phone_attitude is None)
    kwargs.setdefault('dt', 1/29 if static else 1/30)
    decay_steps = max(1, (max_iters - warmup_steps) // 10) * 10
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init_value, betas=(0.8, 0.999), eps=1e-8, weight_decay=1e-5)
    counter = trange(max_iters)
    for i in counter:
        if i < warmup_steps:
            rate = lr_init_value * i / warmup_steps
        else:
            fraction = max(0, i - warmup_steps - transition_begin) / decay_steps
            rate = max(lr_end_value, lr_init_value * (lr_end_value / lr_init_value) ** fraction)
        optimizer.param_groups[0]['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        value, metrics = loss(model, *dataset[np.arange(len(dataset)) + i * len(dataset)],
                              lambda_reprojection=lambda_reprojection, camera_calibration=dataset.camera_params,
                              site_offset_regularization=site_offset_regularization, static=static, center_3d=center_3d, **kwargs)
        if not torch.isfinite(value):
            raise ValueError(f'Non-finite fitting loss on iteration {i}.')
        value.backward()
        # Optax clips the AdamW updates, including weight decay, after the optimizer.
        before = [p.detach().clone() for p in model.parameters()] if clip_by_global_norm > 0 or optimizer_zero_nans else None
        optimizer.step()
        if before is not None:
            with torch.no_grad():
                updates = [p - old for p, old in zip(model.parameters(), before)]
                if optimizer_zero_nans:
                    updates = [torch.where(u.isnan(), 0, u) for u in updates]
                factor = 1.0
                if clip_by_global_norm > 0:
                    norm = torch.stack([u.square().sum().to(model.body_scale) for u in updates]).sum().sqrt()
                    factor = clip_by_global_norm / norm.clamp_min(clip_by_global_norm)
                for parameter, old, update in zip(model.parameters(), before, updates):
                    parameter.copy_(old + factor * update)
        metrics = {key: item.detach() for key, item in metrics.items()}
        if step_callback is not None:
            step_callback(i, model, dataset, {**metrics, 'learning_rate': rate})
        if i % 50 == 0:
            counter.set_postfix(total=metrics['total'].mean().item())
    return model, metrics
