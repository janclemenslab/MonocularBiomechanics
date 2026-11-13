"""Differentiable position-only prototype for the monocular fitting path."""

from typing import NamedTuple

import mujoco
import torch
from torch import nn
from torch.nn import functional as F


class KinematicPositions(NamedTuple):
    xpos: torch.Tensor
    site_xpos: torch.Tensor


def _quat_mul(a, b):
    """Hamilton product with MuJoCo's scalar-first quaternion convention."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw), dim=-1,
    )


def _rotate(vector, quat):
    vector, imaginary = torch.broadcast_tensors(vector, quat[..., 1:])
    cross = torch.linalg.cross(imaginary, vector)
    return vector + 2 * (quat[..., :1] * cross + torch.linalg.cross(imaginary, cross))


class TorchForwardKinematics(nn.Module):
    """MuJoCo tree kinematics with autograd through pose, scale, and site offsets.

    Pass a compiled MjModel and optional site indices in the desired output order.
    Inputs share the module's dtype/device. qpos can have arbitrary batch axes.
    Quaternion pose entries must be nonzero. Scale is a scalar or (nbody, 1)
    tensor shared across poses, and site_offsets
    is a shared (nsite, 3) tensor in the original model site order. Units are
    metres/radians. Body scaling matches the existing scale_model function.

    This prototype returns body and site positions only. Contact, constraints,
    velocities, and dynamics are outside its scope. Mocap bodies are rejected
    because their poses require inputs beyond qpos.
    Parallel transforms are selected on MPS; pass parallel=False for comparison.
    """

    def __init__(self, model: mujoco.MjModel, site_reorder=None, *, dtype=torch.float64, parallel=None):
        super().__init__()
        if model.nmocap:
            raise ValueError("Mocap bodies require externally supplied poses.")
        self.nq, self.nbody, self.nsite = model.nq, model.nbody, model.nsite
        self.parallel = parallel
        self.parents = tuple(int(i) for i in model.body_parentid)
        self.joints = tuple(
            tuple((int(model.jnt_type[j]), int(model.jnt_qposadr[j]), j)
                  for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]))
            for b in range(model.nbody)
        )
        for name in ("body_pos", "body_quat", "jnt_pos", "jnt_axis", "qpos0", "site_pos"):
            self.register_buffer(name, torch.tensor(getattr(model, name).copy(), dtype=dtype))
        self.register_buffer("site_bodyid", torch.tensor(model.site_bodyid.copy(), dtype=torch.long))
        indices = list(range(model.nsite)) if site_reorder is None else list(site_reorder)
        if any(i < 0 or i >= model.nsite for i in indices):
            raise ValueError("Site indices must be within the model's site range.")
        self.register_buffer("site_reorder", torch.tensor(indices, dtype=torch.long))
        self.register_buffer('_parents', torch.tensor(self.parents), persistent=False)
        self._joint_batches = []
        for slot in range(max(map(len, self.joints), default=0)):
            for kind in range(4):
                selected = [(body, joints[slot]) for body, joints in enumerate(self.joints)
                            if len(joints) > slot and joints[slot][0] == kind]
                if not selected:
                    continue
                bodies = [body for body, _ in selected]
                joints = [joint for _, (_, _, joint) in selected]
                width = {0: 7, 1: 4, 2: 1, 3: 1}[kind]
                addresses = [a + k for _, (_, a, _) in selected for k in range(width)]
                names = tuple(f'_batch_{len(self._joint_batches)}_{key}' for key in ('bodies', 'joints', 'addresses'))
                for name, values in zip(names, (bodies, joints, addresses)):
                    self.register_buffer(name, torch.tensor(values), persistent=False)
                self._joint_batches.append((kind, names, width, bool((model.jnt_pos[joints] != 0).any())))
        depth = [0] * model.nbody
        for body in range(1, model.nbody):
            depth[body] = depth[self.parents[body]] + 1
        self._ancestors = []
        ancestors, span = list(self.parents), 1
        while span < max(depth):
            name = f'_ancestors_{span}'
            self.register_buffer(name, torch.tensor(ancestors), persistent=False)
            self._ancestors.append(name)
            ancestors = [ancestors[parent] for parent in ancestors]
            span *= 2

    def forward(self, qpos, scale=1.0, site_offsets=None):
        if qpos.shape[-1] != self.nq:
            raise ValueError(f"Expected {self.nq} pose coordinates, got {qpos.shape[-1]}.")
        scale = torch.as_tensor(scale, dtype=qpos.dtype, device=qpos.device)
        if scale.ndim and scale.shape != (self.nbody, 1):
            raise ValueError(f"Scale must be scalar or have shape ({self.nbody}, 1).")
        if site_offsets is not None and site_offsets.shape != (self.nsite, 3):
            raise ValueError(f"Site offsets must have shape ({self.nsite}, 3).")
        scales = scale.expand(self.nbody, 1)
        parallel = self.parallel if self.parallel is not None else qpos.device.type == 'mps'
        xpos, quaternions = (self._parallel_positions(qpos, scales) if parallel
                             else self._serial_positions(qpos, scales))
        local_sites = self.site_pos * scales[self.site_bodyid]
        if site_offsets is not None:
            local_sites = local_sites + site_offsets
        sites = xpos[..., self.site_bodyid, :] + _rotate(local_sites, quaternions[..., self.site_bodyid, :])
        return KinematicPositions(xpos, sites[..., self.site_reorder, :])

    def _parallel_positions(self, qpos, scales):
        """Batch local joint transforms, then compose ancestors in doubling passes."""
        batch = qpos.shape[:-1]
        positions = (self.body_pos * scales[self._parents]).expand(*batch, self.nbody, 3)
        rotations = self.body_quat.expand(*batch, self.nbody, 4)
        for kind, names, width, anchors in self._joint_batches:
            bodies, joints, addresses = (getattr(self, name) for name in names)
            values = qpos[..., addresses].reshape(*batch, len(bodies), width)
            rotation = rotations.index_select(-2, bodies)
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                positions = positions.index_copy(-2, bodies, values[..., :3])
                rotation = F.normalize(values[..., 3:7], dim=-1)
            elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
                displacement = values[..., 0] - self.qpos0[addresses]
                position = positions.index_select(-2, bodies) + _rotate(self.jnt_axis[joints], rotation) * displacement[..., None]
                positions = positions.index_copy(-2, bodies, position)
            else:
                if anchors:
                    anchor = positions.index_select(-2, bodies) + _rotate(self.jnt_pos[joints], rotation)
                if kind == mujoco.mjtJoint.mjJNT_BALL:
                    local = F.normalize(values, dim=-1)
                else:
                    angle = (values[..., 0] - self.qpos0[addresses]) / 2
                    local = torch.cat((angle.cos()[..., None], angle.sin()[..., None] * self.jnt_axis[joints]), -1)
                rotation = _quat_mul(rotation, local)
                if anchors:
                    positions = positions.index_copy(-2, bodies, anchor - _rotate(self.jnt_pos[joints], rotation))
            rotations = rotations.index_copy(-2, bodies, rotation)
        for name in self._ancestors:
            ancestors = getattr(self, name)
            parent_rotation = rotations.index_select(-2, ancestors)
            positions = positions.index_select(-2, ancestors) + _rotate(positions, parent_rotation)
            rotations = _quat_mul(parent_rotation, rotations)
        return positions, rotations

    def _serial_positions(self, qpos, scales):
        batch = qpos.shape[:-1]
        positions = [qpos.new_zeros((*batch, 3))]
        rotations = [self.body_quat[0].expand(*batch, 4)]
        for body in range(1, self.nbody):
            parent = self.parents[body]
            rotation = _quat_mul(rotations[parent], self.body_quat[body])
            position = positions[parent] + _rotate(self.body_pos[body] * scales[parent], rotations[parent])
            for kind, addr, joint in self.joints[body]:
                if kind == mujoco.mjtJoint.mjJNT_FREE:
                    position = qpos[..., addr:addr + 3]
                    rotation = F.normalize(qpos[..., addr + 3:addr + 7], dim=-1)
                elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
                    displacement = qpos[..., addr] - self.qpos0[addr]
                    position = position + _rotate(self.jnt_axis[joint], rotation) * displacement[..., None]
                else:
                    anchor = position + _rotate(self.jnt_pos[joint], rotation)
                    if kind == mujoco.mjtJoint.mjJNT_BALL:
                        local = F.normalize(qpos[..., addr:addr + 4], dim=-1)
                    else:
                        angle = (qpos[..., addr] - self.qpos0[addr]) / 2
                        local = torch.cat((angle.cos()[..., None], angle.sin()[..., None] * self.jnt_axis[joint]), dim=-1)
                    rotation = _quat_mul(rotation, local)
                    position = anchor - _rotate(self.jnt_pos[joint], rotation)
            positions.append(position)
            rotations.append(rotation)
        xpos = torch.stack(positions, dim=-2)
        quaternions = torch.stack(rotations, dim=-2)
        return xpos, quaternions
