"""Differentiable camera projection using the original calibration conventions."""
import torch


def rotation_matrix(rvec):
    rvec = torch.as_tensor(rvec)
    x, y, z = rvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*rvec.shape[:-1], 3, 3)
    return torch.matrix_exp(skew)


def rotation_quaternion(rvec):
    theta = torch.linalg.vector_norm(rvec, dim=-1, keepdim=True)
    return torch.cat(((theta / 2).cos(), 0.5 * torch.sinc(theta / (2 * torch.pi)) * rvec), -1)


def get_intrinsic(camera_params, i):
    m = torch.as_tensor(camera_params['mtx'])[i] * 1000
    zero, one = m.new_zeros(()), m.new_ones(())
    return torch.stack((m[0], zero, m[2], zero, m[1], m[3], zero, zero, one)).reshape(3, 3)


def get_extrinsic(camera_params, i):
    rvec = torch.as_tensor(camera_params['rvec'])[i]
    return _extrinsic(rvec, torch.as_tensor(camera_params['tvec'])[i].to(rvec) * 1000)


def _extrinsic(rvec, tvec):
    rotation = rotation_matrix(rvec)
    translation = tvec.expand(*rvec.shape[:-1], 3)
    top = torch.cat((rotation, translation[..., None]), -1)
    bottom = top.new_tensor([0, 0, 0, 1]).expand(*top.shape[:-2], 1, 4)
    return torch.cat((top, bottom), -2)


def get_extrinsic_dynamic(camera_params, i, rvec):
    rvec = torch.as_tensor(rvec)
    return _extrinsic(-rvec, torch.as_tensor(camera_params['tvec'])[i].to(rvec) * 1000)


def transform_points(points, extrinsic):
    points = torch.as_tensor(points)
    extrinsic = extrinsic.to(points)
    if points.shape[-1] == 3:
        points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    return torch.einsum('...ij,...bj->...bi', extrinsic, points)[..., :3]


def _project_camera(points, camera_params, i):
    z = points[..., 2]
    denominator = torch.where(z.abs() < 100, 1e6, z)
    xy = points[..., :2] / denominator[..., None]
    radius = torch.nan_to_num(xy.square().sum(-1), nan=0)
    distortion = torch.as_tensor(camera_params['dist'])[i].to(points)
    gamma = (1 + distortion[0] * radius) * torch.where(z < 0, 1e-3, 1.0)
    xy = xy * gamma[..., None]
    intrinsic = get_intrinsic(camera_params, i).to(points)
    return xy * intrinsic.diagonal()[:2] + intrinsic[:2, 2]


def project_distortion(camera_params, i, points):
    return _project_camera(transform_points(points, get_extrinsic(camera_params, i)), camera_params, i)


def project_dynamic(points, calibration, rvec):
    return _project_camera(transform_points(points, get_extrinsic_dynamic(calibration, 0, rvec)), calibration, 0)


def project(camera_params, i, points):
    points = transform_points(points, get_extrinsic(camera_params, i))
    intrinsic = get_intrinsic(camera_params, i).to(points)
    return points[..., :2] / points[..., 2:] * intrinsic.diagonal()[:2] + intrinsic[:2, 2]


def get_projection(camera_params, i):
    return get_intrinsic(camera_params, i) @ get_extrinsic(camera_params, i)[:3]


def distort_3d(camera_params, i, points):
    points = transform_points(points, get_extrinsic(camera_params, i))
    z = points[..., 2:]
    x, y = (points[..., :2] / z).unbind(-1)
    radius = x.square() + y.square()
    d = torch.as_tensor(camera_params['dist'])[i].to(points)
    gamma = 1 + d[0] * radius + d[1] * radius**2 + d[4] * radius**3
    xp = gamma * x + 2 * d[2] * x * y + d[3] * (radius + 2 * x.square())
    yp = gamma * y + d[2] * (radius + 2 * y.square()) + 2 * d[3] * x * y
    return torch.cat((torch.stack((xp, yp), -1) * z, z), -1)
