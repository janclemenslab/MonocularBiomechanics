"""Test a longer Torch fit using cached detections and the original initialization."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import numpy as np
import torch

from analysis.evaluate_opencap import read_mot, score_angles
from analysis.score_opencap_joints import JOINTS, coordinate_signs, score_saved_fit
from analysis.validate_torch_port import transfer_weights
from monocular_demos.dataset import MonocularDataset as JaxDataset
from monocular_demos.torch_dataset import MonocularDataset
from monocular_demos.biomechanics_mjx.monocular_trajectory import get_model as original_model
from monocular_demos.biomechanics_mjx.torch_monocular_trajectory import get_model, fit_model
from monocular_demos.biomechanics_mjx.native_model import ForwardKinematics
from monocular_demos.utils import joint_names


def run(directory, output, iterations=10000, device='cpu', geometry_dtype=None, serial_kinematics=False):
    if iterations < 10:
        raise ValueError('Fitting budget must be at least ten steps.')
    jax.config.update('jax_enable_x64', True)
    torch.set_num_threads(4)
    geometry_dtype = geometry_dtype or ('float32' if device == 'mps' else 'float64')
    if device == 'mps' and geometry_dtype != 'float32':
        raise ValueError('MPS fitting requires float32 geometry.')
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((directory/'metrics.json').read_text())
    with np.load(directory/'trajectories.npz') as arrays:
        source_times, times = arrays['source_time'], arrays['reference_time']
    with np.load(directory/'prepared.calibration.npz') as arrays:
        calibration = {k: arrays[k] for k in arrays.files}

    def dataset(cls, filename):
        with np.load(directory/filename) as arrays:
            return cls(timestamps=[source_times.copy()], keypoints_2d=[arrays['keypoints2d'][None]],
                       keypoints_3d=[arrays['keypoints3d'][None]], keypoint_confidence=[arrays['confs'][None]],
                       camera_params=deepcopy(calibration))

    original_data = dataset(JaxDataset, 'tensorflow_keypoints.npz')
    data = dataset(MonocularDataset, 'torch_keypoints.npz')
    options = dict(joint_names=joint_names, jmp_policy='params=float32,compute=float32,output=float32')
    original = original_model(original_data, **options)
    model = get_model(data, device='cpu', dtype=getattr(torch, geometry_dtype), **options)
    transfer_weights(original, model)
    if serial_kinematics:
        model.kinematics.parallel = False
    del original, original_data
    initial_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        initial_hash.update(name.encode())
        initial_hash.update(value.detach().cpu().float().numpy().tobytes())
    model.to(device=device)
    native = ForwardKinematics().model
    signs = coordinate_signs(report['reference_model'], native)
    addresses = [int(native.joint(n).qposadr[0]) for n in JOINTS]
    reference_time, reference = read_mot(report['mocap_ik'], JOINTS)
    baseline, _ = score_saved_fit(directory, native)
    result = dict(status='running', clip=str(directory.resolve()), iterations=iterations,
                  baseline_iterations=report['iterations'],
                  fit_device=device, geometry_dtype=geometry_dtype, network_compute='float32',
                  kinematics='parallel' if device == 'mps' and not serial_kinematics else 'serial',
                  initial_state_sha256_float32=initial_hash.hexdigest(),
                  mps_cpu_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0'),
                  baseline_metrics=[r for r in baseline if r['pipeline'] == 'torch'],
                  initialization='Same deterministic JAX initialization and weight transfer as baseline',
                  detection_sha256=hashlib.sha256((directory/'torch_keypoints.npz').read_bytes()).hexdigest(),
                  schedule=f'Fresh {iterations}-step fit; learning rate 1e-3 to 1e-7, same AdamW and loss as baseline.',
                  limitations=['One clip per fit; selection is recorded separately in the experiment manifest.',
                               f'Intermediate checkpoints use the {iterations}-step learning-rate schedule.',
                               'Ground truth is used only for scoring; no settings are selected by reference error.'],
                  checkpoints=[])
    if device == 'mps':
        torch.mps.synchronize()
    start = time.perf_counter()

    def checkpoint(step, model, dataset, metrics):
        if step+1 not in (2000, 5000, iterations):
            return
        with torch.no_grad():
            state = model(source_times-source_times[0], skip_vel=True)
            qpos = state[1][0].cpu().numpy()
        rows, _, _ = score_angles(times, np.rad2deg(qpos[:, addresses])*signs,
                                   reference_time, reference, JOINTS)
        result['checkpoints'].append(dict(step=step+1, elapsed_seconds=time.perf_counter()-start,
                                          fitting_loss_before_update=float(metrics['total'].mean()),
                                          mae_deg=float(np.mean([r['mae_deg'] for r in rows])), metrics=rows))
        np.savez(output/f'checkpoint_{step+1}.npz', reference_time=times, qpos=qpos,
                 sites_camera_mm=state[0][0].site_xpos.cpu().numpy()*1000,
                 body_scale=model.body_scale.detach().cpu().numpy(),
                 site_offsets=model.site_offsets.detach().cpu().numpy())
        if step+1 == iterations:
            result['status'] = 'complete'
        partial = output/'metrics.partial.json'
        partial.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        partial.replace(output/'metrics.json')
        print(json.dumps({k:v for k,v in result['checkpoints'][-1].items() if k != 'metrics'}), flush=True)

    fit_model(model, data, max_iters=iterations, lr_init_value=1e-3, step_callback=checkpoint)
    if device == 'mps':
        torch.mps.synchronize()
    result['fitting_seconds'] = time.perf_counter()-start
    partial = output/'metrics.partial.json'
    partial.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    partial.replace(output/'metrics.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--device', choices=['cpu', 'mps'], default='cpu')
    parser.add_argument('--geometry-dtype', choices=['float32', 'float64'])
    parser.add_argument('--serial-kinematics', action='store_true', help='Use the original joint-tree loop for comparison.')
    args = parser.parse_args()
    run(args.clip, args.output, args.iterations, args.device, args.geometry_dtype, args.serial_kinematics)
