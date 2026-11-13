"""Compare original JAX and Torch fitting on identical video detections and weights."""
import argparse
import json
from pathlib import Path
import time

import cv2
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from monocular_demos.dataset import MonocularDataset as JaxDataset
from monocular_demos.biomechanics_mjx.monocular_trajectory import get_model as get_jax_model, fit_model as fit_jax_model, loss as jax_loss
from monocular_demos.torch_dataset import MonocularDataset, get_samsung_calibration
from monocular_demos.biomechanics_mjx.torch_monocular_trajectory import get_model, fit_model, loss
from monocular_demos.utils import joint_names


def transfer_weights(reference, model):
    with torch.no_grad():
        for index, trajectory in enumerate(model.trajectory):
            for source, target in zip(reference.trajectory.layers + [reference.trajectory.final], list(trajectory.layers) + [trajectory.final]):
                target.weight.copy_(torch.tensor(np.asarray(source.weight[index]).copy()))
                target.bias.copy_(torch.tensor(np.asarray(source.bias[index]).copy()))
            trajectory.max_time.copy_(torch.tensor(np.asarray(reference.trajectory.max_time[index]).copy()))
        model.body_scale.copy_(torch.tensor(np.asarray(reference.body_scale).copy()))
        model.site_offsets.copy_(torch.tensor(np.asarray(reference.site_offsets).copy()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=Path('data/running_sagittal.mp4'))
    parser.add_argument('--keypoints', type=Path, required=True)
    parser.add_argument('--torch-keypoints', type=Path)
    parser.add_argument('--output', type=Path, default=Path('validation/torch_port.json'))
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--features', type=int, nargs='+', default=[128, 256, 512, 1024, 2048, 2048, 4096])
    parser.add_argument('--concatenate-layers', type=int, default=3)
    parser.add_argument('--compute', choices=['float16', 'float32'], default='float16')
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    torch.set_num_threads(4)
    cap = cv2.VideoCapture(str(args.video))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    with np.load(args.keypoints) as data:
        kp2d, kp3d, confidence = (data[k].copy() for k in ('keypoints2d', 'keypoints3d', 'confs'))
    assert len(kp2d) == frames, 'Validation requires detections for every frame.'
    assert np.isfinite(kp2d).all() and np.isfinite(kp3d).all() and confidence.sum() > 0
    def dataset(cls):
        return cls(timestamps=[np.arange(frames) / fps], keypoints_2d=[kp2d[None]],
                   keypoints_3d=[kp3d[None]], keypoint_confidence=[confidence[None]], camera_params=get_samsung_calibration())
    original_data, torch_data = dataset(JaxDataset), dataset(MonocularDataset)
    options = dict(features=args.features, concatenate_layers=args.concatenate_layers,
                   jmp_policy=f'params=float32,compute={args.compute},output=float32')
    original = get_jax_model(original_data, joint_names=joint_names, **options)
    model = get_model(torch_data, joint_names=joint_names, **options)
    transfer_weights(original, model)
    loss_options = dict(lambda_reprojection=0.1, site_offset_regularization=100, static=True,
                        camera_calibration=get_samsung_calibration())
    initial_original, _ = jax_loss(original, *original_data[0], **loss_options)
    initial_torch, _ = loss(model, *torch_data[0], **loss_options)
    initial_difference = abs(float(initial_original) - initial_torch.item())
    print('initial losses', float(initial_original), initial_torch.item(), 'difference', initial_difference, flush=True)
    np.testing.assert_allclose(initial_torch.item(), float(initial_original), rtol=3e-3, atol=1e-4)
    # Compare parameter gradients before allowing the two optimizers to diverge.
    (_, _), gradients = eqx.filter_value_and_grad(jax_loss, has_aux=True)(original, *original_data[0], **loss_options)
    initial_torch.backward()
    gradient_errors = {}
    for name in ('body_scale', 'site_offsets'):
        a, b = np.asarray(getattr(gradients, name)), getattr(model, name).grad.numpy()
        gradient_errors[name] = float(np.max(np.abs(a - b)))
        np.testing.assert_allclose(b, a, rtol=5e-3, atol=5e-3)
    a = np.asarray(gradients.trajectory.final.weight[0])
    b = model.trajectory[0].final.weight.grad.numpy()
    gradient_errors['final_weight'] = float(np.max(np.abs(a - b)))
    np.testing.assert_allclose(b, a, rtol=1e-2, atol=5e-3)
    history_jax, history_torch = [], []
    def callback(history):
        return lambda i, m, d, metrics: history.append(float(np.asarray(metrics['total']).mean()))
    start = time.perf_counter()
    original, _ = fit_jax_model(original, original_data, max_iters=args.iters, lr_init_value=1e-3,
                                step_callback=callback(history_jax))
    jax_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model, _ = fit_model(model, torch_data, max_iters=args.iters, lr_init_value=1e-3,
                         step_callback=callback(history_torch))
    torch_seconds = time.perf_counter() - start
    final_jax, metrics_jax = jax_loss(original, *original_data[0], **loss_options)
    with torch.no_grad():
        final_torch, metrics_torch = loss(model, *torch_data[0], **loss_options)
        state_torch, (qpos_torch, _, _), _ = model(torch_data.get_all_timestamps(0), skip_vel=True)
    state_jax, (qpos_jax, _, _), _ = original(original_data.get_all_timestamps(0), skip_action=True, skip_vel=True, steps=0, check_constraints=False)
    assert float(final_jax) < float(initial_original) and final_torch.item() < initial_torch.item()
    # Independent nonlinear fits may take different paths, especially with half precision.
    assert final_torch.item() <= float(final_jax) * 1.15
    report = dict(video=str(args.video), frames=frames, fps=fps, iterations=args.iters,
                  features=args.features, compute=args.compute, initial_loss_jax=float(initial_original),
                  initial_loss_torch=initial_torch.item(), initial_absolute_difference=initial_difference,
                  initial_gradient_max_errors=gradient_errors, final_loss_jax=float(final_jax), final_loss_torch=final_torch.item(),
                  jax_seconds=jax_seconds, torch_seconds=torch_seconds,
                  metrics_jax={k: float(v) for k, v in metrics_jax.items()},
                  metrics_torch={k: v.item() for k, v in metrics_torch.items()},
                  qpos_rmse=float(np.sqrt(np.mean((np.asarray(qpos_jax) - qpos_torch.numpy())**2))),
                  marker_rmse_m=float(np.sqrt(np.mean((np.asarray(state_jax[0].site_xpos) - state_torch[0].site_xpos.numpy())**2))),
                  history_jax=history_jax, history_torch=history_torch,
                  versions=dict(torch=torch.__version__, jax=jax.__version__))
    if args.torch_keypoints:
        with np.load(args.torch_keypoints) as detected:
            kp2d, kp3d, confidence = (detected[k].copy() for k in ('keypoints2d', 'keypoints3d', 'confs'))
        torch_video_data = dataset(MonocularDataset)
        torch_video_model = get_model(torch_video_data, joint_names=joint_names, **options)
        start = time.perf_counter()
        initial_video_loss = loss(torch_video_model, *torch_video_data[0], **loss_options)[0].item()
        torch_video_model, _ = fit_model(torch_video_model, torch_video_data, max_iters=args.iters, lr_init_value=1e-3)
        seconds = time.perf_counter() - start
        with torch.no_grad():
            video_loss, video_metrics = loss(torch_video_model, *torch_video_data[0], **loss_options)
            (video_state, _, _), (pose, velocity, _), rotation = torch_video_model(torch_video_data.get_all_timestamps(0))
        assert torch.isfinite(video_loss) and video_loss.item() < initial_video_loss
        report['end_to_end'] = dict(initial_loss=initial_video_loss, final_loss=video_loss.item(), seconds=seconds,
            metrics={k: v.item() for k, v in video_metrics.items()},
            marker_rmse_vs_original_m=float(np.sqrt(np.mean((np.asarray(state_jax[0].site_xpos) - video_state.site_xpos.numpy())**2))),
            centered_marker_rmse_vs_original_m=float(np.sqrt(np.mean((
                np.asarray(state_jax[0].site_xpos) - np.asarray(state_jax[0].site_xpos).mean(axis=1, keepdims=True)
                - video_state.site_xpos.numpy() + video_state.site_xpos.numpy().mean(axis=1, keepdims=True))**2))))
        np.savez(args.output.with_suffix('.npz'), timestamps=torch_video_data.get_all_timestamps(0),
            qpos=pose.numpy(), qvel=velocity.numpy(), rnc=rotation.numpy(), sites=video_state.site_xpos.numpy(),
            joints=video_state.xpos.numpy(), scale=torch_video_model.body_scale.detach().numpy(), video_path=str(args.video),
            reference_qpos=np.asarray(qpos_jax), reference_sites=np.asarray(state_jax[0].site_xpos))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if not k.startswith('history')}, indent=2))


if __name__ == '__main__':
    main()
