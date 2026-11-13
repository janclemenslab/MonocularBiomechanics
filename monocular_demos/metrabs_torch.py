"""Load upstream's pretrained PyTorch Metrabs and adapt its batched image API."""
from functools import lru_cache
from pathlib import Path
import pickle
import os
import sys
import tarfile
import textwrap
import urllib.request

import numpy as np
import torch
from filelock import FileLock

SOURCE_REVISION = '8b2b116dd27372e7dbd8207809f868df4e3f852e'
POSEPILE_REVISION = 'bc129c3bb08a30f7fd6e164c4c527d39c82d6a14'
MODEL_NAME = 'metrabs_eff2l_384px_800k_28ds_pytorch'


def _download_archive(url, destination):
    archive, _ = urllib.request.urlretrieve(url)
    try:
        with tarfile.open(archive) as tar:
            tar.extractall(destination, filter='data')
    finally:
        Path(archive).unlink(missing_ok=True)


class MetrabsDetector:
    def __init__(self, model, device):
        self.model, self.device = model, device

    @torch.inference_mode()
    def detect_poses_batched(self, images, skeleton='bml_movi_87', **kwargs):
        images = torch.as_tensor(images, device=self.device)
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != torch.uint8:
            raise ValueError('Expected a batch of RGB uint8 images with shape (batch, height, width, 3).')
        if images.shape[0] == 0:
            return dict(boxes=[], poses2d=[], poses3d=[])
        images = images.permute(0, 3, 1, 2)
        kwargs.setdefault('max_detections', 1)
        # Upstream creates auxiliary tensors on the default device.
        with torch.device(self.device):
            prediction = self.model.detect_poses_batched(images, skeleton=skeleton, **kwargs)
        return {name: [p.detach().cpu() for p in values] for name, values in prediction.items()}


@lru_cache(maxsize=1)
def load_metrabs(device=None, cache_dir=None):
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cache = Path(cache_dir or Path.home() / '.cache/monocular-biomechanics')
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / '.load.lock')):
        source = cache / f'metrabs-{SOURCE_REVISION}'
        if not source.exists():
            _download_archive(f'https://codeload.github.com/isarandi/metrabs/tar.gz/{SOURCE_REVISION}', cache)
        posepile_source = cache / f'posepile-{POSEPILE_REVISION}'
        if not posepile_source.exists():
            _download_archive(f'https://codeload.github.com/isarandi/PosePile/tar.gz/{POSEPILE_REVISION}', cache)
        sys.path.insert(0, str(posepile_source / 'src'))
        model_dir = cache / MODEL_NAME
        if not model_dir.exists():
            _download_archive('https://bit.ly/metrabs_l_pt', cache)
        # Dataset loading is unrelated to inference and pulls in training-only dependencies.
        utility = source / 'metrabs_pytorch/util.py'
        text = utility.read_text()
        if '\nimport posepile.datasets3d as ds3d\n' in text:
            text = text.replace('import posepile.datasets3d as ds3d\n', '')
            text = text.replace("    if skeleton_type_dst == '':", "    import posepile.datasets3d as ds3d\n    if skeleton_type_dst == '':")
            utility.write_text(text)
        detector_source = source / 'metrabs_pytorch/multiperson/person_detector.py'
        detector_source.write_text(detector_source.read_text().replace(
            "ultralytics.YOLO('yolov8m.pt')", f"ultralytics.YOLO({str(cache / 'yolov8m.pt')!r})"))
        multiperson = source / 'metrabs_pytorch/multiperson/multiperson_model.py'
        text = multiperson.read_text().replace('torch.split(poses3d_flat, n_box_per_image)', 'torch.split(poses3d_flat, n_box_per_image.tolist())')
        text = text.replace('torch.split(poses2d_flat, n_box_per_image)', 'torch.split(poses2d_flat, n_box_per_image.tolist())')
        counts_line = "        n_box_per_image = torch.tensor([len(b) for b in boxes], device=intrinsic_matrix.device)"
        empty_case = """
            if sum(len(b) for b in boxes) == 0:
                joints = len(self.per_skeleton_joint_names[skeleton])
                pose3d_shape = (0, joints, 3) if average_aug else (0, num_aug, joints, 3)
                pose2d_shape = (0, joints, 2) if average_aug else (0, num_aug, joints, 2)
                return dict(boxes=boxes,
                            poses3d=[images.new_empty(pose3d_shape, dtype=torch.float32) for _ in boxes],
                            poses2d=[images.new_empty(pose2d_shape, dtype=torch.float32) for _ in boxes])
    """
        if 'pose3d_shape = ' not in text:
            text = text.replace(counts_line, counts_line + textwrap.indent(textwrap.dedent(empty_case), '        '))
        multiperson.write_text(text)
        os.environ.setdefault("DATA_ROOT", str(cache))
        sys.path.insert(0, str(source))
        import posepile.joint_info
        from metrabs_pytorch.util import get_config
        from metrabs_pytorch.backbones import efficientnet
        from metrabs_pytorch.models.metrabs import Metrabs
        from metrabs_pytorch.multiperson.multiperson_model import Pose3dEstimator

        cfg = get_config(str(model_dir / 'config.yaml'))
        info = np.load(model_dir / 'joint_info.npz')
        joints = posepile.joint_info.JointInfo(info['joint_names'], info['joint_edges'])
        backbone = getattr(efficientnet, f'efficientnet_v2_{cfg.efficientnet_size}')()
        crop = Metrabs(torch.nn.Sequential(efficientnet.PreprocLayer(), backbone.features), joints).eval()
        crop((torch.zeros(1, 3, cfg.proc_side, cfg.proc_side), torch.eye(3)[None]))
        crop.load_state_dict(torch.load(model_dir / 'ckpt.pt', map_location='cpu', weights_only=True))
        with open(model_dir / 'skeleton_infos.pkl', 'rb') as handle:
            skeletons = pickle.load(handle)
        if 'bml_movi_87' not in skeletons:
            raise ValueError('Metrabs weights do not contain the required bml_movi_87 skeleton.')
        with torch.device(device):
            estimator = Pose3dEstimator(crop.to(device), skeletons, np.load(model_dir / 'joint_transform_matrix.npy'))
        estimator.detector.yolo.overrides['device'] = str(device)
        return MetrabsDetector(estimator, device)
