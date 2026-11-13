"""CPU sampling for Torch video fitting, including ragged trials and phone attitude."""
import copy
from typing import Dict
import numpy as np


def _pad(values, length, axis=0):
    shape = list(values[0].shape)
    shape[axis] = length
    result = np.zeros((len(values), *shape), dtype=np.float64)
    for i, value in enumerate(values):
        count = min(value.shape[axis], length)
        if axis == 0:
            result[i, :count] = value[:count]
        else:
            result[i, :, :count] = value[:, :count]
    return result


def _stack(samples):
    if samples[0] is None:
        return None
    if isinstance(samples[0], tuple):
        return tuple(_stack(list(items)) for items in zip(*samples))
    return np.stack(samples)


class MonocularDataset:
    def __init__(self, timestamps, keypoints_2d, keypoints_3d, keypoint_confidence,
                 camera_params, phone_attitude=None, sample_length=None, align_timestamps=True,
                 max_length=None, future_timestamps=-1, keys=None):
        if future_timestamps != -1:
            raise NotImplementedError('Monocular future-timestamp targets are unsupported.')
        counts = [len(v) for v in (timestamps, keypoints_2d, keypoints_3d, keypoint_confidence)]
        if not counts[0] or len(set(counts)) != 1:
            raise ValueError('Provide matching nonempty lists of timestamps and keypoints.')
        timestamps, kp2d, kp3d, confidence = ([np.array(v, dtype=np.float64, copy=True) for v in values]
                                              for values in (timestamps, keypoints_2d, keypoints_3d, keypoint_confidence))
        attitudes = None if phone_attitude is None else [np.array(v, dtype=np.float64, copy=True) for v in phone_attitude]
        if attitudes is not None and len(attitudes) != counts[0]:
            raise ValueError('Phone attitude must have one recording per video.')
        originals = [t.copy() for t in timestamps]
        for i, time in enumerate(timestamps):
            if not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) < 0):
                raise ValueError('Video timestamps must be finite, nonempty, and sorted.')
            if kp2d[i].shape[:-1] != kp3d[i].shape[:-1] or kp2d[i].shape[:-1] != confidence[i].shape:
                raise ValueError('Keypoint and confidence dimensions must agree.')
            if kp2d[i].shape[1] != len(time) or kp2d[i].shape[-1] != 2 or kp3d[i].shape[-1] != 3:
                raise ValueError('Expected camera/time/joint/coordinate keypoint arrays matching timestamps.')
            if not np.isfinite(confidence[i]).all() or np.any(confidence[i] < 0):
                raise ValueError('Keypoint confidence must be finite and nonnegative.')
            valid = confidence[i] > 0
            if not np.isfinite(kp2d[i][valid]).all() or not np.isfinite(kp3d[i][valid]).all():
                raise ValueError('Confident keypoints must have finite coordinates.')
            kp2d[i][~valid], kp3d[i][~valid] = 0, 0
            if attitudes is not None:
                a = attitudes[i]
                if a.ndim != 2 or a.shape[1] != 4 or not len(a) or not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) < 0):
                    raise ValueError('Phone attitude must contain sorted timestamp/rotation-vector rows.')
            if align_timestamps:
                t0 = time[0] if attitudes is None else time[np.abs(time - attitudes[i][0, 0]).argmin()]
                time -= t0
                if attitudes is not None:
                    attitudes[i][:, 0] -= t0
                    mask = (time >= 0) & (time <= attitudes[i][-1, 0])
                    timestamps[i], originals[i] = time[mask], originals[i][mask]
                    kp2d[i], kp3d[i], confidence[i] = kp2d[i][:, mask], kp3d[i][:, mask], confidence[i][:, mask]
                    if not len(timestamps[i]):
                        raise ValueError('Phone attitude and video timestamps must overlap.')
                    attitude_mask = (attitudes[i][:, 0] >= 0) & (attitudes[i][:, 0] <= timestamps[i][-1])
                    attitudes[i] = attitudes[i][attitude_mask]
                    if not len(attitudes[i]):
                        raise ValueError('Phone attitude and video timestamps must overlap.')
        max_length = max_length if max_length is not None else max(map(len, timestamps))
        self.sample_length = int(sample_length if sample_length is not None else max(map(len, timestamps)))
        if max_length < 1 or self.sample_length < 1:
            raise ValueError('Sample and maximum lengths must be positive.')
        self.num_trials = len(timestamps)
        self.keys, self.camera_params = keys, camera_params
        self.timestamps_original = originals
        self.trial_lengths = np.array([min(len(t), max_length) for t in timestamps])
        self.max_time = max(t[-1] for t in timestamps)
        self.timestamps = _pad(timestamps, max_length)
        self.keypoints, self.keypoints_3d = _pad(kp2d, max_length, 1), _pad(kp3d, max_length, 1)
        self.keypoint_confidence = _pad(confidence, max_length, 1)
        self.phone_attitude, self.attitude_lengths = None, None
        if attitudes is not None:
            self.attitude_lengths = np.array([len(a) for a in attitudes])
            self.attitude_sample_length = sample_length or int(self.attitude_lengths.max())
            self.phone_attitude = _pad(attitudes, int(self.attitude_lengths.max()))

    def __len__(self):
        return self.num_trials

    def __getitem__(self, index, sample_length=None):
        if np.ndim(index) == 1:
            return _stack([self.__getitem__(int(i), sample_length) for i in index])
        if np.ndim(index) != 0:
            raise ValueError('Dataset indices must be scalars or one-dimensional arrays.')
        start, trial = divmod(int(index), len(self))
        length = self.trial_lengths[trial]
        count = self.sample_length if sample_length is None else sample_length
        # Interleave timestamps deterministically, matching the original sampler.
        sample = (np.arange(count) * int(np.ceil(length / count)) + start) % length
        time = self.timestamps[trial, sample]
        kp2d = np.take(self.keypoints[trial], sample, axis=1)
        kp3d = np.take(self.keypoints_3d[trial], sample, axis=1)
        confidence = np.take(self.keypoint_confidence[trial], sample, axis=1)
        attitude_time, attitude = None, None
        if self.phone_attitude is not None:
            length = self.attitude_lengths[trial]
            indices = (np.arange(self.attitude_sample_length) * int(np.ceil(length / self.attitude_sample_length)) + start) % length
            sampled = self.phone_attitude[trial, indices]
            attitude_time, attitude = sampled[:, 0], sampled[:, 1:]
        return (np.array(trial), time, attitude_time), (kp2d, kp3d, confidence, attitude)

    def get_all_timestamps(self, index):
        return self.timestamps[index, :self.trial_lengths[index]]

    def get_all_keypoints(self, index):
        length = self.trial_lengths[index]
        return self.keypoints[index, :, :length], self.keypoint_confidence[index, :, :length]

    def get_all_keypoints_3d(self, index):
        length = self.trial_lengths[index]
        return self.keypoints_3d[index, :, :length].squeeze(), self.keypoint_confidence[index, :, :length].squeeze()

    def get_all_attitude(self, index):
        attitude = self.phone_attitude[index, :self.attitude_lengths[index]]
        return attitude[:, 0].squeeze(), attitude[:, 1:].squeeze()

    def get_all_original_timestamps(self, index):
        return self.timestamps_original[index]
def get_samsung_calibration() -> Dict:
    """Gives the calibration params in the MultiCameraTracking format for our S20. I got these from a checkerboard video I filmed. Not sure how different this is between S20 cameras. Includes no distortion."""
    mtx = np.array(
        [[1.43333476e03, 0.00000000e00, 5.39500000e02], [0.00000000e00, 1.43333476e03, 9.59500000e02], [0.00000000e00, 0.00000000e00, 1.00000000e00]]
    )
    mtx = (mtx[[0, 1, 0, 1], [0, 1, 2, 2]] / 1000).reshape(1, -1)
    dist = np.zeros(5).reshape(1, -1)
    rvec = np.zeros(3).reshape(1, -1)
    tvec = np.zeros(3).reshape(1, -1)
    return dict(mtx=mtx, dist=dist, tvec=tvec, rvec=rvec)
