"""Cache original TensorFlow Metrabs detections in a separate environment."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def detect_video(model, video, output, calibration=None, batch_size=8):
    detection_options = {}
    if calibration:
        with np.load(calibration) as parameters:
            fx, fy, cx, cy = parameters['mtx'][0] * 1000
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        detection_options['intrinsic_matrix'] = intrinsic[None]
    cap = cv2.VideoCapture(str(video))
    poses2d, poses3d, confidences, boxes = [], [], [], []
    try:
        while True:
            batch = []
            for _ in range(batch_size):
                ok, frame = cap.read()
                if not ok:
                    break
                batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not batch:
                break
            prediction = model.detect_poses_batched(np.stack(batch), skeleton='bml_movi_87', **detection_options)
            for i in range(len(batch)):
                found = len(prediction['boxes'][i]) > 0
                poses2d.append(prediction['poses2d'][i][0].numpy() if found else np.zeros((87, 2)))
                poses3d.append(prediction['poses3d'][i][0].numpy() if found else np.zeros((87, 3)))
                confidences.append(np.ones(87) if found else np.zeros(87))
                boxes.append(prediction['boxes'][i][0].numpy() if found else np.zeros(5))
    finally:
        cap.release()
    if not poses2d:
        raise ValueError(f'No video frames could be read from {video}.')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, keypoints2d=poses2d, keypoints3d=poses3d, confs=confidences, boxes=boxes)
    return len(poses2d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, default=Path('data/running_sagittal.mp4'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path)
    args = parser.parse_args()
    model = hub.load('https://bit.ly/metrabs_l')
    count = detect_video(model, args.video, args.output, args.calibration)
    metadata = dict(tensorflow_version=tf.__version__,
                    available_devices=[d.name for d in tf.config.list_logical_devices()], frames=count)
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {count} frames to {args.output}')



if __name__ == '__main__':
    main()
