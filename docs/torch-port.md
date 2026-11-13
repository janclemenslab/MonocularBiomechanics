# Torch port: usage and validation

This fork supports fitting new videos with PyTorch. The original JAX implementation remains available for reference comparisons through the `reference` extra. Native MuJoCo loads model metadata and renders motion; differentiable forward kinematics runs in Torch. Contacts and differentiable physics rollouts are outside this port.

## Installation and devices

Follow the conda/uv installation in the README, then run `python main.py` or the notebook. The detector downloads pinned upstream [Metrabs](https://github.com/isarandi/metrabs) and PosePile source plus pretrained weights into `~/.cache/monocular-biomechanics/` on first use. Upstream pretrained-model terms apply. The adapter preserves the 87-marker skeleton, RGB inputs, millimetre outputs, and empty detections; cached upstream source receives small inference compatibility patches.

The default fitter selects CUDA when available, otherwise CPU. Apple GPU fitting is explicit:

```python
from monocular_demos.biomechanics_mjx.torch_monocular_trajectory import get_model, fit_model

model = get_model(dataset, device="mps")
model, metrics = fit_model(model, dataset, max_iters=5000)
```

Use a `MonocularDataset` from `monocular_demos.torch_dataset`. MPS uses float32 geometry and trajectory computation; CPU defaults to float64 geometry with float32 trajectory computation. CUDA uses float16 trajectory computation with float32 network parameters, but CUDA has not been validated in this fork. The Metrabs detector defaults to CPU on Apple Silicon; MPS fitting does not imply MPS detection.

MPS kinematics batches body-local joint transforms and composes ancestors in doubling passes. Free, ball, hinge and slide joints, pose batches, learned body scales and site offsets are supported. `model.kinematics.parallel = False` retains the serial control. Optimizer update clipping stays on the selected device. CPU fallback was disabled during GPU validation; no compiler setup is required.

## Checks

Install reference dependencies for comparisons with JAX/MJX:

```bash
uv pip install -e '.[reference,dev]'
python -m pytest -q tests/test_torch_kinematics.py tests/test_torch_fitting.py
python -m pytest -q tests/test_opencap_budgets.py tests/test_opencap_evaluation.py tests/test_opencap_joints.py tests/test_opencap_videos.py
python -m pytest -q tests/test_pipeline.py tests/test_rendering.py
```

The first group checks positions and gradients against native MuJoCo and JAX/MJX, serial/parallel parity, camera projection, trajectory sampling, fitting and CPU/MPS parity. The full pipeline tests run real detection, fitting and rendering and may download weights.

The [Windows CPU smoke workflow](../.github/workflows/windows-smoke.yml) runs on relevant pushes to the fork's `main` and can also be started manually in Actions. One Python 3.11 conda environment installs CPU-only Torch and the minimal fitting dependencies, then runs the existing kinematics and synthetic fitting tests. JAX and Apple GPU checks skip because those dependencies/devices are absent. The job has a ten-minute limit and caches downloads; it does not download detector weights or evaluation videos. It checks CPU fitting code and editable package installation, not the complete dependency installation, detection, GUI, CUDA or rendering.

## OpenCap evaluation

The [OpenCap SimTK validation archive](https://simtk.org/frs/?group_id=2385), `LabValidation_withVideos.zip`, is about 19 GB and requires login and acceptance of its license. Keep it outside Git. `analysis.extract_opencap_pilot` extracts a small pilot; `analysis.evaluate_opencap` evaluates one camera/trial. Run either module with `--help` for its inputs. Original TensorFlow detections require a separate environment:

```bash
conda create -n monocular-tf --override-channels -c conda-forge python=3.11 -y
conda activate monocular-tf
uv pip install 'tensorflow==2.15.0' 'tensorflow-hub==0.16.1' 'tf-keras==2.15.1' 'setuptools<81' opencv-python
conda activate monocular-torch
python -m analysis.evaluate_opencap --help
python -m analysis.evaluate_opencap_batch --help
```

Supply the synchronized source video, measured camera calibration, **marker-based** motion-capture IK `.mot` reference and associated scaled OpenSim `.osim` model. `--time-offset` is reference time at source-video frame zero; record independently established synchronization with `--sync-source`. The evaluator preserves actual source-frame times, corrects lens distortion and scores only overlapping reference samples. Reference joint angles are used for scoring, not fitting.

`analysis.score_opencap_joints` maps all 27 comparable relative joint coordinates and reports MAE, RMSE, bias and reference range of motion. Neck, wrists and global pelvis pose are unscored; toe references are nearly static. Longer fitting can improve knees while worsening other coordinates.

To rerun a cached baseline at 5,000 steps on Apple GPU:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m analysis.evaluate_opencap_convergence \
  --clip validation/opencap_100/subject2_STS1_Session0_Cam3 \
  --output validation/opencap_100/torch_5000/subject2_STS1_Session0_Cam3 \
  --iterations 5000 --device mps --geometry-dtype float32
```

This comparison tool requires the cached baseline and reference dependencies: it transfers the same deterministic JAX initialization, verifies detector hashes and clocks, and saves joint poses, fitted sites, learned scales and offsets. It synchronizes MPS around timing. `--serial-kinematics` selects the slower control.

### Six-clip study

Six fixed five-second, 75-frame clips covered participants 2, 3, 4, 7, 9 and 11, with one camera/trial per participant. Cached detections and initialization were held fixed; all 27 joint scores were independently recomputed from saved poses against synchronized marker-based IK.

| Metric | CPU 2k | CPU 5k | Fast MPS 5k |
|---|---:|---:|---:|
| Mean knee MAE | 14.51° | 10.35° | 10.36° |
| Mean MAE across 27 coordinates | 9.46° | 9.07° | 9.07° |
| Median fitting time | — | 303.96 s | 119.65 s |

Compared with CPU 2k, MPS 5k improved mean knee error by 28.6%; 5/6 participant averages and 11/12 individual knees improved. GPU versus CPU 5k differed by at most 0.10° in the six-clip average MAE for any coordinate. These are historical rather than interleaved CPU timings, exclude detection/initialization/rendering, and compare CPU float64 geometry with GPU float32 geometry. Participant and activity effects cannot be separated in this small sample. The larger 100-clip evaluation remains incomplete.

## Portable video gallery

With the six local fit records and their manifest prepared under `validation/opencap_100/torch_5000/`:

```bash
python -m analysis.opencap_gallery
```

The builder verifies fitted geometry, input clocks and joint scores, then renders raw footage / measured ground-truth skeleton / Torch 5k prediction. Footage is lens-corrected without detection overlays. Skeleton views share scale, with pelvis translation and initial heading removed for display only. Ground-truth skeletons use measured markers and hip centers; angle references are marker-based IK.

The exported `validation/opencap_gallery/` contains HTML, MP4s, posters and scores with relative links. Copy the entire folder and open `index.html`; viewing needs no Python, model files, server or internet connection. The builder consumes existing local evaluation artifacts; it does not download the dataset or generate the six-fit manifest. Generated datasets, checkpoints, reports and galleries are excluded from Git and must be transferred separately.
