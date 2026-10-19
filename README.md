# mp-gpu

Fast, faithful **MediaPipe Holistic** pose extraction for large sign-language video collections, on all visible GPUs.

`mp-gpu` re-implements the **legacy MediaPipe Holistic** solution and runs the original MediaPipe models batched on the GPU. That solution is `mp.solutions.holistic` in mediapipe 0.10.21, with `model_complexity=1`, smoothing on and `refine_face_landmarks=False`, as used by `pose_format`'s `video_to_pose`. Its output matches the legacy CPU pipeline to a few hundredths of a pixel on short clips. On an RTX 3090 it is about **100× faster than one CPU process** (~1,400 vs ~12 frames/s).

```bash
mp-gpu /data/videos -o /data/poses            # a folder (recursive), all visible GPUs
mp-gpu a.mp4 b.mov --list more.txt -o poses    # files and/or a list
CUDA_VISIBLE_DEVICES=2,3 mp-gpu ...            # or --gpus 2,3
```

```
mp-gpu  37%|███▋      | 7:51/21:14 video | files 118/381 | in flight 512 (9:02) | 3104 fps, 125x realtime [03:46<06:24]
```

The bar shows video time processed out of total (h:mm), files finished, how many videos are open right now (and their total duration), frames/s, and speed relative to real time.

## Install

Linux and an NVIDIA GPU are required (developed on RTX 3090, driver 550, CUDA 12). Use a Python ≥3.10 env with a CUDA build of PyTorch that matches your driver, then:

```bash
pip install git+ssh://git@github.com/tmh-signlanguage/mp-gpu.git            # or: pip install -e .
pip install "mp-gpu[pose] @ git+ssh://git@github.com/tmh-signlanguage/mp-gpu.git"   # + .pose output
```

This pulls in TensorRT 10.9 (`tensorrt-cu12`), PyNvVideoCodec 2.0.2 (NVDEC), PyAV, onnx and tqdm. Triton comes with PyTorch.

On first use, TensorRT engines are built for your GPU model: a one-time job of 5–30 minutes. They're cached with the color lookup tables in `~/.cache/mp-gpu`, or `$MP_GPU_CACHE`. Concurrent processes wait for each other rather than building twice.

## Output

The output is one file per video, `OUT/<relative path>/<video file name>.npz`, holding the same 576 points as `pose_format.utils.holistic`:

| field | shape | content |
|---|---|---|
| `data` | float32 [T, 576, 3] | x, y in pixels, z as in pose_format. Order: body 33, face 468, left hand 21, right hand 21, body world 33 |
| `conf` | float32 [T, 576] | visibility (body/world), 1/0 presence (face/hands) |
| `fps`, `width`, `height` | scalars | |
| `components`, `points` | str arrays | component names and `"COMPONENT/POINT"` per column |

Missing parts are zeros with `conf = 0`. Use `--format pose` (or `both`) for pose_format `.pose` files instead.

```python
import numpy as np
z = np.load("x.mp4.npz"); z["data"], z["conf"], float(z["fps"])
from mp_gpu.pose_io import npz_to_pose, load      # pose_format.Pose / plain dict, .npz or .pose
```

Existing outputs are skipped, so an interrupted run resumes; `--overwrite` re-does them. Failures are listed in `OUT/mp-gpu-errors.log`.

## Options

| option | default | |
|---|---|---|
| `--format npz\|pose\|both` | `npz` | |
| `--gpus 0,1` | all visible | one worker process per GPU, all sharing one work queue (longest videos first) |
| `--streams N` | 256 | videos processed concurrently per GPU; throughput needs many videos in flight |
| `--decode-mem-gb G` | 10 | per-GPU memory budget for open decoders (NVDEC needs ~28 B/pixel: 49 MB at 1080p, 238 MB at 4K) |
| `--color pyav\|opencv` | `pyav` | YUV→RGB like PyAV/pose_format (honours BT.709 tags), or like `cv2.VideoCapture` (always BT.601). Use `opencv` to match data extracted with OpenCV. |
| `--target-fps F` | – | subsample faster sources (output i ← source frame round(i·src/target)) |
| `--decoder auto\|nvdec\|cpu` | `auto` | see below |
| `--half` | off | fp16 TensorRT: ~20% faster, but ~0.6 px instead of ~0.04 px from legacy |
| `--backend trt\|torch` | `trt` | PyTorch backend: no TensorRT needed, ~3× slower |

## How it works

1. **Decode.**
   - H.264, HEVC, VP9 and AV1 in 8-bit 4:2:0 are decoded by **NVDEC** straight into GPU memory. These codecs decode bit-exactly by specification, verified identical to libavcodec.
   - Frames that libavcodec drops because of MP4 edit lists are dropped too, so frame indices match PyAV.
   - Everything else is decoded on the CPU with PyAV, i.e. exactly the frames pose_format sees: MPEG-4 Part 2, MPEG-2, MJPEG, 10-bit, 4:2:2, ….
2. **Crop.** A fused Triton kernel converts YUV→RGB per bilinear tap through an exact libswscale lookup table, then samples rotated ROIs with an exact emulation of OpenCV `warpPerspective` (fixed-point, 1/32 px). Crops are bit-identical to *decode → rgb24 → cv2 warp*.
3. **Models.**
   - The seven MediaPipe models are ported to PyTorch (relative error ~1e-6 against TFLite) and run with TensorRT fp32, TF32 off.
   - S videos are processed in lock-step, so every model runs at batch ≈ S.
   - All six TensorRT engines share one activation buffer.
4. **Holistic graph.** Pose detection, ROI tracking, heatmap refinement, OneEuro/low-pass smoothing, the hand recrop and tracking, and face detection and tracking are vectorized over streams. `docs/HOLISTIC_SPEC.md` has the algorithm with file:line references to MediaPipe.

## Accuracy against legacy CPU MediaPipe Holistic

| check | result |
|---|---|
| single frames (fresh state), 480p–1080p | ≤ 0.1 px, typically 0.002–0.03 px |
| SSLL, 74 clips with exactly reproducible references | body median 0.04 px, p99 0.4 px; face 0.01 px; hands ~0.07 px; presence disagreement: pose 0, face 0, hands 0.2% |
| YTSL25 swl, 381 YouTube videos (H.264/VP9 incl. 10-bit, 480p–4K, 24–60 fps) | frame count, fps and size identical for all; body median 0.08% of diagonal |
| SSLC, SSLL, WLASL, ASLLVD (MPEG-4 `.mov`), PHOENIX: 10 videos each | early-frame error 0.03–0.10 px |

Over long videos, trajectories drift from legacy by ~0.7 px. That is inherent to MediaPipe's tracking loop: two runs of this pipeline that differ only in float rounding inside the models drift apart by the same amount. On YouTube intros (titles, logos) the tracker can lock onto non-signer content, and any two implementations then diverge until the signer is tracked.

## Speed (RTX 3090)

| | frames/s per GPU |
|---|---|
| legacy Holistic, 1 CPU process | ~12 |
| mp-gpu, 696×956 crops, 256 streams | ~1,380 (fp32), ~1,650 (fp16) |
| mp-gpu, 2 GPUs | ~2,730 total (scales linearly) |
| mp-gpu, YouTube 720p–1080p mix | ~880 while all streams are busy |

Throughput needs many videos in flight; with few long videos the GPUs idle, and splitting into chunks is not implemented. Decoding uses ~2 CPU cores per GPU.

## Tools

- `tools/compare_dataset.py REF_DIR TEST_DIR`: per-video comparison of mp-gpu output with existing legacy `.pose` files. It reports frame count, fps, presence agreement and error, and flags anomalies.
- `tools/tflite_export.py`: re-creates `src/mp_gpu/models` from MediaPipe `.tflite` files (needs TensorFlow).

## License notes

The model weights in `src/mp_gpu/models` are converted from Google MediaPipe (Apache-2.0); see `NOTICE`.
