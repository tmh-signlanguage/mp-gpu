# MediaPipe legacy Holistic (CPU graph, v0.10.21): implementation spec

Configuration: `static_image_mode=False, model_complexity=1, smooth_landmarks=True,
refine_face_landmarks=False, enable_segmentation=False, min_detection_confidence=0.5,
min_tracking_confidence=0.5`.

Path conventions:
- `MP/` = `mediapipe/` in https://github.com/google-ai-edge/mediapipe at tag `v0.10.21`.
- `GIT:` = also a path in that repository at `v0.10.21` (outside `mediapipe/modules`, `calculators/util|tensor|image`).
- `PY/` = `mediapipe/python/` in the installed mediapipe 0.10.21 wheel.

`file:line` numbers are from the files as checked out. Model I/O shapes were read directly from the shipped `.tflite` flatbuffers in `site-packages/mediapipe/modules/`.

Notation: image size is `(W, H)` (MediaPipe's `IMAGE_SIZE` pair is `(width, height)`). A **NormRect** is
`(xc, yc, w, h, rot)`: center and size are normalized by W and H respectively, and `rot` is in radians.
All rect and landmark arithmetic in MediaPipe is **float32** unless noted otherwise. The OneEuro filter internals are float64.

```
NormalizeRadians(a) = a - 2π * floor((a + π) / (2π))          # result in [-π, π)
  MP/calculators/util/detections_to_rects_calculator.h:103-105
  MP/calculators/util/rect_transformation_calculator.cc:35-37
```

---------------------------------------------------------------------------------------------------

## 0. Top-level graph and per-frame data flow

`MP/modules/holistic_landmark/holistic_landmark_cpu.pbtxt` (node lines 107-147 in the file):

```
pose_landmarks, pose_world_landmarks  = PoseLandmarkCpu(image)                   # §1-§4
left_hand, right_hand                 = HandLandmarksLeftAndRightCpu(image, pose_landmarks)   # §5, §7
face_landmarks_from_pose              = pose_landmarks[0:11]   (SplitNormalizedLandmarkList, line 134)
face_landmarks                        = FaceLandmarksFromPoseCpu(image, face_landmarks_from_pose)  # §6
```

- The hands and the face consume the **smoothed** `pose_landmarks`, i.e. the output of PoseLandmarkFiltering, including the smoothed visibility.
- If there are no pose landmarks on a frame, there are **no hands and no face** on that frame either: every downstream input is empty.

Python side (`PY/solution_base.py:312-314`, `PY/solutions/holistic.py:118-129`):
- The packet timestamp is a simulated `t_k = 33333 * k` µs for the k-th `process()` call (k = 1, 2, …). It is **independent of the real fps**.
- `use_prev_landmarks = True`.
- `smooth_landmarks = True`.
- `min_detection_confidence` → `PoseDetectionCpu/TensorsToDetectionsCalculator.min_score_thresh`.
- `min_tracking_confidence` → the pose-presence `ThresholdingCalculator.threshold`.
- Nothing else is parameterised; the hand, face and face-detector thresholds are hard-coded.

### Framework semantics you must reproduce
These are standard MediaPipe calculator semantics, reproduced from the framework sources/comments.
- **Empty packets.** If a stream has no packet at timestamp t, a calculator whose *all* inputs are empty at t is **not invoked** (default input stream handler, no `SetProcessTimestampBounds`). A calculator with at least one non-empty input *is* invoked, and it sees the others as empty.
  - This matters for filter resets (§4).
- **PreviousLoopbackCalculator** (`GIT:calculators/core/previous_loopback_calculator.cc:26-41`): at frame t, `PREV_LOOP` = the LOOP packet of frame t-1. It is empty if frame t-1 produced no LOOP packet; on the first frame it is empty.
- **GateCalculator**: passes its inputs iff `allow`.
  - With `DISALLOW:flag`, allow = !flag (`GIT:gate_calculator.cc:176-183`).
  - `empty_packets_as_allow: true` means "allow when the control packet is empty".
- **ThresholdingCalculator**: `flag = (double)value > threshold`, a **strict** `>` (`MP/calculators/util/thresholding_calculator.cc:123-124`).
- **MergeCalculator**: the first non-empty input wins.

---------------------------------------------------------------------------------------------------

## 1. Pose detection (runs only when there is no tracked ROI from the previous frame)

### 1.1 Gating (`MP/modules/pose_landmark/pose_landmark_cpu.pbtxt`)
```
prev_roi = PreviousLoopback(LOOP = pose_rect_from_landmarks)      # line ~243-250
prev_roi = Gate(prev_roi, allow = use_prev_landmarks = True)      # line ~120-130
run_detection = (prev_roi is empty)                               # Gate DISALLOW:presence, empty_packets_as_allow (151-161)
if run_detection:
    dets = PoseDetectionCpu(image)                                # §1.2-1.5
    det  = dets[0] if dets nonempty else None                     # SplitDetectionVector ranges {0,1} element_only (line 178)
    roi  = PoseDetectionToRoi(det) if det else None               # §1.6
else:
    roi  = prev_roi                                               # MergeCalculator (lines ~201-206)
```
- If `roi` is None, the frame has no pose. Consequently there are no hands and no face, and the PREV_LOOP for the next frame is empty, so detection runs again on the next frame.
- There is **no** periodic re-detection. Tracking continues as long as the pose landmark model's presence score is above 0.5 (§2.4).

### 1.2 Preprocessing
`MP/modules/pose_detection/pose_detection_cpu.pbtxt:44-58`

| Setting | Value |
|---|---|
| Output tensor | 224×224 |
| `keep_aspect_ratio` | true (letterbox) |
| Float range | [-1, 1] |
| Border | `BORDER_ZERO` |
| ROI | none (whole image) |

The algorithm is shared by every ImageToTensor in the graph:

```
# ImageToTensorCalculator::Process  MP/calculators/tensor/image_to_tensor_calculator.cc:190-260
roi = RotatedRect(cx = xc*W, cy = yc*H, w = w*W, h = h*H, rot)      # GetRoi, image_to_tensor_utils.cc:37-51
      (no rect: cx=W/2, cy=H/2, w=W, h=H, rot=0)
# PadRoi (image_to_tensor_utils.cc:53-89), only if keep_aspect_ratio:
ta = Ht/Wt ; ra = roi.h/roi.w
if ta > ra:  roi.h = roi.w*ta ; pad_v = (1 - ra/ta)/2 ; pad_h = 0
else:        roi.w = roi.h/ta ; pad_h = (1 - ta/ra)/2 ; pad_v = 0
letterbox_padding = [pad_h, pad_v, pad_h, pad_v]           # [left, top, right, bottom], normalized to the tensor
```
- The padding only enlarges the ROI; the center is unchanged.
- The CPU converter is OpenCV (`image_to_tensor_converter_opencv.cc:133-171`):

```
corners = cv::boxPoints(RotatedRect((cx,cy),(w,h), rot*180/π))
dst corners = (0,Ht),(0,0),(Wt,0),(Wt,Ht)
M = getPerspectiveTransform(corners -> dst) ; warpPerspective(src_uint8_RGB, M, (Wt,Ht), INTER_LINEAR, border)
tensor = warped_uint8 * scale + offset      # scale=(max-min)/255, offset=min ; float32
```
- Interpolation: **`cv::INTER_LINEAR`**, the default argument (`image_to_tensor_converter_opencv.h:31`).
- Border:
  - `BORDER_ZERO` maps to `cv::BORDER_CONSTANT` with value 0. After normalization a zero pixel becomes -1 in [-1,1] or 0 in [0,1].
  - `BORDER_UNSPECIFIED` (the default) maps to **`BORDER_REPLICATE`** (`image_to_tensor_utils.cc:229-240`).

**Equivalent sampling formula.** Use this for a `grid_sample` implementation. For an output pixel `(u, v)` with integer u in [0, Wt-1] and v in [0, Ht-1], the source location in OpenCV pixel coordinates (pixel centres at integers) is:
```
dx = (u / Wt - 0.5) * roi.w ;  dy = (v / Ht - 0.5) * roi.h
px = cx + cos(rot)*dx - sin(rot)*dy
py = cy + sin(rot)*dx + cos(rot)*dy
```
- Note: this is **not** the half-pixel convention. Output pixel u=0 maps exactly onto the ROI edge, and u=Wt would map onto the opposite edge.
- With `grid_sample(align_corners=False)`, use `gx = (2*px + 1)/W - 1` and `gy = (2*py + 1)/H - 1`.
- **Fidelity details:**
  1. The warp is done on the **uint8** image and rounded to uint8 *before* the conversion to float. To match, round the bilinear result to an integer in [0,255] before normalizing.
  2. OpenCV's 8-bit `warpPerspective`/`remap` uses fixed point: coordinates are quantized to 1/32 pixel (`INTER_TAB_SIZE=32`) and weights to 1/32768 (`INTER_REMAP_COEF_BITS=15`). Expect about ±1 LSB differences against float bilinear. *(From OpenCV internals, not from the MediaPipe checkout.)*
  3. M is computed in double from float32 corners.
- Pose detection example on a 696×956 image: the ROI becomes 956×956 centred at (348, 478), `pad_h = (1 - 696/956)/2 = 0.13598`, and `pad_v = 0`.

### 1.3 Model
- Model: `pose_detection.tflite`.
- Input: `[1,224,224,3]`, RGB, float in [-1,1].
- Output 0: `Identity [1,2254,12]` (raw boxes and keypoints).
- Output 1: `Identity_1 [1,2254,1]` (raw score logits).

The TensorsToDetections default mapping is tensor 0 = boxes and tensor 1 = scores. Check the output order of your ONNX conversion by shape.

### 1.4 Anchors
SsdAnchorsCalculator, `pose_detection_cpu.pbtxt:86-103`. The algorithm is in `GIT:calculators/tflite/ssd_anchors_calculator.cc:43-51,265-372`.

| Option | Value |
|---|---|
| `num_layers` | 5 |
| `min_scale` | 0.1484375 |
| `max_scale` | 0.75 |
| `input_size_height`, `input_size_width` | 224, 224 |
| `anchor_offset_x`, `anchor_offset_y` | 0.5, 0.5 |
| `strides` | [8, 16, 32, 32, 32] |
| `aspect_ratios` | [1.0] |
| `fixed_anchor_size` | true |
| `interpolated_scale_aspect_ratio` | default **1.0** (`ssd_anchors_calculator.proto:60`) |
| `reduce_boxes_in_lowest_layer` | false |

Because `interpolated_scale_aspect_ratio > 0`, every layer contributes 2 anchors per cell. Layers with equal stride are merged.

```
anchors = []
layer = 0
while layer < 5:
    n = 0 ; last = layer
    while last < 5 and strides[last] == strides[layer]:  n += 2 ; last += 1     # 1 aspect + 1 interpolated per layer
    fm = ceil(224 / strides[layer])
    for y in range(fm): for x in range(fm): for a in range(n):
        anchors.append((x_c=(x+0.5)/fm, y_c=(y+0.5)/fm, w=1, h=1))
    layer = last
```
- Stride 8: fm=28, 2 per cell, 1568 anchors.
- Stride 16: fm=14, 2 per cell, 392 anchors.
- Stride 32: fm=7, 6 per cell (three layers × 2), 294 anchors.
- Total: **2254**.
- The order is y-major, then x, then anchor id within the cell. Anchor w and h are 1, so only the centres matter.

### 1.5 Decoding, score and NMS
Options at `pose_detection_cpu.pbtxt:114-132`:

| Option | Value |
|---|---|
| `num_classes` | 1 |
| `num_boxes` | 2254 |
| `num_coords` | 12 |
| `box_coord_offset` | 0 |
| `keypoint_coord_offset` | 4 |
| `num_keypoints` | 4 |
| `num_values_per_keypoint` | 2 |
| `sigmoid_score` | true |
| `score_clipping_thresh` | 100.0 |
| `reverse_output_order` | true, so the box format is **XYWH** (`tensors_to_detections_calculator.cc:131-138`) |
| `x_scale`, `y_scale`, `w_scale`, `h_scale` | 224.0 |
| `min_score_thresh` | 0.5 (= `min_detection_confidence`) |
| `apply_exponential_on_box_size` | false (default) |

```
# DecodeBoxes  tensors_to_detections_calculator.cc:786-871   (all values relative to the 224 letterboxed tensor)
r = raw[i]  (12 floats) ; A = anchor[i]
xc = r[0]/224*A.w + A.x_c ; yc = r[1]/224*A.h + A.y_c
w  = r[2]/224*A.w         ; h  = r[3]/224*A.h
box = (ymin=yc-h/2, xmin=xc-w/2, ymax=yc+h/2, xmax=xc+w/2)
kp_k = (r[4+2k]/224*A.w + A.x_c , r[5+2k]/224*A.h + A.y_c),  k=0..3
# score  lines 425-445
s = clip(raw_score[i], -100, 100) ; s = 1/(1+exp(-s))
keep iff s >= 0.5          # ConvertToDetection line 932: dropped if s < thresh
drop if width<0 or height<0 or NaN   # lines 893-901
```

NMS options at `pose_detection_cpu.pbtxt:140-146`:

| Option | Value |
|---|---|
| `min_suppression_threshold` | 0.3 |
| `overlap_type` | `INTERSECTION_OVER_UNION` |
| `algorithm` | **WEIGHTED** |
| `max_num_detections` | -1 (default) |
| `min_score_threshold` | -1 (default) |

```
# WeightedNonMaxSuppression  MP/calculators/util/non_max_suppression_calculator.cc:336-420
order = sort detections by score desc        # std::sort (not stable)
out = []
while remaining:
    top = remaining[0]
    cand   = [d in remaining with IoU(d.box, top.box) >  0.3]    # includes top itself
    remain = [d in remaining with IoU(d.box, top.box) <= 0.3]
    det = copy(top)                                              # score stays top.score
    if cand:  S = Σ s_c
              xmin = Σ s_c*xmin_c / S ; ymin = Σ s_c*ymin_c / S
              xmax = Σ s_c*(xmin_c+w_c)/S ; ymax = Σ s_c*(ymin_c+h_c)/S
              kp_k = Σ s_c*kp_k,c / S
              det.box = (xmin, ymin, xmax-xmin, ymax-ymin)
    out.append(det)
    if len(remain) == len(remaining): break
    remaining = remain
IoU = inter / (area1 + area2 - inter), 0 if no intersection     # lines 80-100
```
- Letterbox removal (`MP/calculators/util/detection_letterbox_removal_calculator.cc:103-140`):
  - `x' = (x - pad_l)/(1 - pad_l - pad_r)` and `y' = (y - pad_t)/(1 - pad_t - pad_b)`, applied to the box xmin/ymin and to the keypoints.
  - Width and height are divided by the same denominators.
- The pipeline uses **only `dets[0]`**, the highest-scoring weighted detection.
- Keypoint semantics:
  - kp0 = mid-hip (body centre).
  - kp1 = the full-body size/rotation point (towards the top of the head).
  - kp2 and kp3 are unused.

### 1.6 Detection → ROI
`MP/modules/pose_landmark/pose_detection_to_roi.pbtxt:17-44`. The image size is that of the current frame.

```
# AlignmentPointsRectsCalculator  MP/calculators/util/alignment_points_to_rects_calculator.cc:66-100
p0 = (kp0.x*W, kp0.y*H) ; p1 = (kp1.x*W, kp1.y*H)
box = 2*|p1 - p0|
rect.xc = p0.x/W ; rect.yc = p0.y/H ; rect.w = box/W ; rect.h = box/H
# rotation  detections_to_rects_calculator.cc:308-327, target = 90° (π*90/180)
rect.rot = NormalizeRadians(π/2 - atan2(-(p1.y-p0.y), p1.x-p0.x))
# RectTransformation  scale_x=scale_y=1.25, square_long=true, no shift  (pose_detection_to_roi.pbtxt:40-42)
```

RectTransformation for a NormRect (`MP/calculators/util/rect_transformation_calculator.cc:206-245`). It is used everywhere in the graph:
```
w, h, rot = rect.w, rect.h, rect.rot
if rot == 0:  xc += w*shift_x ; yc += h*shift_y
else:
  xc += (W*w*shift_x*cos(rot) - H*h*shift_y*sin(rot)) / W
  yc += (W*w*shift_x*sin(rot) + H*h*shift_y*cos(rot)) / H
if square_long: L = max(w*W, h*H); w = L/W; h = L/H
w *= scale_x ; h *= scale_y
```
The shift is computed with the *pre-square, pre-scale* w and h. The rotation is unchanged.

---------------------------------------------------------------------------------------------------

## 2. Pose landmarks (pose_landmark_full)

### 2.1 Crop
`MP/modules/pose_landmark/pose_landmark_by_roi_cpu.pbtxt:109-125`

| Setting | Value |
|---|---|
| Output tensor | 256×256 |
| `keep_aspect_ratio` | true |
| Float range | [0, 1] |
| Border | not set, so **BORDER_REPLICATE** |
| Interpolation | INTER_LINEAR (OpenCV, §1.2) |
| ROI | the pose ROI from §1.6 or §3 |

- The ROI is square in pixels, so the letterbox padding is 0, up to float rounding of about 1e-7.
- The padding is still output and applied in §2.5. Implement it generally.

### 2.2 Model
`pose_landmark_full.tflite`. Model selection: `pose_landmark_model_loader.pbtxt:21-37` (complexity 1 = full).

| Index | Name | Shape | Meaning |
|---|---|---|---|
| out0 | `Identity` | [1,195] | 39 landmarks × 5 (x, y, z, visibility logit, presence logit), in pixels of the 256 input |
| out1 | `Identity_1` | [1,1] | pose presence flag. **Sigmoid is already inside the model**: the op chain is CONV → LOGISTIC → RESHAPE, verified in the flatbuffer. |
| out2 | `Identity_2` | [1,256,256,1] | segmentation logits (unused, gated off) |
| out3 | `Identity_3` | [1,64,64,39] | heatmap logits, HWC |
| out4 | `Identity_4` | [1,117] | 39 world landmarks × 3 (x, y, z, metres) |

Split: `tensors_to_pose_landmarks_and_segmentation.pbtxt:90-105` (0 = landmarks, 1 = flag, 2 = segmentation, 3 = heatmap, 4 = world).

### 2.3 Decode
`tensors_to_pose_landmarks_and_segmentation.pbtxt`
```
presence_score = out1[0]                                   # TensorsToFloats, activation NONE (proto default)
if not (presence_score > min_tracking_confidence=0.5):     # Thresholding line 125, strict >
    -> NO landmarks this frame (all of landmarks/aux/world gated, lines 131-141)
# TensorsToLandmarks lines 150-162: num_landmarks 39, input 256x256, visibility_activation SIGMOID, presence_activation SIGMOID
# MP/calculators/tensor/tensors_to_landmarks_calculator.cc:152-201 ; normalize_z default 1.0
for i in 0..38:
    x = out0[5i]/256 ; y = out0[5i+1]/256 ; z = out0[5i+2]/256 / 1.0
    vis = sigmoid(out0[5i+3]) ; pres = sigmoid(out0[5i+4])
```

### 2.4 Heatmap refinement
RefineLandmarksFromHeatmapCalculator, `tensors_to_pose_landmarks_and_segmentation.pbtxt:166-176`:
- `kernel_size: 7` in the source pbtxt, **but the shipped v0.10.21 wheel behaves as kernel_size 9** (verified empirically against reference output: frame-0 body error 0.004 px with 9 vs 0.18 px with 7).
- `min_confidence_to_refine` is the default **0.5** (`refine_landmarks_from_heatmap_calculator.proto:25-28`).
- `refine_presence` and `refine_visibility` are false.

The refinement is applied to all 39 landmarks, in normalized crop coordinates (before letterbox removal and projection):
```
# MP/calculators/util/refine_landmarks_from_heatmap_calculator.cc:106-164 ; hm: 64x64x39 HWC
for i in 0..38:
    col = int(x_i * 64) ; row = int(y_i * 64)          # C++ int cast = truncation toward 0 (x in (-1/64,0) -> 0 !)
    if col<0 or col>=64 or row<0 or row>=64: continue
    off = 3
    rows = [max(0,row-3), min(64,row+4)) ; cols = [max(0,col-3), min(64,col+4))
    c = sigmoid(hm[r, q, i]) over the window
    S = Σc ; mx = max(c) (init 0) ; wc = Σ q*c ; wr = Σ r*c
    if mx >= 0.5 and S > 0:  x_i = wc/64/S ; y_i = wr/64/S     # NOTE: pixel index /64, no +0.5
```
Visibility and presence are unchanged by this step.

Then the landmarks are split (lines 180-190):
- `landmarks` = indices 0..32 (33 landmarks).
- `auxiliary_landmarks` = indices 33..34 (2 points: 33 = body centre, 34 = size/rotation point).
- Indices 35..38 are discarded.

### 2.5 Projection to image
`pose_landmarks_and_segmentation_inverse_projection.pbtxt:61-97`. This applies to both `landmarks` and `aux`.
```
# LandmarkLetterboxRemoval  landmark_letterbox_removal_calculator.cc:123-130
x = (x - pad_l)/(1 - pad_l - pad_r) ; y = (y - pad_t)/(1 - pad_t - pad_b) ; z = z/(1 - pad_l - pad_r)
# LandmarkProjectionCalculator, NORM_RECT without IMAGE_DIMENSIONS path  landmark_projection_calculator.cc:193-211
xr = x - 0.5 ; yr = y - 0.5
x' = (cos(rot)*xr - sin(rot)*yr) * roi.w + roi.xc
y' = (sin(rot)*xr + cos(rot)*yr) * roi.h + roi.yc
z' = z * roi.w                          # roi.w normalized by image width
```
- Visibility and presence are copied through unchanged.
- The rotation is applied in normalized space and is correct only for ROIs that are square in pixels. All ROIs in this graph are square.
- The `roi` used here is the ROI *before* PadRoi enlargement. With square ROIs the two are identical.

**World landmarks** (lines 198-232 and `world_landmark_projection_calculator.cc:91-96`):
```
wl = out4 reshaped (39,3) -> first 33   (TensorsToLandmarks num_landmarks 39, num_dims = 117/39 = 3, no activation)
wx' = cos(rot)*wx - sin(rot)*wy ; wy' = sin(rot)*wx + cos(rot)*wy ; wz' = wz      # rotation only, no scale/translation
visibility, presence := copied from the (unsmoothed, post-sigmoid) normalized landmarks 0..32   (VisibilityCopy, copy_visibility & copy_presence)
```

---------------------------------------------------------------------------------------------------

## 3. Tracking: ROI for the next frame
`pose_landmark_cpu.pbtxt` lines ~227-250 and `pose_landmarks_to_roi.pbtxt`.

```
pose_rect_from_landmarks = PoseLandmarksToRoi(filtered_aux_landmarks, image_size)
    # LandmarksToDetection: keypoints = the 2 aux points
    # AlignmentPointsRects: start kp 0 (aux 33, centre), end kp 1 (aux 34), target 90°   (lines 24-35)
    # RectTransformation scale 1.25/1.25, square_long             (lines 39-49)
    -> identical formulas to §1.6 with kp0 := aux[0], kp1 := aux[1]
```
- The ROI is built from the **smoothed** aux landmarks (§4.4) and fed back through PreviousLoopback for frame t+1.
- If frame t has no landmarks (presence ≤ 0.5 or no detection), there is no ROI, so frame t+1 runs detection.
- Presence threshold: 0.5, strict `>`.

Summary state machine per video stream:
```
state.prev_pose_roi = None
frame t:
  roi = state.prev_pose_roi if not None else detect_roi(image)   # may be None
  if roi is None: pose=None; state.prev_pose_roi=None; (filters reset per §4) ; no hands/face ; continue
  run landmark model -> if presence <= 0.5: same as above (prev_pose_roi=None)
  else: smooth (§4) ; state.prev_pose_roi = roi_from(filtered aux)
```

---------------------------------------------------------------------------------------------------

## 4. Smoothing
Source: `MP/modules/pose_landmark/pose_landmark_filtering.pbtxt`. Pose only: **there is no smoothing of hands or face** in the holistic graph. HandLandmarkCpu and FaceLandmarkCpu have no smoothing nodes, and holistic adds none.

All filters are enabled (SwitchContainer `ENABLE: smooth_landmarks = True` selects `contained_node[1]`).

Inputs to the filtering subgraph, all unsmoothed, for the current frame:
- `landmarks` (33 normalized),
- `aux` (2 normalized),
- `world` (33),
- `image_size`.

### 4.1 Object-scale ROI (lines 39-58)
```
roi_s = AlignmentPointsRects(LandmarksToDetection(UNFILTERED aux), start 0, end 1, target 90°)   # no 1.25 scaling
object_scale = (roi_s.w*W + roi_s.h*H)/2 = 2*|aux1_px - aux0_px|      # GetObjectScale(NormalizedRect), landmarks_smoothing_calculator_utils.cc:307-313
```

### 4.2 Visibility low-pass
`VisibilitySmoothingCalculator low_pass_filter alpha 0.1`. Applied to:
- landmarks (lines 62-90),
- world (lines 134-162),
- aux (lines 208-219).

```
# LowPassFilter (GIT:util/filtering/low_pass_filter.cc:26-42), float32 state
v_out = v_in                          on first sample after (re)creation
v_out = 0.1*v_in + 0.9*v_out_prev     otherwise
```
- **Reset semantics:** `visibility_smoothing_calculator.cc:206-213` resets when the input is empty. However, this node's *only* input is the landmarks stream, so on frames without landmarks it is **not invoked** and therefore **never resets** (see §0).
- Consequently, the visibility EMA carries over across pose-loss gaps.
- The world visibility filter runs on the same input visibilities with the same alpha, so in practice it is identical to the landmark visibility.
- *Uncertain:* this rests on framework scheduling semantics. Verify empirically: after a pose-loss gap, the first output visibility should not equal sigmoid(raw).

### 4.3 Pose landmarks OneEuro (lines 93-131)
- Parameters: `min_cutoff 0.05`, `beta 80.0`, `derivate_cutoff 1.0`.
- `frequency` default 30, `min_allowed_object_scale` default 1e-6, `disable_value_scaling` false (`landmarks_smoothing_calculator.proto:55-80`).
- The node inputs are NORM_LANDMARKS, IMAGE_SIZE and OBJECT_SCALE_ROI. IMAGE_SIZE is present every frame, so the node **is** invoked on frames without landmarks and **resets** (clears all filters) (`landmarks_smoothing_calculator.cc:56-61`).

```
# landmarks_smoothing_calculator.cc:63-91 ; utils.cc:179-215, 254-305
P = [(x*W, y*H, z*W) for each landmark]                     # to pixels; z scaled by WIDTH
value_scale = 1/object_scale  (if object_scale < 1e-6: output = input, no filtering)
for each landmark i, coordinate c in {x,y,z}: out = OneEuro[i][c].Apply(t, value_scale, P[i][c])
back to normalized: x/W, y/H, z/W ; visibility = the low-passed visibility (4.2) passes through untouched
```

OneEuro (`GIT:util/filtering/one_euro_filter.cc:15-59`). Arithmetic is double, the LowPassFilter state is float32, and `alpha` is passed as float:
```
init: freq = 30.0 ; last_time = -1 ; x_lp, dx_lp uninitialized
Apply(t_ns, scale, value):
  if last_time >= t_ns: return value
  if last_time != 0 and t_ns != 0: freq = 1/((t_ns - last_time)*1e-9)
  last_time = t_ns
  dvalue = x_lp.initialized ? (value - x_lp.last_raw) * scale * freq : 0
  edvalue = dx_lp.apply(dvalue, alpha(derivate_cutoff))
  cutoff = min_cutoff + beta*|edvalue|
  return x_lp.apply(value, alpha(cutoff))
alpha(fc) = 1/(1 + tau/te),  te = 1/freq, tau = 1/(2π fc)
LowPass.apply(v, a): first call -> out = v ; else out = a*v + (1-a)*prev_out ; last_raw = v ; prev_out = out   (float32)
```
- Timestamps are those of §0: t_k = 33333·k µs. In steady state the frequency is 1/0.033333 s = 30.0003 Hz. The real fps is ignored.
- On the first call after a (re)creation, the output equals the input and the derivative state is set to 0. The freq computed from last_time = -1 is irrelevant because dvalue = 0 on that call.

### 4.4 Aux landmarks OneEuro (lines 222-247)
- Parameters: `min_cutoff 0.01`, `beta 10.0`, `derivate_cutoff 1.0`.
- Same scale ROI, same pixel conversion, and same reset-on-empty as §4.3 (IMAGE_SIZE is connected).
- The aux visibility filter is a low-pass with alpha 0.1. It is never reset, and its output is unused downstream.
- The output, `filtered_aux_landmarks`, drives the next-frame ROI (§3).

### 4.5 World landmarks OneEuro (lines 165-205)
- Parameters: `min_cutoff 0.1`, `beta 40.0`, `derivate_cutoff 1.0`, **`disable_value_scaling: true`**, so value_scale = 1 and there is no object-scale check.
- The values (metres) are filtered directly, with no unit conversion. This is the LANDMARKS path (`landmarks_smoothing_calculator.cc:92-106`).
- The node's only input is LANDMARKS, so on frames without landmarks it is not invoked and is **not reset**. *(Same caveat as §4.2.)*
- After a gap, the first frame uses `freq = 1/gap_duration` and `dvalue = (value - last_raw)*freq`.

---------------------------------------------------------------------------------------------------

## 5. Hands from pose
Files: `MP/modules/holistic_landmark/hand_landmarks_left_and_right_cpu.pbtxt` and `hand_landmarks_from_pose_cpu.pbtxt`.

Each side is run independently with its own tracking state.

### 5.1 Pose-landmark selection (§7 has the left/right mapping)
- Left: pose[15, 17, 19] = [wrist, pinky, index] (lines 25-37).
- Right: pose[16, 18, 20] (lines 52-64).
- These are the **smoothed** pose landmarks with the smoothed visibility.

### 5.2 Visibility gate
`hand_visibility_from_hand_landmarks_from_pose.pbtxt:14-44`
```
if not (wrist.visibility > 0.1): no hand ROI, so no hand landmarks this frame
                                  (prev-hand state still advances: LOOP empty -> next frame no tracking)
```

### 5.3 ROI from pose
`hand_landmarks_from_pose_to_recrop_roi.pbtxt`, `HandDetectionsFromPoseToRectsCalculator` (`MP/modules/holistic_landmark/calculators/hand_detections_from_pose_to_rects_calculator.cc:86-153`):
```
w=(x0*W,y0*H) wrist ; p=(x1*W,y1*H) pinky ; i=(x2*W,y2*H) index
m = (2*i + p)/3                         # estimated middle finger
box = 2*|m - w|
rect = (m.x/W, m.y/H, box/W, box/H)
rect.rot = NormalizeRadians(π/2 - atan2(-(m.y - w.y), m.x - w.x))
RectTransformation: scale 2.7/2.7, shift_y -0.1, square_long   (lines 32-44)
-> hand_roi_from_pose
```

### 5.4 Hand recrop model
`hand_recrop_by_roi_cpu.pbtxt`

- Crop: ImageToTensor **256×256**, `keep_aspect_ratio` true, range [0,1], border not set (so REPLICATE), ROI = hand_roi_from_pose (lines 17-37).
- Model: `hand_recrop.tflite`. Input `[1,256,256,3]`. Output: `output_crop [1,1,1,4]`, i.e. 2 points × (x, y) in pixels of the 256 crop.

```
# TensorsToLandmarks num_landmarks 2, input 256x256 (lines 54-66): num_dims = 4/2 = 2 -> z unset (0)
c0 = (o[0]/256, o[1]/256) ; c1 = (o[2]/256, o[3]/256)
LetterboxRemoval (pad 0 for square roi) ; LandmarkProjection with hand_roi_from_pose (§2.5 formula)
AlignmentPointsRects start 0, end 1, target -90°  (lines 103-116):
    center = c0 ; box = 2*|c1_px - c0_px|
    rot = NormalizeRadians(-π/2 - atan2(-(y1_px - y0_px), x1_px - x0_px))     # (x,y)_px = (x*W, y*H)
RectTransformation scale 1.0/1.0, shift_y -0.1, square_long  (lines 124-137)
-> hand_roi_from_recrop
```
The recrop model has no presence or score output, so it never rejects a hand. Rejection happens only in the landmark model (§5.6).

### 5.5 Hand tracking
`hand_tracking.pbtxt`. It is RoiTrackingCalculator (`MP/modules/holistic_landmark/calculators/roi_tracking_calculator.cc:42-358`).

```
prev_lms  = this side's hand landmarks at frame t-1 (PreviousLoopback MAIN:image_size, LOOP:hand_landmarks; lines 20-30)
prev_rect = HandLandmarksToRoi(prev_lms, current image_size)       # §5.7, lines 33-38
if recrop_rect empty: output nothing (no hand)                      # cc:286-288
elif prev_rect empty: tracking_roi = recrop_rect                    # cc:292-296
else:
  keep = RectReq(prev_rect, recrop_rect, rot_deg=40.0, translation=0.2, scale=0.4)          # pbtxt 52-57
     and LandmarksReq(prev_lms, recrop_rect, recrop_rect_margin=-0.1)                       # pbtxt 58-60
  (no IoU requirement: iou_requirements not set)
  tracking_roi = prev_rect if keep else recrop_rect
```

RectRequirements (`cc:71-153`). Here φ = -recrop.rot, and the coordinates are rotated about the image origin in pixels:
```
P(r) = (r.xc*W*cosφ - r.yc*H*sinφ , r.xc*W*sinφ + r.yc*H*cosφ)
d_rot = deg(prev.rot) - deg(recrop.rot) ; wrap to [-180,180] (single ±360 step) ; |d_rot| > 40  -> fail
|P(prev).x - P(recrop).x| > recrop.w*W*0.2 -> fail ; same for y with recrop.h*H*0.2
|prev.w*W - recrop.w*W| > recrop.w*W*0.4 -> fail ; same for heights
```
`abs` here is the float overload, because `<math.h>` is included.

LandmarksRequirements (`cc:157-202`):
```
rect in rotated frame: center P(recrop), width recrop.w*W*(1-0.1), height recrop.h*H*(1-0.1)
every prev landmark (rotated the same way, from x*W, y*H) must lie STRICTLY inside, else fail
```

### 5.6 Hand landmark model
`MP/modules/hand_landmark/hand_landmark_cpu.pbtxt`. The model is **hand_landmark_full.tflite**: holistic does not pass MODEL_COMPLEXITY, so the default `select: 1` applies (`hand_landmark_model_loader.pbtxt:21,37`).

- Crop: ImageToTensor **224×224**, keep_aspect true, [0,1], border REPLICATE, ROI = tracking_roi (lines 39-57).
- Outputs:

| Index | Name | Shape | Meaning |
|---|---|---|---|
| out0 | `Identity` | [1,63] | 21 × (x, y, z) in pixels of 224 |
| out1 | `Identity_1` | [1,1] | hand presence; **sigmoid already in the model** (LOGISTIC op) |
| out2 | `Identity_2` | [1,1] | handedness; sigmoid in the model; **unused by holistic** |
| out3 | `Identity_3` | [1,63] | world landmarks (unused by holistic) |

```
if not (out1 > 0.5): no hand landmarks (thresholding lines 111-120, strict >)
x = o[3i]/224 ; y = o[3i+1]/224 ; z = o[3i+2]/224/0.4        # normalize_z 0.4 (lines 155-169)
LetterboxRemoval (pad 0) ; LandmarkProjection with tracking_roi (§2.5):  z' = z * roi.w
```
- No visibility or presence fields are set: the tensor has 3 dimensions.
- The result is this side's `hand_landmarks` output and also the LOOP input for the next frame.

### 5.7 HandLandmarksToRoi (previous-frame tracking ROI)
Files: `hand_landmarks_to_roi.pbtxt` and `MP/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc:39-131,175-191`.

```
L = prev landmarks at indices [0,1,2,3,5,6,9,10,13,14,17,18]  (12 points; partial index p)
w0 = L[0] px ; mid = ((L[4]+L[8])/2 + L[6])/2 px             # partial 4=lm5(index MCP),6=lm9(middle MCP),8=lm13(ring MCP)
rot = NormalizeRadians(π/2 - atan2(-(mid.y - w0.y), mid.x - w0.x))
axis-aligned centre (normalized): ac = ((max x + min x)/2, (max y + min y)/2)   [max init = FLT_MIN (≈1e-38, not -inf!)]
rotate each point by -rot about ac in PIXELS: (dx,dy) = ((x-ac.x)*W, (y-ac.y)*H) -> projected
  [max init again FLT_MIN: if all projected x<0, max_x = ~0 -> quirk]
pc = ((max+min)/2 of projected)
centre_px = R(rot)*pc + (ac.x*W, ac.y*H)
rect = (centre_px.x/W, centre_px.y/H, (max_px.x-min_px.x)/W, (max_px.y-min_px.y)/H, rot)
RectTransformation scale 2.0/2.0, shift_y -0.1, square_long   (hand_landmarks_to_roi.pbtxt:24-37)
```
- The variable names in the C++ source say "PIP", but with the partial list the indices actually point at the MCPs.
- The FLT_MIN initialization quirk only matters in degenerate cases: all x<0 in the first loop, which means the hand is fully off-screen to the left.

---------------------------------------------------------------------------------------------------

## 6. Face from pose
File: `MP/modules/holistic_landmark/face_landmarks_from_pose_cpu.pbtxt`.

**Yes, holistic runs a face detector on every frame that has a pose**: the BlazeFace short-range model, restricted to the pose-derived face ROI.

### 6.1 Face ROI from pose
`face_landmarks_from_pose_to_recrop_roi.pbtxt`
```
F = smoothed pose landmarks 0..10 (nose, eyes, ears, mouth)
bbox = axis-aligned bbox of F in normalized coords (LandmarksToDetection; max init FLT_MIN quirk)
rect = (bbox centre, bbox w, bbox h)            # DetectionsToRects default mode = RectFromBox (detections_to_rects_calculator.cc:68-74,110-131)
rot  = NormalizeRadians(0 - atan2(-(y2_px - y5_px), x2_px - x5_px))     # start kp 5 (right eye), end kp 2 (left eye), target 0  (lines 28-33)
RectTransformation scale 3.0/3.0, square_long, no shift   (lines 39-49)
-> face_roi_from_pose
```
There is **no visibility gate** for the face. It always runs when a pose exists; a TODO in the source mentions adding one.

### 6.2 Face detection within that ROI
Files: `MP/modules/face_detection/face_detection_short_range.pbtxt:30-56` and `face_detection.pbtxt`.

- Crop: ImageToTensor 128×128, keep_aspect true, range [-1,1], **BORDER_ZERO**, ROI = face_roi_from_pose (`face_detection.pbtxt:41-59`).
- Model: `face_detection_short_range.tflite`. Input `[1,128,128,3]`. Outputs: `regressors [1,896,16]` and `classificators [1,896,1]`.
- Anchors:
  - `num_layers 4`, strides [8, 16, 16, 16], `min_scale 0.1484375`, `max_scale 0.75`, offset 0.5/0.5, aspect [1.0], `interpolated_scale_aspect_ratio 1.0`, `fixed_anchor_size true`, input 128×128.
  - Stride 8: 16×16×2 = 512. Stride 16: 8×8×6 = 384. Total **896**.
- Decode:
  - `num_coords 16`, `num_keypoints 6`, kp offset 4, `sigmoid_score`, clip 100, reverse_output_order (XYWH), all scales 128.0, **`min_score_thresh 0.5`** (fixed; not tied to min_detection_confidence).
  - Formulas as in §1.5.
- NMS: WEIGHTED, IoU, threshold 0.3 (`face_detection.pbtxt:144-155`).
- Projection back to image (`MP/calculators/util/detection_projection_calculator.cc:65-110`): DetectionProjection with the ImageToTensor MATRIX (`image_to_tensor_utils.cc:103-186`, built from the *padded* ROI). For keypoints:
  ```
  X = cx + cos(rot)*(x-0.5)*w - sin(rot)*(y-0.5)*h ;  Y = cy + sin(rot)*(x-0.5)*w + cos(rot)*(y-0.5)*h   (pixels) ; then /W, /H
  ```
- Face keypoints: 0 = right eye, 1 = left eye, 2 = nose tip, 3 = mouth, 4 = right ear tragion, 5 = left ear tragion.

### 6.3 Detection → face recrop ROI
`face_detection_front_detections_to_roi.pbtxt`
```
if no detections: face_roi_from_detection is EMPTY -> RoiTracking outputs nothing -> NO face landmarks this frame
d = detections[0]  (highest score after NMS)            # DetectionsToRects uses detections[0] (cc:258-270)
conversion_mode USE_KEYPOINTS: rect = bbox of all 6 keypoints (normalized)  (cc:46-66)
rot = NormalizeRadians(-atan2(-(kp1.y_px - kp0.y_px), kp1.x_px - kp0.x_px))   # start 0 right eye, end 1 left eye, target 0
RectTransformation scale 2.0/2.0, shift_y -0.1, square_long   (lines 35-48)
```

### 6.4 Face tracking
`face_tracking.pbtxt`. Same RoiTrackingCalculator as §5.5, with these parameters:

| Parameter | Value |
|---|---|
| `rotation_degrees` | 15.0 |
| `translation` | 0.1 |
| `scale` | 0.3 |
| `recrop_rect_margin` | -0.2 |
| IoU check | none |

```
prev_lms  = face landmarks (468) of frame t-1 (PreviousLoopback MAIN:image_size)
prev_rect = FaceLandmarksToRoi(prev_lms):                      # face_landmarks_to_roi.pbtxt
    bbox (axis-aligned, normalized) of all 468 lms ; rot from lm 33 -> lm 263, target 0
    RectTransformation scale 1.5/1.5, square_long, no shift
tracking_roi = recrop rect / prev_rect per the §5.5 rules (and nothing if the recrop rect is empty)
```

### 6.5 Face landmark model
`MP/modules/face_landmark/face_landmark_cpu.pbtxt`. With `WITH_ATTENTION = refine_face_landmarks = False`, the model is `face_landmark.tflite`.

- Crop: ImageToTensor **192×192**. `keep_aspect_ratio` is **not set (false)**, so the ROI is stretched (it is square anyway) and there is no letterbox. Range [0,1], border REPLICATE (lines 46-61).
- Outputs:
  - out0: `conv2d_21 [1,1,1,1404]` = 468 × (x, y, z) in pixels of 192.
  - out1: `conv2d_31 [1,1,1,1]` = face-flag **logit**. The model has no sigmoid here.

```
face_score = sigmoid(out1)                          # TensorsToFloats activation SIGMOID (lines 126-136)
if not (face_score > 0.5): no face landmarks        # lines 139-149
x = o[3i]/192 ; y = o[3i+1]/192 ; z = o[3i+2]/192   # tensors_to_face_landmarks.pbtxt:13-24, normalize_z 1
LandmarkProjection with tracking_roi (§2.5): z' = z*roi.w     # no letterbox removal node in this graph
```
- No visibility or presence fields are set.
- There is no smoothing.

---------------------------------------------------------------------------------------------------

## 7. Left/right assignment
- `left_hand_landmarks` is derived from pose landmarks 15, 17, 19 (the pose **LEFT** wrist, pinky and index). `right_hand_landmarks` is derived from 16, 18, 20 (`hand_landmarks_left_and_right_cpu.pbtxt:25-75`).
- "Left" means the subject's anatomical left in the pose model's convention. On a non-mirrored camera image, that hand usually appears on the image's right side.
- The hand model's handedness output is ignored, and there is no cross-check and no swapping.
- The two sides have fully independent tracking state (prev landmarks and ROI). Nothing prevents both sides from locking onto the same physical hand.

---------------------------------------------------------------------------------------------------

## 8. Python output format
`PY/solutions/holistic.py:136-167`

| Field | Type | Contents |
|---|---|---|
| `pose_landmarks` | NormalizedLandmarkList, 33 | `x, y` normalized to the image (can be <0 or >1); `z` normalized by image **width** (scale of x, origin about the hips, from the model); `visibility` = smoothed sigmoid. `presence` is **cleared** by Python (`ClearField('presence')`, line 163). Values are after OneEuro (§4.3). |
| `pose_world_landmarks` | LandmarkList, 33 | metres, hip-centred, rotated by the ROI rotation, OneEuro-smoothed (§4.5); `visibility` = smoothed (copied from the raw normalized visibility and then low-passed); presence cleared (line 166). |
| `face_landmarks` | NormalizedLandmarkList, 468 | `x, y, z`; z normalized by image width; no visibility or presence (reading them gives 0.0) |
| `left_hand_landmarks`, `right_hand_landmarks` | NormalizedLandmarkList, 21 | `x, y, z`; z = model_z/224/0.4 × roi.w; no visibility or presence |
| `segmentation_mask` | — | None (disabled) |

Any field is `None` when it is absent on that frame.

---------------------------------------------------------------------------------------------------

## 9. Complete per-frame pseudocode (one stream)

```
state: prev_pose_roi, pose_filters{lm_oneeuro, aux_oneeuro, world_oneeuro, vis_lp(lm), vis_lp(world), vis_lp(aux)},
       prev_hand[L], prev_hand[R], prev_face           (each a landmark list or None)
t_ns = 33333000 * k

def frame(img):                                   # img uint8 RGB HxWx3
  # --- pose ---
  roi = prev_pose_roi
  if roi is None:
     dets = pose_detect(img)                      # §1.2-1.5 ; letterbox 224, [-1,1], zero border
     roi = det_to_roi(dets[0]) if dets else None  # §1.6
  lm39 = None
  if roi is not None:
     t = crop(img, roi, 256, keep_aspect, [0,1], REPLICATE)
     out = pose_full(t)
     if out1 > 0.5:
        lm39 = decode(out0) ; lm39 = heatmap_refine(lm39, out3) ; project(lm39, roi)
        world = rotate(out4[:33], roi.rot) ; world.vis/pres = lm39[:33].vis/pres
  if lm39 is None:
     lm_oneeuro.reset() ; aux_oneeuro.reset()     # vis filters and world filter NOT reset (§4.2/4.5)
     prev_pose_roi = None ; prev_hand[*] = None ; prev_face = None
     return None, None, None, None, None
  lm, aux = lm39[:33], lm39[33:35]
  scale = 2*|aux1_px - aux0_px|                  # unfiltered aux
  lm.vis = vis_lp_lm(lm.vis) ; lm.xyz = lm_oneeuro(lm.xyz px, 1/scale, t_ns)
  aux = aux_oneeuro(aux px, 1/scale, t_ns)
  world.vis = vis_lp_world(world.vis) ; world.xyz = world_oneeuro(world.xyz, 1.0, t_ns)
  prev_pose_roi = aux_to_roi(aux)                 # scale 1.25 square_long, target 90°

  # --- hands ---  (side L uses pose[15,17,19], R uses pose[16,18,20]; smoothed lm incl. smoothed vis)
  for side:
     hand = None
     if wrist.vis > 0.1:
        r0 = hand_roi_from_pose(...)              # scale 2.7, shift_y -0.1, square_long, target 90°
        rc = recrop(img, r0)                      # 256 crop, 2 pts, target -90°, scale 1, shift_y -0.1, square_long
        rt = roi_tracking(prev_hand[side], rc, 40°, 0.2, 0.4, margin -0.1)
        out = hand_full(crop(img, rt, 224, keep_aspect, [0,1], REPLICATE))
        if out1 > 0.5: hand = project(decode(out0, z/224/0.4), rt)
     prev_hand[side] = hand

  # --- face ---
  face = None
  rf = face_roi_from_pose(lm[0:11])              # bbox, rot kp5->kp2 target 0, scale 3.0 square_long
  dets = blazeface_short(img, rf)                 # 128 crop, [-1,1], zero border, thresh 0.5, weighted NMS 0.3
  if dets:
     rc = face_det_to_roi(dets[0])                # kp bbox, rot kp0->kp1, scale 2.0, shift_y -0.1, square_long
     rt = roi_tracking(prev_face, rc, 15°, 0.1, 0.3, margin -0.2)   # prev rect: bbox 468, rot 33->263, scale 1.5
     out = face_mesh(crop(img, rt, 192, stretch, [0,1], REPLICATE))
     if sigmoid(out1) > 0.5: face = project(decode(out0, z/192), rt)
  prev_face = face
  return lm, world, hand[L], hand[R], face
```

Note on `prev_hand` and `prev_face` when there is no pose:
- In the real graph the PreviousLoopback for hands and face is driven by `image_size`, which exists every frame.
- On a frame without pose, the hand and face LOOP inputs are empty, so on the next frame `prev` is empty. The pseudocode above matches this.

---------------------------------------------------------------------------------------------------

## 10. Batching and fidelity notes

- **Sequential dependencies.** Everything is sequential in time within a stream:
  - the pose ROI depends on the smoothed aux landmarks of t-1;
  - the hand and face ROIs depend on the landmarks of t-1;
  - the filters are stateful.

  You can batch across streams (videos), or batch the per-frame models within one frame: L hand, R hand, recrop L/R, face det, face mesh. You **cannot** batch across frames of one stream without breaking tracking.
- **Precision.** Do the ROI math in float32, and keep the OneEuro LowPass state in float32 (alpha cast to float) while doing the other filter math in float64, to match bit-for-bit as closely as possible.
- **Crops.** The crops are the biggest source of drift: uint8 rounding, OpenCV fixed-point bilinear, the edge-mapping convention (§1.2) and the border mode. A half-pixel convention error shifts every landmark by about 0.5 crop pixels.
- **Weighted NMS order.** It depends on `std::sort` (not stable). Ties are rare in practice.

### Uncertain points
1. Visibility and world filters not resetting on empty frames (§4.2, §4.5). This is inferred from framework scheduling, not verified by running.
2. OpenCV fixed-point interpolation details. These are from OpenCV knowledge; that source is not in the checkout.
3. The SSD anchors, gate, loopback, OneEuro and LowPass sources were read via `git show v0.10.21` (outside the directories listed above).
4. The exact framework line numbers for the PreviousLoopback, Gate and Merge nodes in `pose_landmark_cpu.pbtxt` are approximate ("~"). All other numbers are exact.
