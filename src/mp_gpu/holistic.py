"""Batched GPU re-implementation of MediaPipe legacy Holistic (v0.10.21 CPU graph).

Config replicated: static_image_mode=False, model_complexity=1, smooth_landmarks=True,
refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5.
See HOLISTIC_SPEC.md for the algorithm (section numbers referenced below).

S independent video streams are processed in lock-step: every call to step() consumes one
frame per stream. All per-stream state (tracking ROIs, previous landmarks, smoothing filters)
lives in batched tensors, so the models run with batch ~S.
"""
import math, os
import numpy as np
import torch
import torch.nn.functional as F

from .tflite_torch import TFLiteModule
from .crop_triton import crop_triton, FrameTable, NV12Table

USE_TRITON = os.environ.get("HOLISTIC_TORCH_CROP") is None

# TF32 convolutions (PyTorch default on Ampere) cost ~0.1px of landmark accuracy; keep full fp32.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
PI = math.pi
F32 = torch.float32


def norm_rad(a):
    return a - 2 * PI * torch.floor((a + PI) / (2 * PI))


# ---------------------------------------------------------------------------------------------
# Crops: emulation of OpenCV warpPerspective (INTER_LINEAR, fixed point) as used by ImageToTensor
# ---------------------------------------------------------------------------------------------
def pad_roi(rects_px, Wt, Ht, keep_aspect):
    """rects_px [N,5] (cx,cy,w,h,rot) in pixels -> padded rect + letterbox padding [N,4] (l,t,r,b)."""
    r = rects_px.clone()
    pad = torch.zeros(len(r), 4, device=r.device, dtype=F32)
    if not keep_aspect:
        return r, pad
    ta = Ht / Wt
    ra = r[:, 3] / r[:, 2]
    tall = ta > ra
    # ta > ra: h = w*ta, pad_v ; else w = h/ta, pad_h
    pv = (1 - ra / ta) / 2
    ph = (1 - ta / ra) / 2
    r[:, 3] = torch.where(tall, r[:, 2] * ta, r[:, 3])
    r[:, 2] = torch.where(tall, r[:, 2], r[:, 3] / ta)
    pad[:, 0] = torch.where(tall, torch.zeros_like(ph), ph)
    pad[:, 2] = pad[:, 0]
    pad[:, 1] = torch.where(tall, pv, torch.zeros_like(pv))
    pad[:, 3] = pad[:, 1]
    return r, pad


def crop(frames, fidx, img_wh, rects_norm, Wt, Ht, keep_aspect, border, lo, hi, dtype=F32):
    """Sample rotated-rect crops from a padded uint8 frame batch.

    frames [S,Hm,Wm,3] uint8 (cuda); fidx [N] long (which frame); img_wh [N,2] (true W,H);
    rects_norm [N,5] (xc,yc,w,h,rot) normalized. Returns tensor [N,Ht,Wt,3] in [lo,hi], padding [N,4].
    """
    N = len(fidx)
    dev = rects_norm.device
    W = img_wh[:, 0].to(F32); H = img_wh[:, 1].to(F32)
    rp = torch.stack([rects_norm[:, 0] * W, rects_norm[:, 1] * H, rects_norm[:, 2] * W,
                      rects_norm[:, 3] * H, rects_norm[:, 4]], 1)
    rp, pad = pad_roi(rp, Wt, Ht, keep_aspect)
    # cv::RotatedRect::points in float32 (angle in degrees -> radians in double, cos/sin cast to float)
    ang = (rp[:, 4].double() * 180.0 / PI) * PI / 180.0
    b = torch.cos(ang).float() * 0.5
    a = torch.sin(ang).float() * 0.5
    cx, cy, w, h = rp[:, 0], rp[:, 1], rp[:, 2], rp[:, 3]
    p0x = cx - a * h - b * w; p0y = cy + b * h - a * w          # -> dst (0,Ht)
    p1x = cx + a * h - b * w; p1y = cy - b * h - a * w          # -> dst (0,0)
    p2x = 2 * cx - p0x; p2y = 2 * cy - p0y                      # -> dst (Wt,0)
    # affine map dst(u,v) -> src, in double
    p0x, p0y, p1x, p1y, p2x, p2y = [t.double() for t in (p0x, p0y, p1x, p1y, p2x, p2y)]
    scale = np.float32((hi - lo) / 255.0)
    if isinstance(frames, (FrameTable, NV12Table)) or USE_TRITON:
        table = frames if isinstance(frames, (FrameTable, NV12Table)) else FrameTable.from_batch(frames)
        params = torch.stack([p1x, p1y, p2x - p1x, p2y - p1y, p0x - p1x, p0y - p1y], 1)
        return crop_triton(table, fidx, img_wh, params, Wt, Ht, border == "zero", scale, lo, dtype), pad
    u = torch.arange(Wt, device=dev, dtype=torch.float64) / Wt
    v = torch.arange(Ht, device=dev, dtype=torch.float64) / Ht
    sx = p1x[:, None, None] + u[None, None, :] * (p2x - p1x)[:, None, None] + v[None, :, None] * (p0x - p1x)[:, None, None]
    sy = p1y[:, None, None] + u[None, None, :] * (p2y - p1y)[:, None, None] + v[None, :, None] * (p0y - p1y)[:, None, None]
    X = torch.round(sx * 32).to(torch.int32)
    Y = torch.round(sy * 32).to(torch.int32)
    ix = X >> 5; fx = X & 31
    iy = Y >> 5; fy = Y & 31
    Wi = img_wh[:, 0].to(torch.int32)[:, None, None]; Hi = img_wh[:, 1].to(torch.int32)[:, None, None]
    S_, Hm, Wm, _ = frames.shape
    flat = frames.reshape(-1, 3)
    base = (fidx.to(torch.int64) * (Hm * Wm))[:, None, None]
    acc = torch.zeros(N, Ht, Wt, 3, device=dev, dtype=torch.int32)
    for dy in (0, 1):
        for dx in (0, 1):
            xx = ix + dx; yy = iy + dy
            wx = fx if dx else (32 - fx)
            wy = fy if dy else (32 - fy)
            wgt = (wx * wy * 32)
            if border == "replicate":
                xc = xx.clamp(min=0); xc = torch.minimum(xc, Wi - 1)
                yc = yy.clamp(min=0); yc = torch.minimum(yc, Hi - 1)
            else:
                inside = (xx >= 0) & (xx < Wi) & (yy >= 0) & (yy < Hi)
                wgt = wgt * inside
                xc = xx.clamp(min=0); xc = torch.minimum(xc, Wi - 1)
                yc = yy.clamp(min=0); yc = torch.minimum(yc, Hi - 1)
            idx = base + yc.to(torch.int64) * Wm + xc.to(torch.int64)
            px = flat[idx.reshape(-1)].reshape(N, Ht, Wt, 3).to(torch.int32)
            acc += px * wgt[..., None]
    out8 = (acc + (1 << 14)) >> 15
    t = out8.to(F32) * float(scale) + float(lo)
    return t.to(dtype), pad


# ---------------------------------------------------------------------------------------------
# Rect helpers (all normalized NormRect [N,5], image size W,H per row)
# ---------------------------------------------------------------------------------------------
def align_rect(p0, p1, W, H, target):
    """AlignmentPointsRects: p0/p1 [N,2] normalized. center p0, size 2*|p1-p0|px."""
    x0 = p0[:, 0] * W; y0 = p0[:, 1] * H
    x1 = p1[:, 0] * W; y1 = p1[:, 1] * H
    box = 2 * torch.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    rot = norm_rad(target - torch.atan2(-(y1 - y0), x1 - x0))
    return torch.stack([x0 / W, y0 / H, box / W, box / H, rot], 1)


def rect_transform(r, W, H, scale, shift_y=0.0, square_long=True):
    r = r.clone()
    w, h, rot = r[:, 2], r[:, 3], r[:, 4]
    if shift_y != 0.0:
        # rot==0 branch is numerically the same formula with sin=0, cos=1
        r[:, 0] = r[:, 0] + (-H * h * shift_y * torch.sin(rot)) / W
        r[:, 1] = r[:, 1] + (H * h * shift_y * torch.cos(rot)) / H
    if square_long:
        L = torch.maximum(w * W, h * H)
        w = L / W; h = L / H
    r[:, 2] = w * scale; r[:, 3] = h * scale
    return r


def project(lms, pad, rect):
    """Letterbox removal + LandmarkProjection. lms [N,K,3] normalized to crop; rect [N,5]."""
    x = (lms[..., 0] - pad[:, None, 0]) / (1 - pad[:, None, 0] - pad[:, None, 2])
    y = (lms[..., 1] - pad[:, None, 1]) / (1 - pad[:, None, 1] - pad[:, None, 3])
    z = lms[..., 2] / (1 - pad[:, None, 0] - pad[:, None, 2])
    xr = x - 0.5; yr = y - 0.5
    c = torch.cos(rect[:, 4])[:, None]; s = torch.sin(rect[:, 4])[:, None]
    X = (c * xr - s * yr) * rect[:, None, 2] + rect[:, None, 0]
    Y = (s * xr + c * yr) * rect[:, None, 3] + rect[:, None, 1]
    Z = z * rect[:, None, 2]
    return torch.stack([X, Y, Z], -1)


def bbox_rect(pts):
    """Axis-aligned bbox rect of normalized points [N,K,2] (FLT_MIN max-init quirk)."""
    mx = pts.amax(1).clamp(min=1.1754944e-38); mn = pts.amin(1)
    c = (mx + mn) / 2
    return c, mx - mn


# ---------------------------------------------------------------------------------------------
# SSD detection decode + "first weighted-NMS cluster" (only dets[0] is ever used)
# ---------------------------------------------------------------------------------------------
def ssd_anchors(input_size, strides):
    anchors = []
    layer = 0
    n_layers = len(strides)
    while layer < n_layers:
        n = 0; last = layer
        while last < n_layers and strides[last] == strides[layer]:
            n += 2; last += 1
        fm = math.ceil(input_size / strides[layer])
        for y in range(fm):
            for x in range(fm):
                for _ in range(n):
                    anchors.append(((x + 0.5) / fm, (y + 0.5) / fm))
        layer = last
    return torch.tensor(anchors, dtype=F32)


def count_persons(raw_boxes, raw_scores, anchors, size, thresh=0.5, min_rel=0.3, max_k=4):
    """Number of distinct people found by the pose detector: weighted-NMS clusters (IoU > 0.3, as in
    MediaPipe) with score >= thresh whose body size (2 x |hip-centre -> head keypoint|, the size MediaPipe
    uses for the pose ROI) is at least min_rel x the largest person's in that frame. Returns [N] int."""
    sc = torch.sigmoid(raw_scores[..., 0].float().clamp(-100, 100))
    rb = raw_boxes.float()
    ax, ay = anchors[None, :, 0], anchors[None, :, 1]
    xc = rb[..., 0] / size + ax; yc = rb[..., 1] / size + ay
    w = rb[..., 2] / size; h = rb[..., 3] / size
    x0 = xc - w / 2; y0 = yc - h / 2
    k0x = rb[..., 4] / size + ax; k0y = rb[..., 5] / size + ay
    k1x = rb[..., 6] / size + ax; k1y = rb[..., 7] / size + ay
    remaining = (sc >= thresh) & (w >= 0) & (h >= 0)
    sizes = []
    for _ in range(max_k):
        scm = torch.where(remaining, sc, torch.full_like(sc, -1.0))
        top = scm.argmax(1)
        valid = remaining.any(1)
        ar = torch.arange(len(top), device=rb.device)
        tx0, ty0, tw, th = x0[ar, top], y0[ar, top], w[ar, top], h[ar, top]
        ix0 = torch.maximum(x0, tx0[:, None]); iy0 = torch.maximum(y0, ty0[:, None])
        ix1 = torch.minimum(x0 + w, (tx0 + tw)[:, None]); iy1 = torch.minimum(y0 + h, (ty0 + th)[:, None])
        inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
        iou = inter / (w * h + (tw * th)[:, None] - inter).clamp(min=1e-12)
        cluster = remaining & (iou > 0.3)
        cluster[ar, top] |= remaining[ar, top]
        cw = torch.where(cluster, sc, torch.zeros_like(sc))
        S = cw.sum(1).clamp(min=1e-12)
        dx = ((cw * k1x).sum(1) - (cw * k0x).sum(1)) / S
        dy = ((cw * k1y).sum(1) - (cw * k0y).sum(1)) / S
        sizes.append(torch.where(valid, 2 * torch.sqrt(dx ** 2 + dy ** 2), torch.zeros_like(dx)))
        remaining = remaining & ~cluster
    sz = torch.stack(sizes, 1)                                  # [N, max_k], 0 where no person
    big = sz.amax(1, keepdim=True)
    return ((sz > 0) & (sz >= min_rel * big)).sum(1)


def top_detection(raw_boxes, raw_scores, anchors, size, n_kp, thresh):
    """Returns (valid [N], box [N,4] xmin,ymin,w,h, kps [N,n_kp,2]) of the top weighted cluster."""
    sc = torch.sigmoid(raw_scores[..., 0].float().clamp(-100, 100))
    rb = raw_boxes.float()
    ax, ay = anchors[None, :, 0], anchors[None, :, 1]
    xc = rb[..., 0] / size + ax; yc = rb[..., 1] / size + ay
    w = rb[..., 2] / size; h = rb[..., 3] / size
    xmin = xc - w / 2; ymin = yc - h / 2
    kx = rb[..., 4:4 + 2 * n_kp:2] / size + ax[..., None]
    ky = rb[..., 5:5 + 2 * n_kp:2] / size + ay[..., None]
    ok = (sc >= thresh) & (w >= 0) & (h >= 0)
    scm = torch.where(ok, sc, torch.full_like(sc, -1.0))
    top = scm.argmax(1)
    valid = ok.any(1)
    ar = torch.arange(len(top), device=rb.device)
    txmin, tymin, tw, th = xmin[ar, top], ymin[ar, top], w[ar, top], h[ar, top]
    ix0 = torch.maximum(xmin, txmin[:, None]); iy0 = torch.maximum(ymin, tymin[:, None])
    ix1 = torch.minimum(xmin + w, (txmin + tw)[:, None]); iy1 = torch.minimum(ymin + h, (tymin + th)[:, None])
    inter = (ix1 - ix0).clamp(min=0) * (iy1 - iy0).clamp(min=0)
    iou = inter / (w * h + (tw * th)[:, None] - inter)
    iou = torch.where((ix1 > ix0) & (iy1 > iy0), iou, torch.zeros_like(iou))
    cand = ok & (iou > 0.3)
    cw = torch.where(cand, sc, torch.zeros_like(sc))
    S = cw.sum(1).clamp(min=1e-12)
    bx0 = (cw * xmin).sum(1) / S; by0 = (cw * ymin).sum(1) / S
    bx1 = (cw * (xmin + w)).sum(1) / S; by1 = (cw * (ymin + h)).sum(1) / S
    kxs = (cw[..., None] * kx).sum(1) / S[:, None]; kys = (cw[..., None] * ky).sum(1) / S[:, None]
    box = torch.stack([bx0, by0, bx1 - bx0, by1 - by0], 1)
    return valid, box, torch.stack([kxs, kys], -1)


# ---------------------------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------------------------
class OneEuro:
    """Batched OneEuro filter bank, state per (stream, value). Mirrors mediapipe one_euro_filter.cc."""

    def __init__(self, S, D, min_cutoff, beta, dcutoff, dev):
        self.mc, self.beta, self.dc = min_cutoff, beta, dcutoff
        self.freq = torch.full((S,), 30.0, dtype=torch.float64, device=dev)
        self.last_t = torch.full((S,), -1, dtype=torch.int64, device=dev)
        self.x_init = torch.zeros(S, dtype=torch.bool, device=dev)
        self.x_prev = torch.zeros(S, D, dtype=F32, device=dev)
        self.x_raw = torch.zeros(S, D, dtype=F32, device=dev)
        self.dx_prev = torch.zeros(S, D, dtype=F32, device=dev)

    def reset(self, m):
        self.freq[m] = 30.0; self.last_t[m] = -1; self.x_init[m] = False



class LowPass:
    def __init__(self, S, D, alpha, dev):
        self.a = alpha
        self.init = torch.zeros(S, dtype=torch.bool, device=dev)
        self.prev = torch.zeros(S, D, dtype=F32, device=dev)

    def __call__(self, m, v):
        out = torch.where(self.init[:, None], _lp(np.float32(self.a), v, self.prev), v)
        self.prev = torch.where(m[:, None], out, self.prev)
        self.init = self.init | m
        return out


# ---------------------------------------------------------------------------------------------
# ROI tracking (RoiTrackingCalculator)
# ---------------------------------------------------------------------------------------------
def roi_tracking(prev_ok, prev_rect, prev_lms, rec, W, H, rot_deg, trans, scale, margin):
    """Returns rect [N,5]: prev_rect where tracking criteria hold, else recrop rect `rec`."""
    phi = -rec[:, 4]
    c, s = torch.cos(phi), torch.sin(phi)

    def P(x, y):
        return x * c - y * s, x * s + y * c

    def deg(a):
        return a * 180.0 / PI
    d = deg(prev_rect[:, 4]) - deg(rec[:, 4])
    d = torch.where(d > 180, d - 360, torch.where(d < -180, d + 360, d))
    ok = d.abs() <= rot_deg
    px, py = P(prev_rect[:, 0] * W, prev_rect[:, 1] * H)
    rx, ry = P(rec[:, 0] * W, rec[:, 1] * H)
    ok &= (px - rx).abs() <= rec[:, 2] * W * trans
    ok &= (py - ry).abs() <= rec[:, 3] * H * trans
    ok &= (prev_rect[:, 2] * W - rec[:, 2] * W).abs() <= rec[:, 2] * W * scale
    ok &= (prev_rect[:, 3] * H - rec[:, 3] * H).abs() <= rec[:, 3] * H * scale
    # all previous landmarks strictly inside the (margin-shrunk) recrop rect, in the rotated frame
    cc, ss = c[:, None], s[:, None]
    X = prev_lms[..., 0] * W[:, None]; Y = prev_lms[..., 1] * H[:, None]
    lx = X * cc - Y * ss; ly = X * ss + Y * cc
    hw = rec[:, 2] * W * (1 + margin) / 2; hh = rec[:, 3] * H * (1 + margin) / 2
    inside = ((lx > (rx - hw)[:, None]) & (lx < (rx + hw)[:, None]) &
              (ly > (ry - hh)[:, None]) & (ly < (ry + hh)[:, None])).all(1)
    keep = prev_ok & ok & inside
    return torch.where(keep[:, None], prev_rect, rec)


HAND_ROI_IDX = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]


def hand_lms_to_rect(lms, W, H):
    L = lms[:, HAND_ROI_IDX, :2]
    x0 = L[:, 0, 0] * W; y0 = L[:, 0, 1] * H
    x1 = (L[:, 4, 0] + L[:, 8, 0]) / 2; y1 = (L[:, 4, 1] + L[:, 8, 1]) / 2
    x1 = (x1 + L[:, 6, 0]) / 2 * W; y1 = (y1 + L[:, 6, 1]) / 2 * H
    rot = norm_rad(PI * 0.5 - torch.atan2(-(y1 - y0), x1 - x0))
    rev = norm_rad(-rot)
    ac, _ = bbox_rect(L)
    ox = (L[..., 0] - ac[:, None, 0]) * W[:, None]; oy = (L[..., 1] - ac[:, None, 1]) * H[:, None]
    cr, sr = torch.cos(rev)[:, None], torch.sin(rev)[:, None]
    qx = ox * cr - oy * sr; qy = ox * sr + oy * cr
    FMIN = 1.1754944e-38
    mxx = qx.amax(1).clamp(min=FMIN); mxy = qy.amax(1).clamp(min=FMIN)
    mnx = qx.amin(1); mny = qy.amin(1)
    pcx = (mxx + mnx) / 2; pcy = (mxy + mny) / 2
    cx = pcx * torch.cos(rot) - pcy * torch.sin(rot) + W * ac[:, 0]
    cy = pcx * torch.sin(rot) + pcy * torch.cos(rot) + H * ac[:, 1]
    r = torch.stack([cx / W, cy / H, (mxx - mnx) / W, (mxy - mny) / H, rot], 1)
    return rect_transform(r, W, H, 2.0, -0.1, True)


def face_lms_to_rect(lms, W, H):
    c, wh = bbox_rect(lms[..., :2])
    x0 = lms[:, 33, 0] * W; y0 = lms[:, 33, 1] * H
    x1 = lms[:, 263, 0] * W; y1 = lms[:, 263, 1] * H
    rot = norm_rad(0.0 - torch.atan2(-(y1 - y0), x1 - x0))
    r = torch.stack([c[:, 0], c[:, 1], wh[:, 0], wh[:, 1], rot], 1)
    return rect_transform(r, W, H, 1.5, 0.0, True)


# ---------------------------------------------------------------------------------------------
class Bucketed(torch.nn.Module):
    """Pads the batch up to a power of two (min 8) so cuDNN sees a handful of shapes only.

    Without this, the varying number of hands/faces per step produces a new conv shape almost
    every step, and cuDNN plan building dominates (seconds per step)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        n = x.shape[0]
        b = max(8, 1 << (n - 1).bit_length())
        if b != n:
            x = torch.cat([x, x.new_zeros((b - n,) + x.shape[1:])])
        return [o[:n] for o in self.m(x)]


class HolisticGPU:
    def __init__(self, S, device="cuda", half=False, pose_model="pose_landmark_full", backend="torch", smooth=True):
        self.S, self.dev, self.half, self.smooth = S, torch.device(device), half, smooth
        self.dt = torch.float16 if (half and backend == "torch") else F32
        sizes = {"pose_detection": 224, "pose_landmark_full": 256, "pose_landmark_lite": 256, "pose_landmark_heavy": 256,
                 "hand_recrop": 256, "hand_landmark_full": 224, "face_detection_short_range": 128, "face_landmark": 192}

        def load(name, outputs=None):
            if backend == "trt":
                from .trt_runner import TRTModule
                n = 2 * S if name.startswith("hand") else S   # hands: two per stream
                return TRTModule(os.path.join(MODEL_DIR, name), outputs, sizes[name], fp16=half,
                                 max_batch=max(512, n), opt_batch=min(512, max(512, n)), device=self.dev)
            m = TFLiteModule(os.path.join(MODEL_DIR, name), outputs=outputs).to(self.dev).eval()
            if half:
                m = m.half()
            return Bucketed(m.to(memory_format=torch.channels_last))
        self.m_pdet = load("pose_detection")
        self.m_pose = load(pose_model, outputs=[0, 1, 3, 4])  # drop segmentation head
        self.m_recrop = load("hand_recrop")
        self.m_hand = load("hand_landmark_full", outputs=[0, 1])
        self.m_fdet = load("face_detection_short_range")
        self.m_face = load("face_landmark")
        self.pd_anchors = ssd_anchors(224, [8, 16, 32, 32, 32]).to(self.dev)
        self.fd_anchors = ssd_anchors(128, [8, 16, 16, 16]).to(self.dev)
        self.reset(torch.ones(S, dtype=torch.bool, device=self.dev), init=True)

    def reset(self, m, init=False):
        """Reset state for streams in mask m (start of a new video)."""
        S, dev = self.S, self.dev
        if init:
            self.k = torch.zeros(S, dtype=torch.int64, device=dev)
            self.prev_roi = torch.zeros(S, 5, device=dev); self.prev_roi_ok = torch.zeros(S, dtype=torch.bool, device=dev)
            self.prev_hand = torch.zeros(2, S, 21, 3, device=dev); self.prev_hand_ok = torch.zeros(2, S, dtype=torch.bool, device=dev)
            self.prev_face = torch.zeros(S, 468, 3, device=dev); self.prev_face_ok = torch.zeros(S, dtype=torch.bool, device=dev)
            self.f_lm = OneEuro(S, 99, 0.05, 80.0, 1.0, dev)
            self.f_aux = OneEuro(S, 6, 0.01, 10.0, 1.0, dev)
            self.f_world = OneEuro(S, 99, 0.1, 40.0, 1.0, dev)
            self.v_lm = LowPass(S, 33, 0.1, dev)
            return
        self.k[m] = 0
        self.prev_roi_ok[m] = False; self.prev_hand_ok[:, m] = False; self.prev_face_ok[m] = False
        for f in (self.f_lm, self.f_aux, self.f_world):
            f.reset(m)
        self.v_lm.init[m] = False

    @torch.no_grad()
    def count_persons(self, frames, img_wh, ids, min_rel=0.3):
        """Run the pose detector on the full frames of streams `ids` and count distinct people
        (see count_persons). frames: FrameTable/NV12Table/batch as for step(); img_wh [S,2]."""
        if not isinstance(frames, (FrameTable, NV12Table)):
            frames = FrameTable.from_batch(frames)
        n = len(ids)
        full = torch.tensor([[0.5, 0.5, 1.0, 1.0, 0.0]], device=self.dev).expand(n, 5).contiguous()
        t, _ = crop(frames, ids, img_wh[ids], full, 224, 224, True, "zero", -1.0, 1.0, self.dt)
        boxes, scores = self.m_pdet(t)
        return count_persons(boxes, scores, self.pd_anchors, 224.0, 0.5, min_rel)

    @torch.no_grad()
    def step(self, frames, img_wh, active):
        """frames: [S,Hm,Wm,3] uint8 cuda batch, or a FrameTable (per-stream tensors);
        img_wh [S,2] int; active [S] bool.
        Returns dict of tensors: pose [S,33,4] (x,y,z normalized + vis), world [S,33,4], face [S,468,3],
        lhand/rhand [S,21,3], and presence masks."""
        S, dev = self.S, self.dev
        if not isinstance(frames, (FrameTable, NV12Table)):
            frames = FrameTable.from_batch(frames)
        W = img_wh[:, 0].to(F32); H = img_wh[:, 1].to(F32)
        self.k = self.k + active.long()
        ar = torch.arange(S, device=dev)

        # ---------------- pose ROI: tracking or detection (§1, §3) ----------------
        roi = self.prev_roi.clone(); roi_ok = self.prev_roi_ok & active
        need_det = active & ~self.prev_roi_ok
        if need_det.any():
            ids = ar[need_det]
            full = torch.stack([torch.full_like(W[ids], 0.5), torch.full_like(W[ids], 0.5),
                                torch.ones_like(W[ids]), torch.ones_like(W[ids]), torch.zeros_like(W[ids])], 1)
            t, pad = crop(frames, ids, img_wh[ids], full, 224, 224, True, "zero", -1.0, 1.0, self.dt)
            boxes, scores = self.m_pdet(t)
            valid, box, kp = top_detection(boxes, scores, self.pd_anchors, 224.0, 4, 0.5)
            # letterbox removal
            kx = (kp[..., 0] - pad[:, None, 0]) / (1 - pad[:, None, 0] - pad[:, None, 2])
            ky = (kp[..., 1] - pad[:, None, 1]) / (1 - pad[:, None, 1] - pad[:, None, 3])
            Wi, Hi = W[ids], H[ids]
            r = align_rect(torch.stack([kx[:, 0], ky[:, 0]], 1), torch.stack([kx[:, 1], ky[:, 1]], 1), Wi, Hi, PI / 2)
            r = rect_transform(r, Wi, Hi, 1.25, 0.0, True)
            roi[ids] = r
            roi_ok[ids] = valid

        # ---------------- pose landmarks (§2) ----------------
        pose_ok = torch.zeros(S, dtype=torch.bool, device=dev)
        lm = torch.zeros(S, 33, 3, device=dev); vis = torch.zeros(S, 33, device=dev)
        aux = torch.zeros(S, 2, 3, device=dev); world = torch.zeros(S, 33, 3, device=dev)
        if roi_ok.any():
            ids = ar[roi_ok]
            t, pad = crop(frames, ids, img_wh[ids], roi[ids], 256, 256, True, "replicate", 0.0, 1.0, self.dt)
            o_lm, o_flag, o_hm, o_world = [x.float() for x in self.m_pose(t)]
            pres = o_flag.reshape(-1) > 0.5
            l39 = o_lm.reshape(-1, 39, 5)
            x = l39[..., 0] / 256; y = l39[..., 1] / 256; z = l39[..., 2] / 256
            v = torch.sigmoid(l39[..., 3])
            x, y = refine_heatmap(x, y, o_hm)
            p = project(torch.stack([x, y, z], -1), pad, roi[ids])
            r = roi[ids, 4]
            w3 = o_world.reshape(-1, 39, 3)[:, :33]
            c, s = torch.cos(r)[:, None], torch.sin(r)[:, None]
            wl = torch.stack([c * w3[..., 0] - s * w3[..., 1], s * w3[..., 0] + c * w3[..., 1], w3[..., 2]], -1)
            sel = ids[pres]
            pose_ok[sel] = True
            lm[sel] = p[pres, :33]; aux[sel] = p[pres, 33:35]; vis[sel] = v[pres, :33]; world[sel] = wl[pres]

        # ---------------- smoothing (§4) ----------------
        lost = active & ~pose_ok
        self.f_lm.reset(lost); self.f_aux.reset(lost)
        tn = self.k * 33333000  # per-stream simulated timestamps (Python solution API: 33333us/frame)
        # object scale from unfiltered aux
        ax0 = aux[:, 0, 0] * W; ay0 = aux[:, 0, 1] * H; ax1 = aux[:, 1, 0] * W; ay1 = aux[:, 1, 1] * H
        obj = 2 * torch.sqrt((ax1 - ax0) ** 2 + (ay1 - ay0) ** 2)
        ok_scale = obj >= 1e-6
        vs = (1.0 / obj.double().clamp(min=1e-6))
        vis_s = self.v_lm(pose_ok, vis)
        px = torch.stack([lm[..., 0] * W[:, None], lm[..., 1] * H[:, None], lm[..., 2] * W[:, None]], -1).reshape(S, -1)
        pa = torch.stack([aux[..., 0] * W[:, None], aux[..., 1] * H[:, None], aux[..., 2] * W[:, None]], -1).reshape(S, -1)
        fl = _apply_filter(self.f_lm, pose_ok & ok_scale, tn, vs, px).reshape(S, 33, 3)
        fa = _apply_filter(self.f_aux, pose_ok & ok_scale, tn, vs, pa).reshape(S, 2, 3)
        lm_s = torch.stack([fl[..., 0] / W[:, None], fl[..., 1] / H[:, None], fl[..., 2] / W[:, None]], -1)
        aux_s = torch.stack([fa[..., 0] / W[:, None], fa[..., 1] / H[:, None]], -1)
        world_s = _apply_filter(self.f_world, pose_ok, tn, torch.ones(S, dtype=torch.float64, device=dev),
                                world.reshape(S, -1)).reshape(S, 33, 3)
        if not self.smooth:  # smooth_landmarks=False: landmark/world/visibility filters are pass-through
            lm_s, world_s, vis_s = lm, world, vis   # (aux smoothing is not gated by the flag)
        # next-frame ROI (§3)
        nr = rect_transform(align_rect(aux_s[:, 0], aux_s[:, 1], W, H, PI / 2), W, H, 1.25, 0.0, True)
        self.prev_roi = torch.where(pose_ok[:, None], nr, self.prev_roi)
        self.prev_roi_ok = torch.where(active, pose_ok, self.prev_roi_ok)

        # ---------------- hands (§5) ----------------
        hands = torch.zeros(2, S, 21, 3, device=dev); hands_ok = torch.zeros(2, S, dtype=torch.bool, device=dev)
        gate = torch.stack([pose_ok & (vis_s[:, 15] > 0.1), pose_ok & (vis_s[:, 16] > 0.1)])  # [2,S]
        if gate.any():
            sid, sidx = torch.nonzero(gate, as_tuple=True)     # side id, stream idx
            Ws, Hs = W[sidx], H[sidx]
            wi = torch.tensor([15, 16], device=dev)[sid]
            P = lambda j: lm_s[sidx, j, :2]
            wr, pk, ix = P(wi), P(wi + 2), P(wi + 4)
            wpx = torch.stack([wr[:, 0] * Ws, wr[:, 1] * Hs], 1)
            mpx = (2 * torch.stack([ix[:, 0] * Ws, ix[:, 1] * Hs], 1) + torch.stack([pk[:, 0] * Ws, pk[:, 1] * Hs], 1)) / 3
            # §5.3: centre = middle-finger estimate m, size 2|m-wrist|, rotation wrist->m
            box = 2 * torch.sqrt(((mpx - wpx) ** 2).sum(1))
            rot = norm_rad(PI / 2 - torch.atan2(-(mpx[:, 1] - wpx[:, 1]), mpx[:, 0] - wpx[:, 0]))
            r0 = torch.stack([mpx[:, 0] / Ws, mpx[:, 1] / Hs, box / Ws, box / Hs, rot], 1)
            r0 = rect_transform(r0, Ws, Hs, 2.7, -0.1, True)
            t, pad = crop(frames, sidx, img_wh[sidx], r0, 256, 256, True, "replicate", 0.0, 1.0, self.dt)
            o = self.m_recrop(t)[0].float().reshape(-1, 2, 2) / 256
            c2 = project(torch.cat([o, torch.zeros_like(o[..., :1])], -1), pad, r0)
            rc = align_rect(c2[:, 0, :2], c2[:, 1, :2], Ws, Hs, -PI / 2)
            rc = rect_transform(rc, Ws, Hs, 1.0, -0.1, True)
            prev = self.prev_hand[sid, sidx]; prev_ok = self.prev_hand_ok[sid, sidx]
            prect = hand_lms_to_rect(prev, Ws, Hs)
            rt = roi_tracking(prev_ok, prect, prev[..., :2], rc, Ws, Hs, 40.0, 0.2, 0.4, -0.1)
            t, pad = crop(frames, sidx, img_wh[sidx], rt, 224, 224, True, "replicate", 0.0, 1.0, self.dt)
            o_lm, o_pres = [x.float() for x in self.m_hand(t)]
            pres = o_pres.reshape(-1) > 0.5
            l = o_lm.reshape(-1, 21, 3)
            l = torch.stack([l[..., 0] / 224, l[..., 1] / 224, l[..., 2] / 224 / 0.4], -1)
            hp = project(l, pad, rt)
            hands[sid[pres], sidx[pres]] = hp[pres]
            hands_ok[sid[pres], sidx[pres]] = True
        self.prev_hand = torch.where(active[None, :, None, None], hands, self.prev_hand)
        self.prev_hand_ok = torch.where(active[None], hands_ok, self.prev_hand_ok)

        # ---------------- face (§6) ----------------
        face = torch.zeros(S, 468, 3, device=dev); face_ok = torch.zeros(S, dtype=torch.bool, device=dev)
        if pose_ok.any():
            ids = ar[pose_ok]; Wi, Hi = W[ids], H[ids]
            F11 = lm_s[ids, :11, :2]
            c, wh = bbox_rect(F11)
            p5 = F11[:, 5]; p2 = F11[:, 2]
            rot = norm_rad(0.0 - torch.atan2(-(p2[:, 1] * Hi - p5[:, 1] * Hi), p2[:, 0] * Wi - p5[:, 0] * Wi))
            rf = rect_transform(torch.stack([c[:, 0], c[:, 1], wh[:, 0], wh[:, 1], rot], 1), Wi, Hi, 3.0, 0.0, True)
            t, pad = crop(frames, ids, img_wh[ids], rf, 128, 128, True, "zero", -1.0, 1.0, self.dt)
            boxes, scores = self.m_fdet(t)
            valid, box, kp = top_detection(boxes, scores, self.fd_anchors, 128.0, 6, 0.5)
            # project keypoints with the padded ROI (§6.2) into normalized image coords
            rpx = torch.stack([rf[:, 0] * Wi, rf[:, 1] * Hi, rf[:, 2] * Wi, rf[:, 3] * Hi, rf[:, 4]], 1)
            rpx, _ = pad_roi(rpx, 128, 128, True)
            cr, sr = torch.cos(rpx[:, 4])[:, None], torch.sin(rpx[:, 4])[:, None]
            dx = (kp[..., 0] - 0.5) * rpx[:, None, 2]; dy = (kp[..., 1] - 0.5) * rpx[:, None, 3]
            KX = (rpx[:, None, 0] + cr * dx - sr * dy) / Wi[:, None]
            KY = (rpx[:, None, 1] + sr * dx + cr * dy) / Hi[:, None]
            kpi = torch.stack([KX, KY], -1)
            kc, kwh = bbox_rect(kpi)
            frot = norm_rad(-torch.atan2(-(KY[:, 1] * Hi - KY[:, 0] * Hi), KX[:, 1] * Wi - KX[:, 0] * Wi))
            rc = rect_transform(torch.stack([kc[:, 0], kc[:, 1], kwh[:, 0], kwh[:, 1], frot], 1), Wi, Hi, 2.0, -0.1, True)
            prev = self.prev_face[ids]; prev_ok = self.prev_face_ok[ids]
            prect = face_lms_to_rect(prev, Wi, Hi)
            rt = roi_tracking(prev_ok, prect, prev[..., :2], rc, Wi, Hi, 15.0, 0.1, 0.3, -0.2)
            ids = ids[valid]; rt = rt[valid]
            if len(ids):
                t, pad = crop(frames, ids, img_wh[ids], rt, 192, 192, False, "replicate", 0.0, 1.0, self.dt)
                o_lm, o_flag = [x.float() for x in self.m_face(t)]
                pres = torch.sigmoid(o_flag.reshape(-1)) > 0.5
                l = o_lm.reshape(-1, 468, 3) / 192
                fp = project(l, pad, rt)
                face[ids[pres]] = fp[pres]; face_ok[ids[pres]] = True
        self.prev_face = torch.where(active[:, None, None], face, self.prev_face)
        self.prev_face_ok = torch.where(active, face_ok, self.prev_face_ok)

        return {"pose": torch.cat([lm_s, vis_s[..., None]], -1), "pose_ok": pose_ok,
                "world": torch.cat([world_s, vis_s[..., None]], -1),
                "face": face, "face_ok": face_ok,
                "lhand": hands[0], "lhand_ok": hands_ok[0], "rhand": hands[1], "rhand_ok": hands_ok[1]}


def _apply_filter(f, m, t_ns, scale, value):
    """OneEuro with per-stream timestamps; filters rows in m, passes others through."""
    # all rows share the same step index pattern only per stream, so handle per-stream t via loop-free trick:
    # OneEuro.__call__ takes a scalar t; emulate per-stream t by temporarily vectorizing.
    upd = m & (f.last_t < t_ns)
    newf = torch.where((f.last_t != 0) & (t_ns != 0), 1.0 / ((t_ns - f.last_t).double() * 1e-9), f.freq)
    freq = torch.where(upd, newf, f.freq)
    v64 = value.double()
    dvalue = torch.where(f.x_init[:, None], (v64 - f.x_raw.double()) * scale[:, None] * freq[:, None],
                         torch.zeros_like(v64))
    te = 1.0 / freq[:, None]
    a_d = (1.0 / (1.0 + (1.0 / (2 * PI * f.dc)) / te)).float()
    first = ~f.x_init[:, None]
    ed = torch.where(first, dvalue.float(), _lp(a_d, dvalue.float(), f.dx_prev))
    cutoff = f.mc + f.beta * ed.double().abs()
    a = (1.0 / (1.0 + (1.0 / (2 * PI * cutoff)) / te)).float()
    out = torch.where(first, value, _lp(a, value, f.x_prev))
    u = upd[:, None]
    f.dx_prev = torch.where(u, ed, f.dx_prev)
    f.x_prev = torch.where(u, out, f.x_prev)
    f.x_raw = torch.where(u, value, f.x_raw)
    f.x_init = f.x_init | upd
    f.freq = freq
    f.last_t = torch.where(upd, t_ns, f.last_t)
    return torch.where(m[:, None], out, value)


def _lp(a, v, prev):
    """mediapipe LowPassFilter::Apply: float(alpha*value [float] + (1.0-alpha)*stored [double])."""
    a32 = a if torch.is_tensor(a) else torch.tensor(a, device=v.device)
    return ((a32 * v).double() + (1.0 - a32.double()) * prev.double()).float()


def refine_heatmap(x, y, hm, ks=9):
    """§2.4. x,y [N,39] normalized; hm [N,64,64,39] logits (NHWC).

    NOTE: the v0.10.21 source pbtxt says kernel_size 7, but the shipped wheel matches the
    reference output only with a 9x9 window (frame-0 error 0.004px vs 0.18px with 7)."""
    N, K = x.shape
    col = torch.trunc(x * 64).long(); row = torch.trunc(y * 64).long()
    inb = (col >= 0) & (col < 64) & (row >= 0) & (row < 64)
    o = (ks - 1) // 2
    off = torch.arange(-o, o + 1, device=x.device)
    rr = row[..., None, None] + off[:, None]          # [N,K,7,1]
    cc = col[..., None, None] + off[None, :]          # [N,K,1,7]
    valid = (rr >= 0) & (rr < 64) & (cc >= 0) & (cc < 64)
    rrc = rr.clamp(0, 63); ccc = cc.clamp(0, 63)
    hmT = hm.permute(0, 3, 1, 2)                       # [N,K,64,64]
    flat = hmT.reshape(N, K, 64 * 64)
    idx = (rrc * 64 + ccc).expand(N, K, ks, ks).reshape(N, K, ks * ks)
    conf = torch.sigmoid(flat.gather(2, idx)).reshape(N, K, ks, ks)
    conf = conf * valid
    S = conf.sum((2, 3))
    mx = conf.amax((2, 3))
    wc = (conf * cc.float()).sum((2, 3)); wr = (conf * rr.float()).sum((2, 3))
    do = inb & (mx >= 0.5) & (S > 0)
    nx = torch.where(do, wc / 64 / S.clamp(min=1e-30), x)
    ny = torch.where(do, wr / 64 / S.clamp(min=1e-30), y)
    return nx, ny
