"""Exact emulation of PyAV/libswscale yuv420p -> rgb24 (the conversion the reference poses were made with).

swscale's unscaled yuv420p->rgb24 path is a per-pixel function of (Y, U, V) with nearest-neighbour
chroma, so it can be captured exactly as a 256^3 x 3 lookup table by converting a synthetic frame
that contains every (Y, U, V) combination. Tables are built on demand per (colorspace, color_range)
into the mp-gpu cache (~/.cache/mp-gpu/yuv_lut). Verify against real videos with:  python yuv_lut.py video1.mp4 ...
"""
import os
import numpy as np
import torch

from .cache import cache_dir, file_lock


def lut_key(frame):
    """PyAV honours the stream's colour tags, so the LUT depends on (colorspace, color_range)."""
    return int(frame.colorspace), int(frame.color_range)


def get_lut(colorspace, color_range):
    path = os.path.join(cache_dir("yuv_lut"), f"cs{colorspace}_r{color_range}.npy")
    with file_lock(path):
        if not os.path.exists(path):
            np.save(path + ".tmp.npy", build_lut(colorspace, color_range))
            os.replace(path + ".tmp.npy", path)
    return np.load(path)


def get_lut_gpu(colorspace, color_range, device):
    """LUT as int32 packed RGB (r | g<<8 | b<<16), flat [2^24], on device (for the NV12 crop kernel)."""
    lut = get_lut(colorspace, color_range).reshape(-1, 3).astype(np.int32)
    packed = lut[:, 0] | (lut[:, 1] << 8) | (lut[:, 2] << 16)
    return torch.from_numpy(packed).to(device)


def build_lut(colorspace=2, color_range=0, width_blocks=4096):
    import av
    # every (y,u,v): chroma block (bu,bv) = (u,v) index; each 2x2 luma block holds one y value.
    # Layout: 256*256 chroma combos x 256 luma values -> blocks arranged in a 2D grid.
    n = 256 ** 3
    cw = width_blocks                 # chroma width (blocks per row)
    ch = n // cw                      # chroma height
    idx = np.arange(n, dtype=np.int64).reshape(ch, cw)
    yv, uv, vv = (idx >> 16) & 255, (idx >> 8) & 255, idx & 255
    Y = np.repeat(np.repeat(yv.astype(np.uint8), 2, 0), 2, 1)
    lut = np.empty((n, 3), np.uint8)
    rows = 512  # process in horizontal strips to limit memory
    for r0 in range(0, ch, rows):
        r1 = min(ch, r0 + rows)
        yy = Y[2 * r0:2 * r1]
        uu = uv[r0:r1].astype(np.uint8); vvv = vv[r0:r1].astype(np.uint8)
        buf = np.concatenate([yy.ravel(), uu.ravel(), vvv.ravel()]).reshape(-1, cw * 2)
        f = av.VideoFrame.from_ndarray(buf, format="yuv420p")
        f.colorspace = colorspace; f.color_range = color_range
        rgb = f.to_ndarray(format="rgb24")            # [2*(r1-r0), 2*cw, 3]
        lut[idx[r0:r1].ravel()] = rgb[::2, ::2].reshape(-1, 3)
    return lut.reshape(256, 256, 256, 3)               # [y,u,v,c]


def _plane(frame, i, h, w):
    p = frame.planes[i]
    return np.frombuffer(p, np.uint8).reshape(h, p.line_size)[:, :w]


def verify(video, nframes=20):
    """Check that LUT(Y, U, V) with nearest chroma reproduces PyAV rgb24 bit-exactly on real frames."""
    import av
    bad = 0
    lut = None
    for i, f in zip(range(nframes), av.open(video).decode(video=0)):
        if lut is None:
            lut = get_lut(*lut_key(f))
        ref = f.to_ndarray(format="rgb24")
        yuv = f.to_ndarray(format="yuv420p")
        H, W = ref.shape[:2]
        y = yuv[:H]
        ch, cw = (H + 1) // 2, (W + 1) // 2
        u = _plane(f, 1, ch, cw)
        v = _plane(f, 2, ch, cw)
        uu = np.repeat(np.repeat(u, 2, 0), 2, 1)[:H, :W]
        vv = np.repeat(np.repeat(v, 2, 0), 2, 1)[:H, :W]
        out = lut[y, uu, vv]
        bad += int((out != ref).any(-1).sum())
    return bad


if __name__ == "__main__":
    import sys
    for v in sys.argv[1:]:
        print(v, "mismatching pixels:", verify(v))
