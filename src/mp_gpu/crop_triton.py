"""Fused Triton kernel for rotated-ROI crops, bit-compatible with OpenCV warpPerspective
(INTER_LINEAR, 1/32-pixel coordinate quantization, 15-bit fixed-point weights, uint8 rounding),
followed by the ImageToTensor normalization. Replaces ~40 elementwise torch ops per pixel.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _crop_kernel(addr_ptr, strides_ptr, out_ptr, params_ptr, wh_ptr,
                 Ht, Wt, scale, lo,
                 ZERO_BORDER: tl.constexpr, BLOCK: tl.constexpr, OUT_F16: tl.constexpr):
    """Each crop n reads its source image from a raw address with (channel, row, col) strides,
    so frames can live in separate tensors (e.g. per-stream decoder output, CHW or HWC)."""
    n = tl.program_id(0)
    pb = tl.program_id(1)
    offs = pb * BLOCK + tl.arange(0, BLOCK)
    npx = Ht * Wt
    m = offs < npx
    v = offs // Wt
    u = offs - v * Wt
    # params (float64): p1x, p1y, dux, duy, dvx, dvy   (src = p1 + (u/Wt)*du + (v/Ht)*dv)
    pp = params_ptr + n * 6
    p1x = tl.load(pp + 0); p1y = tl.load(pp + 1)
    dux = tl.load(pp + 2); duy = tl.load(pp + 3)
    dvx = tl.load(pp + 4); dvy = tl.load(pp + 5)
    uf = u.to(tl.float64) / Wt
    vf = v.to(tl.float64) / Ht
    sx = p1x + uf * dux + vf * dvx
    sy = p1y + uf * duy + vf * dvy
    X = libdevice.rint(sx * 32.0).to(tl.int32)
    Y = libdevice.rint(sy * 32.0).to(tl.int32)
    ix = X >> 5; fx = X & 31
    iy = Y >> 5; fy = Y & 31
    W = tl.load(wh_ptr + n * 2); H = tl.load(wh_ptr + n * 2 + 1)
    src = tl.load(addr_ptr + n).to(tl.pointer_type(tl.uint8))
    s_c = tl.load(strides_ptr + n * 3 + 0); s_y = tl.load(strides_ptr + n * 3 + 1); s_x = tl.load(strides_ptr + n * 3 + 2)
    acc0 = tl.zeros([BLOCK], tl.int32); acc1 = tl.zeros([BLOCK], tl.int32); acc2 = tl.zeros([BLOCK], tl.int32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            xx = ix + dx; yy = iy + dy
            if dx == 1:
                wx = fx
            else:
                wx = 32 - fx
            if dy == 1:
                wy = fy
            else:
                wy = 32 - fy
            wgt = wx * wy * 32
            if ZERO_BORDER:
                inside = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                wgt = tl.where(inside, wgt, 0)
            xc = tl.minimum(tl.maximum(xx, 0), W - 1)
            yc = tl.minimum(tl.maximum(yy, 0), H - 1)
            pix = yc.to(tl.int64) * s_y + xc.to(tl.int64) * s_x
            acc0 += tl.load(src + pix, mask=m, other=0).to(tl.int32) * wgt
            acc1 += tl.load(src + pix + s_c, mask=m, other=0).to(tl.int32) * wgt
            acc2 += tl.load(src + pix + 2 * s_c, mask=m, other=0).to(tl.int32) * wgt
    o0 = ((acc0 + 16384) >> 15).to(tl.float32) * scale + lo
    o1 = ((acc1 + 16384) >> 15).to(tl.float32) * scale + lo
    o2 = ((acc2 + 16384) >> 15).to(tl.float32) * scale + lo
    ob = out_ptr + (n.to(tl.int64) * npx + offs) * 3
    if OUT_F16:
        o0 = o0.to(tl.float16); o1 = o1.to(tl.float16); o2 = o2.to(tl.float16)
    tl.store(ob + 0, o0, mask=m)
    tl.store(ob + 1, o1, mask=m)
    tl.store(ob + 2, o2, mask=m)


@triton.jit
def _crop_nv12_kernel(addr_ptr, out_ptr, params_ptr, wh_ptr,
                      Ht, Wt, scale, lo,
                      ZERO_BORDER: tl.constexpr, BLOCK: tl.constexpr, OUT_F16: tl.constexpr):
    """Like _crop_kernel, but sources are NV12 (NVDEC output). Each tap is converted to RGB with an
    exact libswscale lookup table (int32 packed RGB, indexed by Y<<16|U<<8|V) before interpolation,
    which reproduces decode->rgb24->warpPerspective bit-exactly.
    addr table per crop: [y_addr, uv_addr, y_pitch, uv_pitch, lut_addr].
    lut_addr == 0 marks an RGB source instead: y_addr = HWC uint8 image, y_pitch = row stride."""
    n = tl.program_id(0)
    pb = tl.program_id(1)
    offs = pb * BLOCK + tl.arange(0, BLOCK)
    npx = Ht * Wt
    m = offs < npx
    v = offs // Wt
    u = offs - v * Wt
    pp = params_ptr + n * 6
    p1x = tl.load(pp + 0); p1y = tl.load(pp + 1)
    dux = tl.load(pp + 2); duy = tl.load(pp + 3)
    dvx = tl.load(pp + 4); dvy = tl.load(pp + 5)
    uf = u.to(tl.float64) / Wt
    vf = v.to(tl.float64) / Ht
    sx = p1x + uf * dux + vf * dvx
    sy = p1y + uf * duy + vf * dvy
    X = libdevice.rint(sx * 32.0).to(tl.int32)
    Y = libdevice.rint(sy * 32.0).to(tl.int32)
    ix = X >> 5; fx = X & 31
    iy = Y >> 5; fy = Y & 31
    W = tl.load(wh_ptr + n * 2); H = tl.load(wh_ptr + n * 2 + 1)
    a = addr_ptr + n * 5
    ysrc = tl.load(a + 0).to(tl.pointer_type(tl.uint8))
    uvsrc = tl.load(a + 1).to(tl.pointer_type(tl.uint8))
    py = tl.load(a + 2); puv = tl.load(a + 3)
    lut_addr = tl.load(a + 4)
    lut = lut_addr.to(tl.pointer_type(tl.int32))
    is_rgb = lut_addr == 0                      # RGB (HWC uint8) source, e.g. CPU-decoded fallback
    acc0 = tl.zeros([BLOCK], tl.int32); acc1 = tl.zeros([BLOCK], tl.int32); acc2 = tl.zeros([BLOCK], tl.int32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            xx = ix + dx; yy = iy + dy
            if dx == 1:
                wx = fx
            else:
                wx = 32 - fx
            if dy == 1:
                wy = fy
            else:
                wy = 32 - fy
            wgt = wx * wy * 32
            if ZERO_BORDER:
                inside = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
                wgt = tl.where(inside, wgt, 0)
            xc = tl.minimum(tl.maximum(xx, 0), W - 1).to(tl.int64)
            yc = tl.minimum(tl.maximum(yy, 0), H - 1).to(tl.int64)
            if is_rgb:
                po = yc * py + xc * 3
                r_ = tl.load(ysrc + po, mask=m, other=0).to(tl.int32)
                g_ = tl.load(ysrc + po + 1, mask=m, other=0).to(tl.int32)
                b_ = tl.load(ysrc + po + 2, mask=m, other=0).to(tl.int32)
            else:
                yv = tl.load(ysrc + yc * py + xc, mask=m, other=0).to(tl.int32)
                uvo = (yc >> 1) * puv + (xc >> 1) * 2
                uu = tl.load(uvsrc + uvo, mask=m, other=0).to(tl.int32)
                vv = tl.load(uvsrc + uvo + 1, mask=m, other=0).to(tl.int32)
                rgb = tl.load(lut + ((yv << 16) | (uu << 8) | vv), mask=m, other=0)
                r_ = rgb & 255
                g_ = (rgb >> 8) & 255
                b_ = (rgb >> 16) & 255
            acc0 += r_ * wgt
            acc1 += g_ * wgt
            acc2 += b_ * wgt
    o0 = ((acc0 + 16384) >> 15).to(tl.float32) * scale + lo
    o1 = ((acc1 + 16384) >> 15).to(tl.float32) * scale + lo
    o2 = ((acc2 + 16384) >> 15).to(tl.float32) * scale + lo
    ob = out_ptr + (n.to(tl.int64) * npx + offs) * 3
    if OUT_F16:
        o0 = o0.to(tl.float16); o1 = o1.to(tl.float16); o2 = o2.to(tl.float16)
    tl.store(ob + 0, o0, mask=m)
    tl.store(ob + 1, o1, mask=m)
    tl.store(ob + 2, o2, mask=m)


class NV12Table:
    """Per-stream sources: NV12 [y_addr, uv_addr, y_pitch, uv_pitch, lut_addr] or RGB [addr, 0, row_stride, 0, 0]."""

    def __init__(self, table, keep):
        self.table, self.keep = table, keep

    @staticmethod
    def from_list(frames, heights, luts, device):
        """frames: per-stream NV12 uint8 tensor [H + ceil(H/2), pitch], or RGB uint8 [H, W, 3], or None;
        heights: true H; luts: per-stream int32 LUT tensors [2^24] (ignored for RGB frames)."""
        rows = []
        for f, h, lut in zip(frames, heights, luts):
            if f is None:
                rows.append((0, 0, 0, 0, 0))
            elif f.dim() == 3:                 # RGB HWC
                rows.append((f.data_ptr(), 0, f.stride(0), 0, 0))
            else:
                p = f.stride(0)
                rows.append((f.data_ptr(), f.data_ptr() + h * p, p, p, lut.data_ptr()))
        t = torch.tensor(rows, dtype=torch.int64).to(device, non_blocking=True)
        return NV12Table(t, (frames, luts))


class FrameTable:
    """Per-stream source image descriptors for the crop kernel: address + (c, y, x) strides.

    from_batch(): a padded uint8 batch [S,Hm,Wm,3] (HWC).  from_list(): one uint8 tensor per stream,
    either [3,H,W] (CHW, e.g. torchcodec CUDA output) or [H,W,3]; None entries are allowed.
    The tensors must stay alive while the table is in use."""

    def __init__(self, addr, strides, keep):
        self.addr, self.strides, self.keep = addr, strides, keep

    @staticmethod
    def from_batch(frames):
        S, Hm, Wm, _ = frames.shape
        base = frames.data_ptr()
        addr = base + torch.arange(S, device=frames.device, dtype=torch.int64) * (Hm * Wm * 3)
        st = torch.tensor([1, Wm * 3, 3], device=frames.device, dtype=torch.int64).expand(S, 3)
        return FrameTable(addr, st, frames)

    @staticmethod
    def from_list(frames, device):
        addr, st = [], []
        for f in frames:
            if f is None:
                addr.append(0); st.append((0, 0, 0))
            elif f.shape[0] == 3:  # CHW
                addr.append(f.data_ptr()); st.append((f.stride(0), f.stride(1), f.stride(2)))
            else:                  # HWC
                addr.append(f.data_ptr()); st.append((f.stride(2), f.stride(0), f.stride(1)))
        a = torch.tensor(addr, dtype=torch.int64).to(device, non_blocking=True)
        s = torch.tensor(st, dtype=torch.int64).to(device, non_blocking=True)
        return FrameTable(a, s, frames)


def crop_triton(table, fidx, img_wh, params, Wt, Ht, zero_border, scale, lo, dtype):
    """table: FrameTable or NV12Table; params [N,6] float64; returns [N,Ht,Wt,3] dtype."""
    N = len(fidx)
    out = torch.empty(N, Ht, Wt, 3, device=params.device, dtype=dtype)
    if N == 0:
        return out
    BLOCK = 512
    grid = (N, triton.cdiv(Ht * Wt, BLOCK))
    if isinstance(table, NV12Table):
        _crop_nv12_kernel[grid](table.table[fidx].contiguous(), out, params.contiguous(),
                                img_wh.to(torch.int32).contiguous(), Ht, Wt, float(scale), float(lo),
                                ZERO_BORDER=zero_border, BLOCK=BLOCK, OUT_F16=(dtype == torch.float16))
        return out
    _crop_kernel[grid](table.addr[fidx].contiguous(), table.strides[fidx].contiguous(), out, params.contiguous(),
                       img_wh.to(torch.int32).contiguous(), Ht, Wt,
                       float(scale), float(lo), ZERO_BORDER=zero_border, BLOCK=BLOCK,
                       OUT_F16=(dtype == torch.float16))
    return out
