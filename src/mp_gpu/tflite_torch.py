"""Run a TFLite graph (exported by tflite_export.py) as a PyTorch module.

4D tensors are kept NCHW internally; inputs are NHWC float like TFLite and rank-4
outputs are returned NHWC, so outputs match the TFLite interpreter exactly.
"""
import json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACT = {"NONE": lambda x: x, "RELU": F.relu, "RELU6": lambda x: F.hardtanh(x, 0.0, 6.0)}
NHWC2NCHW_AXIS = {0: 0, 1: 2, 2: 3, 3: 1, -1: 1, -2: 3, -3: 2}


def _same_pad(size, k, s, d=1):
    out = math.ceil(size / s)
    total = max((out - 1) * s + (k - 1) * d + 1 - size, 0)
    return total // 2, total - total // 2


class _Weights:
    """npz weights (stored as fp16 where that is lossless) read back as float32."""

    def __init__(self, z):
        self.z, self.files = z, z.files

    def __getitem__(self, k):
        return self.z[k].astype(np.float32)


class TFLiteModule(nn.Module):
    def __init__(self, base, outputs=None):
        """outputs: optional list of graph-output indices to keep; ops not needed for them are pruned."""
        super().__init__()
        g = json.load(open(base + ".graph.json"))
        w = _Weights(np.load(base + ".weights.npz"))
        self.shapes = {int(k): v for k, v in g["shapes"].items()}
        self.inp, self.outs = g["inputs"][0], g["outputs"]
        ops = list(enumerate(g["ops"]))
        if outputs is not None:
            self.outs = [self.outs[i] for i in outputs]
            need, keep = set(self.outs), []
            for i, op in reversed(ops):
                if need & set(op["outputs"]):
                    keep.append((i, op)); need |= set(op["inputs"])
            ops = keep[::-1]
        self.consts = {}
        self.ops = []
        for i, op in ops:
            self.ops.append(self._build(i, op, w))
        for k in w.files:  # remaining constants (e.g. ADD operands)
            self._const(int(k), w)

    # -- constants -------------------------------------------------------
    def _const(self, idx, w):
        if idx in self.consts:
            return self.consts[idx]
        a = torch.from_numpy(w[str(idx)].copy())
        if a.dim() == 4:
            a = a.permute(0, 3, 1, 2)
        elif a.dim() in (1, 3) and a.shape[-1] > 1:  # per-channel broadcast vector (NHWC last axis)
            a = a.reshape(1, -1, 1, 1) if a.dim() == 1 or a.numel() == a.shape[-1] else a
        name = f"c{idx}"
        self.register_buffer(name, a.contiguous())
        self.consts[idx] = name
        return name

    def _build(self, i, op, w):
        t, o, ins = op["op"], op["opts"], op["inputs"]
        spec = {"t": t, "in": ins, "out": op["outputs"], "act": ACT[o.get("fusedActivationFunction", "NONE")]}
        if t in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            k = w[str(ins[1])]  # conv: [O,H,W,I]; depthwise: [1,H,W,C*m]
            if t == "CONV_2D":
                kt = torch.from_numpy(k).permute(0, 3, 1, 2).contiguous()
                groups = 1
            else:
                kt = torch.from_numpy(k).permute(3, 0, 1, 2).contiguous()
                groups = self.shapes[ins[0]][3]
            conv = nn.Conv2d(kt.shape[1] * groups, kt.shape[0], kt.shape[2:], stride=(o["strideH"], o["strideW"]),
                             dilation=(o["dilationHFactor"], o["dilationWFactor"]), groups=groups,
                             bias=len(ins) > 2 and ins[2] >= 0)
            conv.weight.data.copy_(kt)
            if conv.bias is not None:
                conv.bias.data.copy_(torch.from_numpy(w[str(ins[2])]))
            self.add_module(f"op{i}", conv)
            ih, iw = self.shapes[ins[0]][1:3]
            if o["padding"] == "SAME":
                ph = _same_pad(ih, kt.shape[2], o["strideH"], o["dilationHFactor"])
                pw = _same_pad(iw, kt.shape[3], o["strideW"], o["dilationWFactor"])
                spec["pad"] = (pw[0], pw[1], ph[0], ph[1])
            else:
                spec["pad"] = None
            spec["mod"] = f"op{i}"
        elif t == "MAX_POOL_2D":
            ih, iw = self.shapes[ins[0]][1:3]
            kh, kw = o["filterHeight"], o["filterWidth"]
            spec.update(k=(kh, kw), s=(o["strideH"], o["strideW"]), pad=None)
            if o["padding"] == "SAME":
                ph = _same_pad(ih, kh, o["strideH"]); pw = _same_pad(iw, kw, o["strideW"])
                spec["pad"] = (pw[0], pw[1], ph[0], ph[1])
        elif t == "PAD":
            p = w[str(ins[1])].astype(int)  # NHWC [[n0,n1],[h0,h1],[w0,w1],[c0,c1]]
            spec["pad"] = (int(p[2, 0]), int(p[2, 1]), int(p[1, 0]), int(p[1, 1]), int(p[3, 0]), int(p[3, 1]))
        elif t == "PRELU":
            spec["alpha"] = self._const(ins[1], w)
            a = getattr(self, spec["alpha"])
            setattr(self, spec["alpha"], a.reshape(1, -1, 1, 1))
        elif t == "RESHAPE":
            spec["shape"] = self.shapes[op["outputs"][0]]
        elif t == "CONCATENATION":
            nd = len(self.shapes[op["outputs"][0]])
            spec["axis"] = NHWC2NCHW_AXIS[o["axis"]] if nd == 4 else o["axis"]
        elif t == "RESIZE_BILINEAR":
            spec["size"] = self.shapes[op["outputs"][0]][1:3]
            assert o["halfPixelCenters"] or o["alignCorners"]
            spec["align"] = bool(o["alignCorners"])
        elif t == "DEPTH_TO_SPACE":
            spec["b"] = o["blockSize"]
        elif t == "MEAN":
            ax = [int(a) for a in np.atleast_1d(w[str(ins[1])])]
            spec["axis"] = [NHWC2NCHW_AXIS[a] for a in ax]
            spec["keep"] = bool(o.get("keepDims", False))
        elif t == "FULLY_CONNECTED":
            lin = nn.Linear(*w[str(ins[1])].shape[::-1], bias=len(ins) > 2 and ins[2] >= 0)
            lin.weight.data.copy_(torch.from_numpy(w[str(ins[1])]))
            if lin.bias is not None:
                lin.bias.data.copy_(torch.from_numpy(w[str(ins[2])]))
            self.add_module(f"op{i}", lin)
            spec["mod"] = f"op{i}"
        elif t == "STRIDED_SLICE":
            spec.update(b=w[str(ins[1])].astype(int).tolist(), e=w[str(ins[2])].astype(int).tolist(),
                        st=w[str(ins[3])].astype(int).tolist(), bm=o.get("beginMask", 0), em=o.get("endMask", 0),
                        sm=o.get("shrinkAxisMask", 0))
        elif t in ("ADD", "MUL", "SUB"):
            for x in ins:
                if str(x) in w.files:
                    self._const(x, w)
        elif t in ("LOGISTIC", "RELU", "TANH"):
            pass
        else:
            raise NotImplementedError(t)
        return spec

    # -- execution --------------------------------------------------------
    @staticmethod
    def _to_nhwc(x):
        return x.permute(0, 2, 3, 1) if x.dim() == 4 else x

    def _get(self, env, idx):
        if idx in env:
            return env[idx]
        return getattr(self, self.consts[idx])

    def forward(self, x_nhwc):
        env = {self.inp: x_nhwc.permute(0, 3, 1, 2)}
        for s in self.ops:
            t = s["t"]
            a = self._get(env, s["in"][0])
            if t in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                if s["pad"] is not None and any(s["pad"]):
                    a = F.pad(a, s["pad"])
                y = s["act"](getattr(self, s["mod"])(a))
            elif t == "MAX_POOL_2D":
                if s["pad"] is not None and any(s["pad"]):
                    a = F.pad(a, s["pad"], value=float("-inf"))
                y = F.max_pool2d(a, s["k"], s["s"])
            elif t == "PAD":
                y = F.pad(a, s["pad"])
            elif t == "PRELU":
                y = torch.where(a >= 0, a, a * getattr(self, s["alpha"]))
            elif t in ("ADD", "MUL", "SUB"):
                b = self._get(env, s["in"][1])
                y = s["act"](a + b if t == "ADD" else a * b if t == "MUL" else a - b)
            elif t == "LOGISTIC":
                y = torch.sigmoid(a)
            elif t == "RELU":
                y = F.relu(a)
            elif t == "TANH":
                y = torch.tanh(a)
            elif t == "RESHAPE":
                y = self._to_nhwc(a).reshape([a.shape[0]] + s["shape"][1:])
                if y.dim() == 4:
                    y = y.permute(0, 3, 1, 2)
            elif t == "CONCATENATION":
                y = torch.cat([self._get(env, i) for i in s["in"]], s["axis"])
            elif t == "RESIZE_BILINEAR":
                y = F.interpolate(a, size=s["size"], mode="bilinear", align_corners=s["align"])
            elif t == "DEPTH_TO_SPACE":
                b = s["b"]; n, c, h, w = a.shape
                y = a.reshape(n, b, b, c // (b * b), h, w).permute(0, 3, 4, 1, 5, 2).reshape(n, c // (b * b), h * b, w * b)
            elif t == "MEAN":
                y = a.mean(dim=s["axis"], keepdim=s["keep"])
            elif t == "FULLY_CONNECTED":
                y = getattr(self, s["mod"])(self._to_nhwc(a).reshape(a.shape[0], -1))
                y = s["act"](y)
            elif t == "STRIDED_SLICE":
                v = self._to_nhwc(a)
                sl = []
                for d in range(v.dim()):
                    b0 = None if (s["bm"] >> d) & 1 else s["b"][d]
                    e0 = None if (s["em"] >> d) & 1 else s["e"][d]
                    if d == 0:
                        b0, e0 = None, None  # keep batch
                    sl.append(b0 if (s["sm"] >> d) & 1 else slice(b0, e0, s["st"][d]))
                y = v[tuple(sl)]
                if y.dim() == 4:
                    y = y.permute(0, 3, 1, 2)
            env[s["out"][0]] = y
        return [self._to_nhwc(env[o]) for o in self.outs]

