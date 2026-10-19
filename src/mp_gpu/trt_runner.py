"""TensorRT engines for the Holistic models (built from the verified TFLiteModule ports).

Engines are cached under models/trt/<name>_<outs>_<fp16|fp32>_<maxbatch>.engine. Requires the
tensorrt python package (pip install tensorrt-cu12==10.9.0.34; see README).
"""
import os
import torch
import tensorrt as trt

from .tflite_torch import TFLiteModule
from .cache import cache_dir, file_lock
LOGGER = trt.Logger(trt.Logger.WARNING)


def _trt_dir(device):
    """Engines are specific to TensorRT version and GPU model."""
    name = torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")
    return cache_dir("trt", f"trt{trt.__version__}_{name}")


def _onnx_path(base, outputs):
    tag = "all" if outputs is None else "-".join(map(str, outputs))
    return os.path.join(cache_dir("onnx"), f"{os.path.basename(base)}_{tag}.onnx")


def export_onnx(base, outputs, size):
    path = _onnx_path(base, outputs)
    with file_lock(path):
        if not os.path.exists(path):
            _export_onnx(base, outputs, size, path)
    return path


def _export_onnx(base, outputs, size, path):
    m = TFLiteModule(base, outputs=outputs).eval()
    x = torch.zeros(2, size, size, 3)
    n_out = len(m.outs)
    torch.onnx.export(m, (x,), path + ".tmp", input_names=["x"], output_names=[f"y{i}" for i in range(n_out)],
                      dynamic_axes={"x": {0: "b"}, **{f"y{i}": {0: "b"} for i in range(n_out)}},
                      opset_version=17, dynamo=False)
    os.replace(path + ".tmp", path)


def build_engine(base, outputs, size, fp16, max_batch, opt_batch, device="cuda"):
    onnx = export_onnx(base, outputs, size)
    eng = os.path.join(_trt_dir(device), os.path.basename(onnx)[:-5] + f"_{'fp16' if fp16 else 'fp32'}_{max_batch}.engine")
    with file_lock(eng):
        if not os.path.exists(eng):
            print(f"mp-gpu: building TensorRT engine {os.path.basename(eng)} (one-time, a few minutes)", flush=True)
            _build_engine(onnx, eng, size, fp16, max_batch, opt_batch)
    return eng


def _build_engine(onnx, eng, size, fp16, max_batch, opt_batch):
    builder = trt.Builder(LOGGER)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, LOGGER)
    with open(onnx, "rb") as f:
        assert parser.parse(f.read()), [parser.get_error(i) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(os.environ.get("TRT_WORKSPACE_GB", 4)) << 30)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    else:
        cfg.clear_flag(trt.BuilderFlag.TF32)   # strict fp32 for fidelity
    prof = builder.create_optimization_profile()
    prof.set_shape("x", (1, size, size, 3), (opt_batch, size, size, 3), (max_batch, size, size, 3))
    cfg.add_optimization_profile(prof)
    ser = builder.build_serialized_network(net, cfg)
    assert ser is not None, f"engine build failed for {onnx}"
    with open(eng + ".tmp", "wb") as f:
        f.write(ser)
    os.replace(eng + ".tmp", eng)


class _SharedActivations:
    """One activation buffer per device shared by all execution contexts. The models run one after
    another on the same stream, so sharing is safe; per-context buffers would cost ~12 GB at max batch 512
    instead of ~4 GB."""
    _bufs, _ctxs = {}, {}

    @classmethod
    def attach(cls, dev, ctx, size):
        buf = cls._bufs.get(dev)
        cls._ctxs.setdefault(dev, []).append(ctx)
        if buf is None or buf.numel() < size:
            buf = cls._bufs[dev] = torch.empty(max(size, 1), dtype=torch.uint8, device=dev)
            for c in cls._ctxs[dev]:          # (re)point every context at the (grown) buffer
                c.set_device_memory(buf.data_ptr(), buf.numel())
        else:
            ctx.set_device_memory(buf.data_ptr(), buf.numel())


class TRTModule(torch.nn.Module):
    """Callable like TFLiteModule: x [N,H,W,3] -> list of outputs (NHWC like TFLite)."""

    def __init__(self, base, outputs, size, fp16, max_batch=1024, opt_batch=512, device="cuda"):
        super().__init__()
        self.dev = torch.device(device)
        with torch.cuda.device(self.dev):
            path = build_engine(base, outputs, size, fp16, max_batch, opt_batch, self.dev)
            self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(open(path, "rb").read())
            self.ctx = self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            _SharedActivations.attach(self.dev, self.ctx, self.engine.device_memory_size_v2)
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.outs = sorted([n for n in names if n != "x"], key=lambda n: int(n[1:]))

    def forward(self, x):
        x = x.float().contiguous()  # engine I/O is float32; fp16 (if enabled) is internal
        self.ctx.set_input_shape("x", tuple(x.shape))
        self.ctx.set_tensor_address("x", x.data_ptr())
        outs = []
        for n in self.outs:
            shp = tuple(self.ctx.get_tensor_shape(n))
            o = torch.empty(shp, device=x.device, dtype=torch.float32)
            self.ctx.set_tensor_address(n, o.data_ptr())
            outs.append(o)
        self.ctx.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream)
        return outs
