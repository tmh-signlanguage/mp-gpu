"""Per-GPU extraction worker.

Pipeline on one GPU:
  decode   NVDEC (PyNvVideoCodec, NV12 in device memory) for H.264/HEVC/VP9/AV1 8-bit 4:2:0, which
           decode bit-exactly; PyAV rgb24 on the CPU for everything else
  crop     fused Triton kernel: exact libswscale yuv->rgb24 (LUT) + OpenCV-exact warpPerspective
  models   TensorRT (or PyTorch), S video streams batched in lock-step
  output   <out>.npz (lossless numpy, see pose_io) and/or <out>.pose (pose_format)

Videos come from a queue shared by all GPU processes; progress/events go to a reporting queue.
"""
import collections, ctypes, os, queue, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import av
import PyNvVideoCodec as nvc

from .holistic import HolisticGPU
from .crop_triton import NV12Table
from .yuv_lut import get_lut_gpu, lut_key
from .pose_io import save_npz, save_pose

# Codecs whose decoding is bit-exact by specification, so NVDEC output equals libavcodec's.
# MPEG-4 Part 2 / MPEG-2 / MJPEG etc. allow IDCT mismatch (and PyNvVideoCodec dropped the first
# MPEG-4 frame in .mov files); 10-bit / 4:2:2 do not fit the NV12 path. Those use the CPU path,
# which by construction sees the same rgb24 frames as PyAV/pose_format.
NVDEC_CODECS = {"h264", "hevc", "vp9", "av1"}
NVDEC_PIXFMTS = {"yuv420p", "yuvj420p"}
STOP = "__STOP__"


def cuda_context():
    torch.zeros(1, device="cuda")
    ctx = ctypes.c_void_p()
    assert ctypes.CDLL("libcuda.so.1").cuCtxGetCurrent(ctypes.byref(ctx)) == 0
    return ctx.value


def video_meta(path):
    """dict(W, H, fps, n, key, codec, pix_fmt) from the container / first frame (PyAV, cheap)."""
    c = av.open(path)
    s = c.streams.video[0]
    fps = float(s.average_rate or 25)
    n = int(s.frames or 0)
    f = next(c.decode(s))
    m = dict(W=f.width, H=f.height, fps=fps, n=n, key=lut_key(f), codec=s.codec_context.name, pix_fmt=f.format.name)
    c.close()
    return m


def discarded_frames(path, n_frames):
    """Display-order indices of frames that libavcodec/PyAV drops because the container's edit list marks
    their packets as discard (common in MP4/MOV: a leading or trailing frame). NVDEC's demuxer still
    decodes them, so they are skipped to keep frame indices identical to PyAV.
    Discards only occur at the start/end, so only the first packets and the last ~15 s are demuxed."""
    c = av.open(path)
    try:
        if "mov" not in c.format.name and "mp4" not in c.format.name:
            return set()                        # edit lists are an MP4/MOV feature
        s = c.streams.video[0]

        def scan(limit=None):
            out = []
            for p in c.demux(s):
                if p.size > 0 and p.pts is not None:
                    out.append((p.pts, p.is_discard))
                    if limit and len(out) >= limit:
                        break
            return out
        if not n_frames or n_frames <= 600 or not s.duration:
            pk = sorted(scan())
            return {i for i, (_, d) in enumerate(pk) if d}
        head = sorted(scan(200))
        disc = {i for i, (_, d) in enumerate(head) if d}
        c.seek(max(0, s.duration - int(15 / s.time_base)), stream=s, backward=True, any_frame=False)
        tail = sorted(scan())
        disc |= {n_frames - len(tail) + i for i, (_, d) in enumerate(tail) if d}
        return disc
    finally:
        c.close()


class MemBudget:
    """GPU-memory admission control for open videos. NVDEC needs ~28 B/pixel per decoder (surfaces), plus
    ~6 B/pixel for queued NV12 frames, so e.g. 256 x 1080p streams would need ~18 GB.
    A video is opened only if its cost fits; at least one video is always admitted."""

    BYTES_PER_PIXEL = 36

    def __init__(self, budget_bytes):
        self.budget, self.used, self.n, self.lock = budget_bytes, 0, 0, threading.Lock()

    def acquire(self, w, h, bpp=BYTES_PER_PIXEL):
        c = w * h * bpp
        with self.lock:
            if self.n and self.used + c > self.budget:
                return 0
            self.used += c; self.n += 1
            return c

    def release(self, c):
        if c:
            with self.lock:
                self.used -= c; self.n -= 1


class Todo:
    """Work source for this process. A feeder thread pulls videos from the shared queue (which ends with a
    STOP sentinel per process) and a small pool prepares them ahead of time (probe + edit-list scan), so
    opening a video in a decode thread never stalls on slow I/O. Videos deferred because the memory
    budget was full are kept locally."""

    def __init__(self, q, opt, ahead=32, n_prep=6):
        self.q, self.opt, self.ahead = q, opt, ahead
        self.ready, self.deferred = collections.deque(), []
        self.exhausted, self.pending = False, 0
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(n_prep)
        threading.Thread(target=self._feed, daemon=True).start()

    def _prep(self, item):
        try:
            m = video_meta(item["path"])
            dec = self.opt["decoder"]
            m["nvdec"] = dec == "nvdec" or (dec == "auto" and m["codec"] in NVDEC_CODECS and m["pix_fmt"] in NVDEC_PIXFMTS)
            m["discard"] = discarded_frames(item["path"], m["n"]) if m["nvdec"] else set()
            item = dict(item, meta=m)
        except Exception as e:  # noqa
            item = dict(item, error=e)
        with self.lock:
            self.ready.append(item); self.pending -= 1

    def _feed(self):
        while True:
            with self.lock:
                room = len(self.ready) + self.pending < self.ahead
            if not room:
                time.sleep(0.05); continue
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item == STOP:
                with self.lock:
                    self.exhausted = True
                return
            with self.lock:
                self.pending += 1
            self.pool.submit(self._prep, item)

    def pop(self):
        with self.lock:
            if self.deferred:
                return self.deferred.pop()
            return self.ready.popleft() if self.ready else None

    def defer(self, item):
        with self.lock:
            self.deferred.append(item)

    def done(self):
        with self.lock:
            return self.exhausted and not self.pending and not self.ready and not self.deferred


class Slot:
    def __init__(self):
        self.q = collections.deque()     # items: ("start", meta) | frame tensor | ("end", err)


class _Src:
    """Per-slot decoding state. kind "nvdec": NV12 frames from NVDEC; kind "cpu": rgb24 frames from PyAV."""

    def __init__(self, kind, it, dec, step, nfr, cost, container=None, opencv=False, discard=()):
        self.kind, self.it, self.dec, self.step, self.nfr, self.cost = kind, it, dec, step, nfr, cost
        self.container, self.opencv, self.discard = container, opencv, discard
        self.pend = collections.deque()
        self.raw_i = 0      # decoder output index (display order), before edit-list discards
        self.src_i = 0      # source frame index (as PyAV sees it)
        self.out_i = 0      # output frame index

    def keep(self):
        """Frame selection for --target-fps: output i <- source round(i * src_fps / target_fps)."""
        j = self.src_i; self.src_i += 1
        if self.step is None:
            return True
        if self.nfr and self.out_i >= max(1, round(self.nfr / self.step)):   # n_out cap
            return False
        want = round(self.out_i * self.step)
        if self.nfr:
            want = min(want, self.nfr - 1)
        if j != want:
            return False
        self.out_i += 1
        return True


class DecodeWorker(threading.Thread):
    """Owns a set of slots; keeps each slot's queue filled with up to `depth` decoded frames."""

    def __init__(self, slots, todo, ctx, luts, opt, budget, report):
        super().__init__(daemon=True)
        self.slots, self.todo, self.ctx, self.luts, self.opt, self.budget, self.report = \
            slots, todo, ctx, luts, opt, budget, report
        self.state = {id(s): None for s in slots}
        self.finished = False
        self.stop = threading.Event()
        self.start()

    def _open(self, slot):
        item = self.todo.pop()
        if item is None:
            return False
        path, opt, cost = item["path"], self.opt, 0
        try:
            if "error" in item:
                raise item["error"]
            m = item["meta"]
            W, H = m["W"], m["H"]
            use_nvdec = m["nvdec"]
            cost = self.budget.acquire(W, H, MemBudget.BYTES_PER_PIXEL if use_nvdec else 3 * (opt["depth"] + 1))
            if not cost:                       # over the decode-memory budget: retry later
                self.todo.defer(item)
                return None
            # "pyav": honour colour tags like PyAV/pose_format; "opencv": cv2.VideoCapture always uses BT.601
            lut = self.luts(m["key"] if opt["color"] == "pyav" else (2, 0))
            tfps = opt["target_fps"]
            step = m["fps"] / tfps if tfps and m["fps"] > tfps + 1e-3 else None
            src = None
            if use_nvdec:
                try:
                    dmx = nvc.CreateDemuxer(filename=path)
                    dec = nvc.CreateDecoder(gpuid=0, codec=dmx.GetNvCodecId(), cudacontext=self.ctx,
                                            cudastream=torch.cuda.current_stream().cuda_stream, usedevicememory=True)
                    src = _Src("nvdec", iter(dmx), dec, step, m["n"], cost, discard=m["discard"])
                except Exception as e:  # noqa
                    if opt["decoder"] == "nvdec":
                        raise
                    self.report(("warn", path, f"NVDEC open failed ({e}); using CPU decoding"))
            if src is None:
                c = av.open(path)
                vs = c.streams.video[0]
                vs.thread_type = "AUTO"
                src = _Src("cpu", c.decode(vs), None, step, m["n"], cost, container=c, opencv=opt["color"] == "opencv")
            self.state[id(slot)] = src
            self.report(("open", path))
            item = {k: v for k, v in item.items() if k != "meta"}
            slot.q.append(("start", dict(item, W=W, H=H, fps=(tfps if step else m["fps"]), lut=lut, decoder=src.kind)))
        except Exception as e:  # noqa
            self.budget.release(cost)
            item = {k: v for k, v in item.items() if k not in ("meta", "error")}
            slot.q.append(("start", dict(item, W=1, H=1, fps=25, lut=None, decoder="none")))
            slot.q.append(("end", e))
        return True

    def _close(self, slot):
        st = self.state[id(slot)]
        if st is not None:
            self.budget.release(st.cost)
            if st.container is not None:
                st.container.close()
        self.state[id(slot)] = None

    def _pump(self, slot):
        """Decode until one frame is queued for this slot (or the video ends)."""
        st = self.state[id(slot)]
        if st.kind == "cpu":
            for f in st.it:
                if not st.keep():
                    continue
                if st.opencv:                     # cv2.VideoCapture converts with BT.601, ignoring tags
                    f.colorspace = 2; f.color_range = 0
                slot.q.append(torch.from_numpy(f.to_ndarray(format="rgb24")).to("cuda"))
                return
            slot.q.append(("end", None))
            self._close(slot)
            return
        while not st.pend:
            pkt = next(st.it, None)
            frames = st.dec.Decode(pkt if pkt is not None else nvc.PacketData())
            for f in frames:
                r = st.raw_i; st.raw_i += 1
                if r in st.discard:              # edit-list discard: PyAV never outputs this frame
                    continue
                if st.keep():
                    st.pend.append(torch.from_dlpack(f).clone())
            if pkt is None:
                if not st.pend:
                    slot.q.append(("end", None))
                    self._close(slot)
                    return
                break
        slot.q.append(st.pend.popleft())

    def run(self):
        torch.cuda.set_device(0)
        depth = self.opt["depth"]
        while not self.stop.is_set():
            busy = False
            can_open = True
            for s in self.slots:
                if len(s.q) >= depth:
                    continue
                if self.state[id(s)] is None:
                    if not can_open:
                        continue
                    r = self._open(s)
                    if r is None:                   # budget full: leave idle slots idle this round
                        can_open = False
                        continue
                    if not r:
                        can_open = False            # nothing to open right now
                        continue
                    if self.state[id(s)] is None:   # open failed; its error is already queued
                        busy = True
                        continue
                try:
                    self._pump(s)
                except Exception as e:  # noqa  decoder error: fail this video only, keep the thread alive
                    s.q.append(("end", e))
                    self._close(s)
                busy = True
            if not busy:
                if all(self.state[id(s)] is None for s in self.slots) and self.todo.done():
                    self.finished = True
                    return
                time.sleep(0.001)


def pack_outputs(out, wh):
    """Step outputs -> data [S,576,3], conf [S,576] in pose_format holistic layout (on GPU)."""
    W = wh[:, 0].float()[:, None]; H = wh[:, 1].float()[:, None]

    def xy(p, ok, conf=None):
        d = torch.stack([p[..., 0] * W, p[..., 1] * H, p[..., 2]], -1) * ok[:, None, None]
        c = (conf if conf is not None else torch.ones_like(p[..., 0])) * ok[:, None]
        return d, c
    b, bc = xy(out["pose"], out["pose_ok"], out["pose"][..., 3])
    w, wc = xy(out["world"], out["pose_ok"], out["world"][..., 3])
    f, fc = xy(out["face"], out["face_ok"])
    l, lc = xy(out["lhand"], out["lhand_ok"])
    r, rc = xy(out["rhand"], out["rhand_ok"])
    return torch.cat([b, f, l, r, w], 1), torch.cat([bc, fc, lc, rc, wc], 1)


def write_outputs(out_base, fmt, data, conf, fps, w, h, extra=None):
    """out_base: output path without the .npz/.pose suffix. extra: additional npz arrays."""
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    if fmt in ("npz", "both"):
        save_npz(out_base + ".npz", data, conf, fps, w, h, **(extra or {}))
    if fmt in ("pose", "both"):
        save_pose(out_base + ".pose", data, conf, fps, w, h)


def run_worker(work_q, report_q, opt):
    """Process videos from work_q (dicts with path/out) on the (single) visible GPU until STOP."""
    def report(msg):
        report_q.put((opt["rank"],) + tuple(msg))

    dev = torch.device("cuda")
    ctx = cuda_context()
    S = opt["streams"]
    hol = HolisticGPU(S, device=dev, half=opt["half"], backend=opt["backend"])
    report(("ready",))
    lut_cache, lut_lock = {}, threading.Lock()

    def luts(key):
        with lut_lock:
            if key not in lut_cache:
                lut_cache[key] = get_lut_gpu(*key, dev)
            return lut_cache[key]

    todo = Todo(work_q, opt)
    slots = [Slot() for _ in range(S)]
    budget = MemBudget(int(opt["decode_mem_gb"] * 2 ** 30))
    nw = opt["decode_threads"]
    workers = [DecodeWorker(slots[k::nw], todo, ctx, luts, opt, budget, report) for k in range(nw)]
    # The pipeline runs on its own stream (TensorRT dislikes the default stream); decode threads use the
    # default stream, so each step waits on it and frames are marked as used by the pipeline stream.
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    writer = ThreadPoolExecutor(4)
    meta = [None] * S
    res = [None] * S
    psamp = [None] * S           # per stream: [(t, n_persons)] for the multi-person check
    p_int = opt.get("persons_interval") or 0.0
    sec_acc = 0.0; frames_acc = 0; last = time.time()

    def finish(m, parts, ps):
        try:
            d = np.concatenate([x[0] for x in parts]); c = np.concatenate([x[1] for x in parts])
            extra = None
            if p_int > 0:
                pt = np.array([t for t, _ in ps], np.float32); pn = np.array([n for _, n in ps], np.int8)
                extra = dict(persons_t=pt, persons_n=pn)
                share = float((pn >= 2).mean()) if len(pn) else 0.0
                if len(pn) and share >= opt["persons_min_share"]:
                    report(("persons", m["path"], share, int(pn.max()), len(pn)))
            write_outputs(m["out"], opt["format"], d, c, m["fps"], m["W"], m["H"], extra)
            report(("done", m["path"], None))
        except Exception as e:  # noqa
            report(("done", m["path"], f"write failed: {e}"))

    while True:
        frames = [None] * S
        fresh = torch.zeros(S, dtype=torch.bool)
        for s, sl in enumerate(slots):
            while True:
                if not sl.q:
                    if meta[s] is None:     # idle slot: don't stall the batch waiting for a new video
                        break
                    time.sleep(0.0002)      # mid-video: wait for the decoder
                    continue
                item = sl.q.popleft()
                if isinstance(item, tuple) and item[0] == "start":
                    meta[s] = item[1]; res[s] = []; psamp[s] = []; fresh[s] = True
                    meta[s]["k"] = 0; meta[s]["next_check"] = 0.0
                    continue
                if isinstance(item, tuple) and item[0] == "end":
                    m = meta[s]
                    if item[1] is not None:
                        report(("done", m["path"], f"{type(item[1]).__name__}: {item[1]}"))
                    elif res[s]:
                        writer.submit(finish, m, res[s], psamp[s])
                    else:
                        report(("done", m["path"], "no frames decoded"))
                    meta[s] = None; res[s] = None
                    continue
                frames[s] = item
                break
        active_l = [f is not None for f in frames]
        now = time.time()
        if now - last > 0.5:
            report(("tick", sec_acc, frames_acc, sum(m is not None for m in meta)))
            sec_acc = 0.0; frames_acc = 0; last = now
        if not any(active_l):
            if all(w.finished for w in workers) and all(not sl.q for sl in slots):
                break
            time.sleep(0.001)
            continue
        stream.wait_stream(torch.cuda.default_stream())
        for f in frames:
            if f is not None:
                f.record_stream(stream)
        active = torch.tensor(active_l).to(dev)
        wh = torch.tensor([[meta[s]["W"], meta[s]["H"]] if active_l[s] else [1, 1] for s in range(S)],
                          dtype=torch.int32).to(dev)
        table = NV12Table.from_list(frames, [meta[s]["H"] if active_l[s] else 0 for s in range(S)],
                                    [meta[s]["lut"] if active_l[s] else None for s in range(S)], dev)
        hol.reset(fresh.to(dev) & active)
        out = hol.step(table, wh, active)
        d, c = pack_outputs(out, wh)
        check = []                       # streams whose current frame is due for the multi-person check
        if p_int > 0:
            for s in range(S):
                if active_l[s] and meta[s]["k"] / meta[s]["fps"] >= meta[s]["next_check"]:
                    check.append(s)
            if check:
                npers = hol.count_persons(table, wh, torch.tensor(check, device=dev),
                                          opt["persons_min_rel"]).tolist()
        d = d.cpu().numpy(); c = c.cpu().numpy()
        for s, n in zip(check, npers if check else ()):
            t = meta[s]["k"] / meta[s]["fps"]
            psamp[s].append((t, n))
            meta[s]["next_check"] = t + p_int
        for s in range(S):
            if active_l[s]:
                meta[s]["k"] += 1
        for s in range(S):
            if active_l[s]:
                res[s].append((d[s:s + 1].copy(), c[s:s + 1].copy()))
                sec_acc += 1.0 / meta[s]["fps"]
                frames_acc += 1
    report(("tick", sec_acc, frames_acc, 0))
    for w in workers:
        w.stop.set(); w.join()
    writer.shutdown(wait=True)
    report(("exit",))
