"""mp-gpu: batched GPU MediaPipe Holistic (legacy-compatible) extraction on all visible GPUs.

    mp-gpu VIDEO_OR_FOLDER [...] -o OUT_DIR [--list FILE] [options]

Folders are searched recursively and their relative layout is mirrored under OUT_DIR; single files and
--list entries are written as OUT_DIR/<file name>.npz. Existing outputs are skipped (unless --overwrite).
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor

# Keep this module light: GPU worker processes are spawned and import it before CUDA is configured.

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".mts", ".m2ts", ".ts",
              ".flv", ".wmv", ".3gp", ".ogv")


def hm(sec):
    sec = max(0, int(sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}"


def collect(inputs, list_file, out_dir, exts):
    """-> list of (video path, output base path). Folders mirror their layout under out_dir."""
    items = []
    for inp in inputs:
        if os.path.isdir(inp):
            for root, dirs, files in os.walk(inp):
                dirs.sort()
                for f in sorted(files):
                    if f.lower().endswith(exts) and not f.startswith("."):
                        p = os.path.join(root, f)
                        items.append((p, os.path.join(out_dir, os.path.relpath(p, inp))))
        elif os.path.isfile(inp):
            items.append((inp, os.path.join(out_dir, os.path.basename(inp))))
        else:
            sys.exit(f"mp-gpu: no such file or directory: {inp}")
    if list_file:
        for line in open(list_file):
            p = line.strip()
            if p and not p.startswith("#"):
                items.append((p, os.path.join(out_dir, os.path.basename(p))))
    seen, dup = {}, []
    for p, o in items:
        if o in seen and seen[o] != p:
            dup.append((seen[o], p))
        seen[o] = p
    if dup:
        msg = "\n".join(f"  {a}\n  {b}" for a, b in dup[:10])
        sys.exit(f"mp-gpu: {len(dup)} output name collision(s) (same file name in different places):\n{msg}\n"
                 "Pass the parent folder instead, so the folder layout is mirrored under the output dir.")
    return list(dict.fromkeys(items))


def probe(path):
    """Duration in seconds (0 if unknown/unreadable)."""
    import av
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            if s.duration and s.time_base:
                return float(s.duration * s.time_base)
            if c.duration:
                return c.duration / 1e6
            if s.frames and s.average_rate:
                return s.frames / float(s.average_rate)
    except Exception:  # noqa
        pass
    return 0.0


def visible_gpus(spec):
    """Physical ids (strings) of the GPUs to use: --gpus, else CUDA_VISIBLE_DEVICES, else all."""
    if spec:
        return [g.strip() for g in spec.split(",") if g.strip()]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip()]
    import subprocess
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return [str(i) for i, l in enumerate(out.splitlines()) if l.startswith("GPU ")]


def _gpu_main(phys_id, work_q, report_q, opt):
    try:  # die with the parent (no orphaned GPU workers if mp-gpu is killed)
        import ctypes, signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except Exception:  # noqa
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = phys_id
    try:
        from .extract import run_worker
        run_worker(work_q, report_q, opt)
    except BaseException as e:  # noqa
        import traceback
        report_q.put((opt["rank"], "fatal", traceback.format_exc()))
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mp-gpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="video files and/or folders (searched recursively)")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--list", help="text file with one video path per line")
    ap.add_argument("--format", default="npz", choices=["npz", "pose", "both"],
                    help="npz: lossless numpy (default); pose: pose_format .pose; both")
    ap.add_argument("--overwrite", action="store_true", help="re-extract videos whose output already exists")
    ap.add_argument("--gpus", help="comma-separated GPU ids (default: all visible GPUs)")
    ap.add_argument("--streams", type=int, default=256, help="videos processed concurrently per GPU (default 256)")
    ap.add_argument("--half", action="store_true", help="fp16 TensorRT engines (~20%% faster, lower fidelity)")
    ap.add_argument("--backend", default="trt", choices=["trt", "torch"])
    ap.add_argument("--decoder", default="auto", choices=["auto", "nvdec", "cpu"],
                    help="auto: NVDEC for H.264/HEVC/VP9/AV1 8-bit 4:2:0 (bit-exact), CPU (PyAV) otherwise")
    ap.add_argument("--color", default="pyav", choices=["pyav", "opencv"],
                    help="match YUV->RGB of PyAV/pose_format (honours colour tags) or of cv2.VideoCapture (BT.601)")
    ap.add_argument("--target-fps", type=float, help="subsample faster sources to this frame rate")
    ap.add_argument("--decode-mem-gb", type=float, default=10.0,
                    help="GPU memory budget for open decoders per GPU (limits concurrent high-res videos)")
    ap.add_argument("--decode-threads", type=int, default=8, help="decode threads per GPU")
    ap.add_argument("--depth", type=int, default=4, help="decoded frames buffered per stream")
    ap.add_argument("--exts", default=",".join(e[1:] for e in VIDEO_EXTS), help="video extensions for folder search")
    a = ap.parse_args(argv)
    if not a.inputs and not a.list:
        ap.error("give video files, folders and/or --list")

    exts = tuple("." + e.lower().lstrip(".") for e in a.exts.split(","))
    items = collect(a.inputs, a.list, a.out, exts)
    suffixes = {"npz": [".npz"], "pose": [".pose"], "both": [".npz", ".pose"]}[a.format]
    todo = [(p, o) for p, o in items if a.overwrite or not all(os.path.exists(o + s) for s in suffixes)]
    print(f"mp-gpu: {len(items)} videos, {len(items) - len(todo)} already done, {len(todo)} to process", flush=True)
    if not todo:
        return 0

    from tqdm import tqdm
    with ThreadPoolExecutor(32) as ex:
        durs = list(tqdm(ex.map(probe, [p for p, _ in todo]), total=len(todo), desc="probing", unit="video",
                         leave=False, dynamic_ncols=True))
    order = sorted(range(len(todo)), key=lambda i: -durs[i])      # longest first: shortens the tail
    jobs = [dict(path=todo[i][0], out=todo[i][1], dur=durs[i]) for i in order]
    total = sum(durs)

    gpus = visible_gpus(a.gpus)
    if not gpus:
        sys.exit("mp-gpu: no GPUs visible")
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    work_q, report_q = ctx.Queue(), ctx.Queue()
    for j in jobs:
        work_q.put(j)
    for _ in gpus:
        work_q.put("__STOP__")
    opt = dict(streams=a.streams, half=a.half, backend=a.backend, decoder=a.decoder, color=a.color,
               target_fps=a.target_fps, decode_mem_gb=a.decode_mem_gb, decode_threads=a.decode_threads,
               depth=a.depth, format=a.format)
    procs = []
    for r, g in enumerate(gpus):
        p = ctx.Process(target=_gpu_main, args=(g, work_q, report_q, dict(opt, rank=r)), daemon=True)
        p.start(); procs.append(p)
    print(f"mp-gpu: {hm(total)} (h:mm) of video on {len(gpus)} GPU(s) [{','.join(gpus)}], "
          f"{a.streams} streams/GPU, {'fp16' if a.half else 'fp32'}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    err_path = os.path.join(a.out, "mp-gpu-errors.log")
    dur_of = {j["path"]: j["dur"] for j in jobs}
    inflight = {}                 # path -> duration
    ndone = nerr = 0
    frames = 0
    alive = set(range(len(gpus))); ready = set()
    bar = tqdm(total=max(total, 1e-9), unit="s", dynamic_ncols=True, smoothing=0.05,
               bar_format="mp-gpu {percentage:3.0f}%|{bar}| {desc} [{elapsed}<{remaining}]")
    t0 = time.time()

    def refresh():
        el = max(time.time() - t0, 1e-6)
        bar.set_description_str(f"{hm(bar.n)}/{hm(bar.total)} video | files {ndone}/{len(jobs)}"
                            f"{f' ({nerr} failed)' if nerr else ''} | in flight {len(inflight)} ({hm(sum(inflight.values()))})"
                            f" | {frames / el:.0f} fps, {bar.n / el:.0f}x realtime", refresh=False)
        bar.refresh()

    import signal

    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)
    last = 0.0
    try:
        while alive:
            try:
                msg = report_q.get(timeout=1.0)
            except Exception:  # noqa
                for r in list(alive):
                    if not procs[r].is_alive():
                        alive.discard(r)
                        bar.write(f"mp-gpu: GPU {gpus[r]} worker exited unexpectedly (code {procs[r].exitcode})")
                refresh()
                continue
            r, kind = msg[0], msg[1]
            if kind == "tick":
                _, _, sec, nfr, _ = msg
                if bar.n + sec > bar.total:     # probed durations can be underestimates
                    bar.total = bar.n + sec
                bar.update(sec)
                frames += nfr
            elif kind == "open":
                inflight[msg[2]] = dur_of.get(msg[2], 0.0)
            elif kind == "done":
                path, err = msg[2], msg[3]
                inflight.pop(path, None)
                ndone += 1
                if err:
                    nerr += 1
                    bar.write(f"mp-gpu: ERROR {path}: {err}")
                    with open(err_path, "a") as f:
                        f.write(f"{path}\t{err}\n")
            elif kind == "warn":
                bar.write(f"mp-gpu: {msg[2]}: {msg[3]}")
            elif kind == "ready":
                ready.add(r)
                if len(ready) == len(gpus):
                    bar.write(f"mp-gpu: all {len(gpus)} GPU workers ready")
            elif kind == "fatal":
                bar.write(f"mp-gpu: GPU {gpus[r]} worker failed:\n{msg[2]}")
                alive.discard(r)
            elif kind == "exit":
                alive.discard(r)
            if time.time() - last > 0.5:
                refresh(); last = time.time()
    except KeyboardInterrupt:
        bar.write("mp-gpu: interrupted; finished videos are kept, rerun to resume")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=10)
        return 130
    finally:
        refresh(); bar.close()
    for p in procs:
        p.join(timeout=10)
    el = time.time() - t0
    print(f"mp-gpu: done {ndone - nerr}/{len(jobs)} videos ({nerr} failed) in {hm(el)}; "
          f"{hm(bar.n)} of video, {frames / max(el, 1e-6):.0f} fps"
          + (f"; errors in {err_path}" if nerr else ""), flush=True)
    return 1 if nerr or ndone < len(jobs) else 0


if __name__ == "__main__":
    sys.exit(main())
