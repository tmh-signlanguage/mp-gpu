"""Cache locations and cross-process build locks (TensorRT engines, YUV LUTs)."""
import contextlib, fcntl, os


def cache_dir(*sub):
    """$MP_GPU_CACHE or ~/.cache/mp-gpu, plus optional subdirectories (created)."""
    root = os.environ.get("MP_GPU_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "mp-gpu")
    d = os.path.join(root, *sub)
    os.makedirs(d, exist_ok=True)
    return d


@contextlib.contextmanager
def file_lock(path):
    """Exclusive lock on path + '.lock' (several GPU processes may want to build the same file)."""
    with open(path + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
