"""Pose output formats.

npz (default): lossless, numpy-only container of the same data a holistic .pose file holds.
  data    float32 [T, 576, 3]  x, y in pixels, z as in pose_format (body / face / left_hand /
                               right_hand / world, same order as pose_format.utils.holistic)
  conf    float32 [T, 576]     visibility (body/world) or 1/0 presence (face/hands)
  fps, width, height           scalars
  components  str [C]          component names, in order
  points      str [576]        "COMPONENT/POINT" per column
No pickle is needed to read it:  z = np.load(path); z["data"], z["conf"], ...

.pose: standard pose_format file (pose_format.utils.holistic layout).
"""
import numpy as np


def _header(width, height):
    from pose_format.pose_header import PoseHeader, PoseHeaderDimensions
    from pose_format.utils.holistic import holistic_components
    return PoseHeader(version=0.2, dimensions=PoseHeaderDimensions(width=width, height=height, depth=0),
                      components=holistic_components("XYZC", 0))


_NAMES = None


def holistic_names():
    """(component names [C], point names [576]) for the holistic layout."""
    global _NAMES
    if _NAMES is None:
        comps = _header(1, 1).components
        _NAMES = (np.array([c.name for c in comps]),
                  np.array([f"{c.name}/{p}" for c in comps for p in c.points]))
    return _NAMES


def save_npz(path, data, conf, fps, width, height, compress=True):
    comps, points = holistic_names()
    assert data.shape[1] == len(points), (data.shape, len(points))
    (np.savez_compressed if compress else np.savez)(
        path, data=data.astype(np.float32), conf=conf.astype(np.float32),
        fps=np.float32(fps), width=np.int32(width), height=np.int32(height),
        components=comps, points=points)


def save_pose(path, data, conf, fps, width, height):
    from pose_format import Pose
    from pose_format.numpy.pose_body import NumPyPoseBody
    pose = Pose(_header(width, height), NumPyPoseBody(fps=fps, data=data[:, None], confidence=conf[:, None]))
    with open(path, "wb") as f:
        pose.write(f)


def load(path):
    """Load .npz or .pose -> dict(data [T,576,3], conf [T,576], fps, width, height)."""
    if str(path).endswith(".npz"):
        z = np.load(path)
        return dict(data=z["data"], conf=z["conf"], fps=float(z["fps"]), width=int(z["width"]), height=int(z["height"]))
    from pose_format import Pose
    p = Pose.read(open(path, "rb").read())
    return dict(data=np.asarray(p.body.data[:, 0].filled(0), np.float32), conf=np.asarray(p.body.confidence[:, 0], np.float32),
                fps=float(p.body.fps), width=p.header.dimensions.width, height=p.header.dimensions.height)


def npz_to_pose(path):
    """npz -> pose_format.Pose (to use pose_format tools: normalize, visualize, component selection)."""
    from pose_format import Pose
    from pose_format.numpy.pose_body import NumPyPoseBody
    d = load(path)
    return Pose(_header(d["width"], d["height"]),
                NumPyPoseBody(fps=d["fps"], data=d["data"][:, None], confidence=d["conf"][:, None]))


def pose_to_npz(pose_path, npz_path):
    d = load(pose_path)
    save_npz(npz_path, d["data"], d["conf"], d["fps"], d["width"], d["height"])
