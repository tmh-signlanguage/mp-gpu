"""mp-gpu: batched GPU re-implementation of legacy MediaPipe Holistic.

Library use:
    from mp_gpu.holistic import HolisticGPU      # batched per-frame API (frames already on the GPU)
    from mp_gpu.pose_io import load, npz_to_pose  # read outputs
Command line: mp-gpu --help
"""
__version__ = "0.1.0"
