"""Skeleton-sequence preprocessing: the single definition of what the action classifier sees.

Matches the legacy inference path (`ActionsEstLoader.predict`) and the legacy dataset builder
(`create_dataset_3.py`): per-frame min-max scaling of x/y to [-1, 1], then a synthetic neck
joint, then a frame-difference motion stream. Any training or evaluation code must use this module.
"""
import numpy as np
import torch

from . import _vendor  # noqa: F401
from pose_utils import normalize_points_with_size, scale_pose  # inherited (upstream pose_utils.py)

# Output order of the inherited SPPE pose model after eyes/ears are dropped.
POSE_JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
               "right_knee", "left_ankle", "right_ankle")
GRAPH_NODES = POSE_JOINTS + ("neck",)  # node layout of the 'coco_cut' ST-GCN graph
CHANNELS = 3  # x, y, confidence


def add_neck(pts):
    """Append a neck node (mean of both shoulders, all channels) as node 13. (T,13,C) -> (T,14,C)."""
    neck = (pts[:, 1, :] + pts[:, 2, :]) / 2
    return np.concatenate((pts, neck[:, None, :]), axis=1)


def normalize_sequence(keypoints, image_size):
    """(T,13,3) keypoints in pixel coordinates of an image of `image_size` = (width, height)
    -> (T,14,3) float32 features: x/y scaled per frame to [-1, 1], confidence unchanged, plus neck."""
    pts = np.array(keypoints, dtype=np.float32, copy=True)
    if pts.ndim != 3 or pts.shape[1:] != (len(POSE_JOINTS), CHANNELS):
        raise ValueError(f"expected (T, {len(POSE_JOINTS)}, {CHANNELS}) keypoints, got {pts.shape}")
    pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
    pts[:, :, :2] = scale_pose(pts[:, :, :2])
    return add_neck(pts)


def to_model_inputs(features):
    """(N,T,14,3) or (T,14,3) features -> (points (N,3,T,14), motion (N,2,T-1,14)) tensors."""
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if x.dim() == 3:
        x = x[None]
    pts = x.permute(0, 3, 1, 2).contiguous()
    mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
    return pts, mot
