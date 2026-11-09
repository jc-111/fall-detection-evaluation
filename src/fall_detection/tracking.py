"""Adapter for the inherited Kalman/IoU (Deep SORT-style) tracker."""
import numpy as np
import torch

from . import _vendor  # noqa: F401


def keypoint_box(keypoints, margin=20):
    """Box around all keypoints, expanded by `margin` pixels (legacy `kpt2bbox`)."""
    return np.array((keypoints[:, 0].min() - margin, keypoints[:, 1].min() - margin,
                     keypoints[:, 0].max() + margin, keypoints[:, 1].max() + margin))


class PersonTracker:
    def __init__(self, cfg):
        from Track.Tracker import Tracker  # inherited

        self._tracker = Tracker(max_iou_distance=cfg.max_iou_distance, max_age=cfg.max_age,
                                n_init=cfg.n_init)

    def predict(self):
        """Advance every track one frame with its Kalman filter."""
        self._tracker.predict()

    def predicted_boxes(self):
        """Kalman-predicted boxes of existing tracks as extra detections, (N, 7) or None.

        The legacy pipeline feeds these to the pose model alongside real detections so a briefly
        missed person can still get a pose; pose-NMS then merges duplicates.
        """
        rows = [t.to_tlbr().tolist() + [0.5, 1.0, 0.0] for t in self._tracker.tracks]
        return torch.tensor(rows, dtype=torch.float32) if rows else None

    def update(self, poses):
        from Track.Tracker import Detection  # inherited

        detections = [Detection(keypoint_box(p["keypoints"].numpy()),
                                np.concatenate((p["keypoints"].numpy(), p["kp_score"].numpy()), axis=1),
                                p["kp_score"].mean().numpy())
                      for p in poses]
        self._tracker.update(detections)

    def visible_tracks(self, include_tentative=False):
        """Tracks matched to a detection in the current frame (upstream Track objects).

        By default only confirmed tracks. Dataset extraction also records tentative ones, because
        their keypoints already enter the track's 30-frame buffer used once it is confirmed. A track
        created this frame is excluded: upstream does not buffer the detection that creates it.
        """
        return [t for t in self._tracker.tracks
                if t.time_since_update == 0 and len(t.keypoints_list) > 0
                and (include_tentative or t.is_confirmed())]
