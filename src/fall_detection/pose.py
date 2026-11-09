"""Adapter for the inherited AlphaPose SPPE FastPose single-person pose estimator."""
from . import _vendor  # noqa: F401


class PoseEstimator:
    def __init__(self, cfg, device="cpu"):
        from PoseEstimateLoader import SPPE_FastPose  # inherited

        self._model = SPPE_FastPose(cfg.backbone, cfg.input_height, cfg.input_width,
                                    device=device, weights_file=str(cfg.weights))

    def estimate(self, frame, boxes, box_scores):
        """Pose for each box in a letterboxed RGB frame, after pose-NMS merging of overlapping boxes.

        Returns a list of dicts with 'keypoints' (13, 2), 'kp_score' (13, 1) and 'proposal_score'.
        """
        return self._model.predict(frame, boxes, box_scores)
