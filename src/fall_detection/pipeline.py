"""Video -> person detection -> pose -> tracking -> 30-frame skeleton window -> action scores.

The per-frame order reproduces the deployed legacy web app (legacy/web_app/detect/main.py); the deliberate
differences are listed in docs/BASELINE.md.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from .classifier import ActionClassifier
from .config import load_config
from .detector import PersonDetector
from .frames import letterbox
from .outputs import FrameResult, PersonResult, VideoResult
from .pose import PoseEstimator
from .tracking import PersonTracker


class FallDetectionPipeline:
    def __init__(self, config):
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        cfg = self.config
        if cfg.classifier.window != 30:
            raise ValueError("the inherited tracker buffers exactly 30 keypoint frames per track")
        torch.set_grad_enabled(False)
        self.detector = PersonDetector(cfg.detector, cfg.frame.size, cfg.device)
        self.pose = PoseEstimator(cfg.pose, cfg.device)
        self.classifier = ActionClassifier(cfg.classifier, cfg.device)
        self.tracker = None
        self.reset()

    def reset(self):
        """Forget all tracks (call between unrelated videos)."""
        self.tracker = PersonTracker(self.config.tracker)

    def extract_frame(self, frame_bgr):
        """Extraction stage only (letterbox -> detect -> pose -> track); shared with dataset building.

        Returns (letterboxed RGB image, Letterbox geometry). Track state is updated in self.tracker.
        """
        cfg = self.config
        image, lb = letterbox(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), cfg.frame.size, cfg.frame.border)

        boxes = self.detector.detect(image)
        self.tracker.predict()
        predicted = self.tracker.predicted_boxes()
        if predicted is not None:
            boxes = predicted if boxes is None else torch.cat([boxes, predicted], dim=0)
        poses = self.pose.estimate(image, boxes[:, 0:4], boxes[:, 4]) if boxes is not None else []
        self.tracker.update(poses)
        return image, lb

    def process_frame(self, frame_bgr, frame_index, fps):
        image, lb = self.extract_frame(frame_bgr)
        result = FrameResult(frame_index=frame_index, timestamp_s=round(frame_index / fps, 4))
        for track in self.tracker.visible_tracks():
            label = scores = None
            if len(track.keypoints_list) == self.classifier.window:
                s = self.classifier.score_window(np.array(track.keypoints_list, dtype=np.float32),
                                                 (image.shape[1], image.shape[0]))
                label = self.classifier.label(s)
                scores = {c: float(v) for c, v in zip(self.classifier.class_names, s)}
            kp = np.asarray(track.keypoints_list[-1], dtype=np.float64)
            kp_xy = lb.to_original(kp[:, :2])
            result.persons.append(PersonResult(
                track_id=int(track.track_id),
                bbox=[round(float(v), 1) for v in lb.box_to_original(track.to_tlbr())],
                keypoints=[[round(float(x), 1), round(float(y), 1), round(float(c), 4)]
                           for (x, y), c in zip(kp_xy, kp[:, 2])],
                label=label, scores=scores))
        return result

    def run_video(self, path, max_frames=None, on_frame=None):
        """Process a video file. `on_frame(frame_bgr, frame_result)` is called for every frame."""
        self.reset()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        result = VideoResult(video=str(path), fps=fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                             class_names=list(self.classifier.class_names),
                             config=self.config.to_dict())
        try:
            while max_frames is None or result.frame_count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                fr = self.process_frame(frame, result.frame_count, fps)
                result.frames.append(fr)
                if on_frame:
                    on_frame(frame, fr)
        finally:
            cap.release()
        return result
