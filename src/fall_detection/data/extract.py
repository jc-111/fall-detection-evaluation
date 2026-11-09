"""Canonical pose extraction for datasets.

Runs the inference pipeline's own extraction stage (`FallDetectionPipeline.extract_frame`) over a
video and stores every buffered track observation, so classifier windows rebuilt offline are the
exact windows the live pipeline classifies (see windows.py).

Coordinates are stored in letterboxed-frame pixels, which is what the models consume; the
letterbox geometry is stored alongside so original-frame pixels are recoverable
(`frames.Letterbox.to_original`).
"""
import hashlib
import json
import platform
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..config import REPO_ROOT

EXTRACT_FORMAT = 1


@lru_cache(maxsize=None)
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version():
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", "src", "third_party", "configs"))}


def extractor_record(config):
    """Everything that determines extracted numbers, plus an `extractor_id` hashing the numeric part."""
    import torch
    det, pose, trk = config.detector, config.pose, config.tracker
    numeric = {
        "format": EXTRACT_FORMAT,
        "frame": {"size": config.frame.size, "border": config.frame.border},
        "detector": {"name": "tiny-yolov3-onecls", "cfg_sha256": file_sha256(det.cfg_file),
                     "weights_sha256": file_sha256(det.weights), "conf_threshold": det.conf_threshold,
                     "nms_threshold": det.nms_threshold, "expand_bbox": det.expand_bbox},
        "pose": {"name": "alphapose-sppe-fastpose", "backbone": pose.backbone,
                 "weights_sha256": file_sha256(pose.weights),
                 "input_hw": [pose.input_height, pose.input_width]},
        "tracker": {"max_age": trk.max_age, "n_init": trk.n_init, "max_iou_distance": trk.max_iou_distance},
        "device": config.device,
        "versions": {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                     "python": platform.python_version()},
    }
    extractor_id = hashlib.sha256(json.dumps(numeric, sort_keys=True).encode()).hexdigest()[:12]
    return {"extractor_id": extractor_id, "config_name": config.name, **numeric, "code": code_version()}


def extract_video(pipeline, video_path, video_id):
    """Run the extraction stage over a whole video. Returns (arrays, meta)."""
    pipeline.reset()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = {"frame_index": [], "track_id": [], "confirmed": [], "keypoints": [], "bbox": []}
    n, lb, t0 = 0, None, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, lb = pipeline.extract_frame(frame)
        for t in pipeline.tracker.visible_tracks(include_tentative=True):
            rows["frame_index"].append(n)
            rows["track_id"].append(t.track_id)
            rows["confirmed"].append(t.is_confirmed())
            rows["keypoints"].append(np.asarray(t.keypoints_list[-1], dtype=np.float32))
            rows["bbox"].append(t.to_tlbr().astype(np.float32))
        n += 1
    elapsed = time.time() - t0
    cap.release()
    arrays = {
        "frame_index": np.asarray(rows["frame_index"], dtype=np.int32),
        "track_id": np.asarray(rows["track_id"], dtype=np.int32),
        "confirmed": np.asarray(rows["confirmed"], dtype=bool),
        "keypoints": np.asarray(rows["keypoints"], dtype=np.float32).reshape(-1, 13, 3),
        "bbox": np.asarray(rows["bbox"], dtype=np.float32).reshape(-1, 4),
    }
    meta = {"video_id": video_id, "video_path": str(video_path), "video_sha256": file_sha256(str(video_path)),
            "fps": fps, "n_frames": n, "width": width, "height": height,
            "letterbox": None if lb is None else {"ratio": lb.ratio, "left": lb.left, "top": lb.top, "size": lb.size},
            "extractor": extractor_record(pipeline.config), "elapsed_s": round(elapsed, 2),
            "coordinates": "letterboxed-frame pixels; keypoints are [x, y, confidence] in features.POSE_JOINTS order"}
    return arrays, meta


def save_extraction(path, arrays, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load_extraction(path):
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return arrays, meta
