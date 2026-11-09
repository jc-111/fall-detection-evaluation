"""Draws a FrameResult onto an original-resolution BGR frame."""
import cv2
import numpy as np

from . import _vendor  # noqa: F401
from fn import draw_single  # inherited skeleton drawing (upstream fn.py)

LABEL_COLORS = {"Fall Down": (0, 0, 255), "Lying Down": (0, 165, 255)}  # BGR
DEFAULT_COLOR = (0, 200, 0)
PENDING_COLOR = (200, 200, 200)


def draw_frame(frame_bgr, frame_result):
    out = frame_bgr.copy()
    scale = max(out.shape[:2]) / 640
    for p in frame_result.persons:
        out = draw_single(out, np.asarray(p.keypoints, dtype=np.float32))
        x1, y1, x2, y2 = (int(round(v)) for v in p.bbox)
        color = LABEL_COLORS.get(p.label, DEFAULT_COLOR) if p.label else PENDING_COLOR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, int(2 * scale)))
        text = f"#{p.track_id} {p.label or 'pending'}"
        cv2.putText(out, text, (x1, max(0, y1 - int(6 * scale))), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, color, max(1, int(scale)))
    cv2.putText(out, f"frame {frame_result.frame_index}  t={frame_result.timestamp_s:.2f}s",
                (10, int(20 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255),
                max(1, int(scale)))
    return out
