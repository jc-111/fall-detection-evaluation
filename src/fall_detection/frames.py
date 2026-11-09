"""Frame letterboxing and coordinate mapping (project-owned).

Reimplements upstream `Detection.Utils.ResizePadding` for square targets, with the padding mode
made explicit: the deployed legacy web app used edge replication, upstream used black borders, and
the choice changes detector and pose inputs.
"""
from dataclasses import dataclass

import cv2
import numpy as np

_BORDERS = {"replicate": cv2.BORDER_REPLICATE, "constant": cv2.BORDER_CONSTANT}


@dataclass(frozen=True)
class Letterbox:
    """Maps between original-frame pixels and the size x size letterboxed frame."""
    ratio: float
    left: int
    top: int
    size: int

    def to_original(self, xy):
        xy = np.asarray(xy, dtype=np.float64).copy()
        xy[..., 0] = (xy[..., 0] - self.left) / self.ratio
        xy[..., 1] = (xy[..., 1] - self.top) / self.ratio
        return xy

    def box_to_original(self, box):
        return self.to_original(np.asarray(box, dtype=np.float64).reshape(2, 2)).reshape(4)


def letterbox(image, size, border="replicate"):
    """Resize keeping aspect ratio so the longer side equals `size`, then pad to size x size."""
    h, w = image.shape[:2]
    ratio = float(size) / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    dh, dw = size - new_h, size - new_w
    top, left = dh // 2, dw // 2
    out = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left, _BORDERS[border])
    return out, Letterbox(ratio=ratio, left=left, top=top, size=size)
