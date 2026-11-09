"""Adapter for the inherited Tiny-YOLOv3 one-class person detector."""
from . import _vendor  # noqa: F401


class PersonDetector:
    def __init__(self, cfg, input_size, device="cpu"):
        from DetectorLoader import TinyYOLOv3_onecls  # inherited

        self.expand_bbox = cfg.expand_bbox
        self._model = TinyYOLOv3_onecls(input_size, config_file=str(cfg.cfg_file),
                                        weight_file=str(cfg.weights), nms=cfg.nms_threshold,
                                        conf_thres=cfg.conf_threshold, device=device)

    def detect(self, frame):
        """Letterboxed RGB frame -> tensor (N, 7) [x1, y1, x2, y2, obj_score, cls_score, cls], or None."""
        return self._model.detect(frame, need_resize=False, expand_bb=self.expand_bbox)
