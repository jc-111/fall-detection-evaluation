"""Adapter for the inherited two-stream ST-GCN action classifier."""
import numpy as np
import torch

from . import _vendor  # noqa: F401
from .features import normalize_sequence, to_model_inputs


class ActionClassifier:
    """Loads a TSSTG checkpoint strictly against the configured class list.

    Scores are independent sigmoid outputs (the model was trained with BCE on soft labels),
    not a softmax distribution, and are not calibrated probabilities.
    """

    def __init__(self, cfg, device="cpu"):
        from Actionsrecognition.Models import TwoStreamSpatialTemporalGraph  # inherited

        self.class_names = list(cfg.class_names)
        self.window = cfg.window
        self.device = device
        self.model = TwoStreamSpatialTemporalGraph({"strategy": "spatial"}, len(self.class_names))
        state = torch.load(cfg.weights, map_location=device)
        self.model.load_state_dict(state)  # strict: fails on class-count or architecture mismatch
        self.model.to(device).eval()

    @torch.no_grad()
    def score_features(self, features, batch_size=256):
        """(N,T,14,3) normalized features -> (N, num_classes) scores."""
        pts, mot = to_model_inputs(features)
        out = [self.model((pts[i:i + batch_size].to(self.device), mot[i:i + batch_size].to(self.device)))
               for i in range(0, len(pts), batch_size)]
        return torch.cat(out).cpu().numpy()

    def score_window(self, keypoints, image_size):
        """(T,13,3) pixel keypoints of one track -> (num_classes,) scores."""
        return self.score_features(normalize_sequence(keypoints, image_size)[None])[0]

    def label(self, scores):
        return self.class_names[int(np.argmax(scores))]
