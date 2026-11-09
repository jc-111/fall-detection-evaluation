"""Classifier windows rebuilt from an extraction table.

A track's recorded rows are exactly the keypoints upstream appends to its 30-frame buffer, so the
window for (track, frame) is that track's last 30 rows up to the frame. The live pipeline classifies
a visible track when it is confirmed and its buffer is full; `iter_windows` applies the same rule.
"""
import numpy as np

from ..features import normalize_sequence


def iter_windows(arrays, window=30):
    """Yield (row_index, frame_index, track_id, keypoints (window, 13, 3)) in the live pipeline's order."""
    by_track = {}
    for i in np.argsort(arrays["frame_index"], kind="stable"):
        tid = int(arrays["track_id"][i])
        hist = by_track.setdefault(tid, [])
        hist.append(i)
        if arrays["confirmed"][i] and len(hist) >= window:
            idx = hist[-window:]
            yield i, int(arrays["frame_index"][i]), tid, arrays["keypoints"][idx]


def classify_extraction(arrays, classifier, image_size, batch_size=512):
    """Scores for every classifiable (frame, track). Returns dict of arrays: frame_index, track_id, scores."""
    frames, tracks, feats = [], [], []
    for _, f, tid, kp in iter_windows(arrays, classifier.window):
        frames.append(f)
        tracks.append(tid)
        feats.append(normalize_sequence(kp, image_size))
    scores = (classifier.score_features(np.stack(feats), batch_size) if feats
              else np.zeros((0, len(classifier.class_names)), dtype=np.float32))
    return {"frame_index": np.asarray(frames, dtype=np.int32), "track_id": np.asarray(tracks, dtype=np.int32),
            "scores": scores}
