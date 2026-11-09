"""Fall-event construction, matching and metrics (whole videos are the unit of evaluation).

Event conversion rule (identical for every detector): frames flagged as "fall" are grouped into
events; a new event starts at a flagged frame more than `merge_gap_s` after the previous flagged
frame. The event's detection time is its first flagged frame.

Matching rule: a predicted event matches a ground-truth fall if its start lies in
[gt_start - before_s, gt_end + after_s]. Each ground-truth fall matches at most one prediction
(the earliest); every other prediction is a false alarm, including duplicates during a fall.
"""
from dataclasses import dataclass

import numpy as np

MERGE_GAP_S = 2.0
MATCH_BEFORE_S = 1.0
MATCH_AFTER_S = 3.0


@dataclass(frozen=True)
class Event:
    start_frame: int
    end_frame: int


def frames_to_events(flagged_frames, fps, merge_gap_s=MERGE_GAP_S):
    frames = np.unique(np.asarray(flagged_frames, dtype=int))
    events, gap = [], merge_gap_s * fps
    for f in frames:
        if events and f - events[-1][1] <= gap:
            events[-1][1] = f
        else:
            events.append([f, f])
    return [Event(int(s), int(e)) for s, e in events]


def match_events(gt, pred, fps, before_s=MATCH_BEFORE_S, after_s=MATCH_AFTER_S):
    """Returns (list of (gt_index, pred_index), list of unmatched pred indices)."""
    used, matches = set(), []
    for gi, g in enumerate(gt):
        lo, hi = g.start_frame - before_s * fps, g.end_frame + after_s * fps
        for pi, p in enumerate(pred):
            if pi not in used and lo <= p.start_frame <= hi:
                used.add(pi)
                matches.append((gi, pi))
                break
    return matches, [i for i in range(len(pred)) if i not in used]


def evaluate_video(gt, pred, fps, n_frames, before_s=MATCH_BEFORE_S, after_s=MATCH_AFTER_S, ignore_spans=()):
    """Per-video counts. Non-fall time = video duration minus the ground-truth matching windows.

    `ignore_spans` ([start, end] frames, e.g. person not visible): unmatched predictions starting
    there are neither false alarms nor counted predictions, and that time leaves the non-fall duration.
    """
    matches, false_alarms = match_events(gt, pred, fps, before_s, after_s)
    covered = np.zeros(n_frames, dtype=bool)
    for g in gt:
        lo, hi = int(max(0, g.start_frame - before_s * fps)), int(min(n_frames, g.end_frame + after_s * fps + 1))
        covered[lo:hi] = True
    ignored = np.zeros(n_frames, dtype=bool)
    for s, e in ignore_spans:
        ignored[max(0, s):min(n_frames, e + 1)] = True
    dropped = [i for i in false_alarms if ignored[min(pred[i].start_frame, n_frames - 1)]]
    return {
        "n_falls": len(gt), "n_pred": len(pred) - len(dropped), "n_detected": len(matches),
        "n_false_alarms": len(false_alarms) - len(dropped), "n_ignored_pred": len(dropped),
        "latencies_s": [(pred[pi].start_frame - gt[gi].start_frame) / fps for gi, pi in matches],
        "duration_s": n_frames / fps, "non_fall_duration_s": float((~covered & ~ignored).sum()) / fps,
    }


def aggregate(per_video):
    """Sum per-video results into event recall/precision, false alarms per hour and latency stats."""
    n_falls = sum(v["n_falls"] for v in per_video)
    n_pred = sum(v["n_pred"] for v in per_video)
    det = sum(v["n_detected"] for v in per_video)
    fa = sum(v["n_false_alarms"] for v in per_video)
    nonfall_h = sum(v["non_fall_duration_s"] for v in per_video) / 3600
    lat = [x for v in per_video for x in v["latencies_s"]]
    return {
        "videos": len(per_video), "falls": n_falls, "detected": det, "predicted_events": n_pred,
        "false_alarms": fa,
        "event_recall": det / n_falls if n_falls else None,
        "event_precision": det / n_pred if n_pred else None,
        "false_alarms_per_hour": fa / nonfall_h if nonfall_h else None,
        "non_fall_hours": nonfall_h,
        "latency_median_s": float(np.median(lat)) if lat else None,
        "latency_p90_s": float(np.percentile(lat, 90)) if lat else None,
    }
