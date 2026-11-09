"""Event construction, matching and metrics on hand-made examples (no model files needed)."""
import numpy as np
import pytest

from fall_detection.baselines.rule import flag_falls
from fall_detection.evaluation.events import Event, aggregate, evaluate_video, frames_to_events, match_events

FPS = 25.0


def test_frames_to_events_merges_within_gap():
    ev = frames_to_events([10, 11, 12, 50, 61, 200], FPS, merge_gap_s=2.0)  # gap 2 s = 50 frames
    assert ev == [Event(10, 61), Event(200, 200)]


def test_matching_window_and_duplicates():
    gt = [Event(100, 140)]
    pred = [Event(70, 75), Event(80, 90), Event(150, 160), Event(400, 410)]
    matches, fa = match_events(gt, pred, FPS)  # window = [75, 215]
    assert matches == [(0, 1)] and fa == [0, 2, 3]  # duplicate inside the window is a false alarm


def test_evaluate_and_aggregate():
    v1 = evaluate_video([Event(100, 140)], [Event(110, 120)], FPS, n_frames=500)
    v2 = evaluate_video([], [Event(10, 20), Event(300, 301)], FPS, n_frames=900)
    assert v1["latencies_s"] == [pytest.approx(10 / FPS)]
    assert v1["non_fall_duration_s"] == pytest.approx((500 - (215 - 75 + 1)) / FPS)
    agg = aggregate([v1, v2])
    assert (agg["falls"], agg["detected"], agg["false_alarms"]) == (1, 1, 2)
    assert agg["event_recall"] == 1.0 and agg["event_precision"] == pytest.approx(1 / 3)
    assert agg["false_alarms_per_hour"] == pytest.approx(2 / ((v1["non_fall_duration_s"] + 36) / 3600))


def _track(frames, hip_y, tilt_deg, torso=40.0):
    kp = np.ones((len(frames), 13, 3), dtype=np.float32)
    for i, (y, a) in enumerate(zip(hip_y, tilt_deg)):
        dx, dy = torso * np.sin(np.radians(a)), torso * np.cos(np.radians(a))
        kp[i, 7, :2] = kp[i, 8, :2] = (100, y)
        kp[i, 1, :2] = kp[i, 2, :2] = (100 + dx, y - dy)
    return {"frame_index": np.asarray(frames), "track_id": np.zeros(len(frames), int), "keypoints": kp}


def test_rule_flags_fast_drop_with_tilt_only():
    f = np.arange(100)
    fall = _track(f, np.where(f < 60, 200, 260), np.where(f < 60, 0, 80))
    sit = _track(f, np.where(f < 60, 200, 260), np.zeros(100))  # same drop, torso stays upright
    lie_slowly = _track(f, 200 + f * 0.3, np.linspace(0, 85, 100))  # tilts, but drops < 1 torso / s
    flagged = flag_falls(fall, FPS)
    assert len(flagged) and flagged.min() == 60
    assert len(flag_falls(sit, FPS)) == 0 and len(flag_falls(lie_slowly, FPS)) == 0


def test_ignore_spans_exclude_unjudgeable_predictions():
    gt, pred = [Event(100, 140)], [Event(110, 120), Event(300, 301), Event(450, 460)]
    strict = evaluate_video(gt, pred, FPS, n_frames=500)
    vis = evaluate_video(gt, pred, FPS, n_frames=500, ignore_spans=[(400, 499)])
    assert strict["n_false_alarms"] == 2 and vis["n_false_alarms"] == 1 and vis["n_ignored_pred"] == 1
    assert vis["n_detected"] == 1 and vis["n_pred"] == 2
    assert vis["non_fall_duration_s"] == pytest.approx(strict["non_fall_duration_s"] - 100 / FPS)
