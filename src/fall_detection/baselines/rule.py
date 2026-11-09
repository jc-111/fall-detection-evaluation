"""Deliberately simple, interpretable fall rule on tracked 2D keypoints.

A frame of a track is flagged as "fall" when, compared with the same track about one second
earlier, the hip centre has dropped by at least DROP_TORSO torso lengths AND the torso is now
tilted at least MIN_TILT_DEG from vertical. Torso length is the track's median shoulder-to-hip
distance over the preceding two seconds, so the rule is scale-free. Frames whose shoulder/hip
confidences average below MIN_CONF are skipped.

Thresholds are fixed a priori (not fitted to any test data).
"""
import numpy as np

DROP_TORSO = 1.0
MIN_TILT_DEG = 50.0
LOOKBACK_S = 1.0
SCALE_WINDOW_S = 2.0
MIN_CONF = 0.2

L_SH, R_SH, L_HIP, R_HIP = 1, 2, 7, 8


def torso_geometry(kp):
    """(N,13,3) -> hip-centre y, torso length, tilt from vertical in degrees, mean torso confidence."""
    sh = (kp[:, L_SH, :2] + kp[:, R_SH, :2]) / 2
    hip = (kp[:, L_HIP, :2] + kp[:, R_HIP, :2]) / 2
    v = sh - hip
    length = np.linalg.norm(v, axis=1)
    tilt = np.degrees(np.arctan2(np.abs(v[:, 0]), np.abs(v[:, 1])))
    conf = kp[:, [L_SH, R_SH, L_HIP, R_HIP], 2].mean(axis=1)
    return hip[:, 1], length, tilt, conf


def flag_falls(arrays, fps, drop_torso=DROP_TORSO, min_tilt_deg=MIN_TILT_DEG):
    """Frame indices flagged as fall, using every recorded track observation of an extraction."""
    flagged = []
    for tid in np.unique(arrays["track_id"]):
        idx = np.where(arrays["track_id"] == tid)[0]
        idx = idx[np.argsort(arrays["frame_index"][idx])]
        frames = arrays["frame_index"][idx]
        hip_y, length, tilt, conf = torso_geometry(arrays["keypoints"][idx])
        ok = conf >= MIN_CONF
        for j in range(len(idx)):
            if not ok[j]:
                continue
            past = np.where(ok & (frames <= frames[j] - LOOKBACK_S * fps) & (frames >= frames[j] - 2 * LOOKBACK_S * fps))[0]
            scale = ok & (frames < frames[j]) & (frames >= frames[j] - SCALE_WINDOW_S * fps)
            if len(past) == 0 or scale.sum() < 3:
                continue
            ref = np.median(length[scale])
            drop = hip_y[j] - hip_y[past[-1]]  # image y grows downward
            if ref > 0 and drop >= drop_torso * ref and tilt[j] >= min_tilt_deg:
                flagged.append(int(frames[j]))
    return np.asarray(sorted(set(flagged)), dtype=int)
