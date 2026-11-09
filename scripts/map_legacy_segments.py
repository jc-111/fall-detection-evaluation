"""Map the legacy training pickles' Le2i segments back to official videos.

Evidence used:
1. Content: for every stored segment (recovery/lineage/segments_*.csv) and every extracted official
   video, the masked RMSE between the stored per-frame skeleton features and the canonical
   extraction's features (features.normalize_sequence, x/y, 14 nodes), at the best frame offset that
   places the whole segment inside the video.
2. Order: the legacy builder concatenated videos in `sorted(os.listdir())` order of one folder per
   pickle ("Coffee_room": official videos 1-70, "Home": 1-60), i.e. lexicographic file names
   (video (1), video (10), video (11), ...). Segments therefore map monotonically onto that list
   (a video may yield several segments or none).

The final assignment is the order-preserving one with minimum total RMSE (dynamic programming);
confidence compares it with the content-only best match:
  high       DP video == content-only best over all 190 videos, and margin to the runner-up >= 1.2
  medium     DP video == content-only best, smaller margin; or DP video within 10% of the best RMSE
  low        content evidence contradicts the order-constrained assignment

    python scripts/map_legacy_segments.py --extracted data/extracted/legacy_r101 --out data/le2i/legacy_mapping.csv
"""
import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
from fall_detection.config import REPO_ROOT  # noqa: E402
from fall_detection.data.extract import load_extraction  # noqa: E402
from fall_detection.features import normalize_sequence  # noqa: E402

LEGACY = REPO_ROOT / "legacy" / "web_app" / "detect" / "DATA"
PICKLES = {"coffee": LEGACY / "Coffee_room_new-set(labelXscrw).pkl", "home": LEGACY / "Home_new-set(labelXscrw).pkl"}
SEGMENTS = {k: REPO_ROOT / "recovery" / "lineage" / f"segments_{k}.csv" for k in PICKLES}
FOLDER = {"coffee": ("coffee_room_01", "coffee_room_02", 70), "home": ("home_01", "home_02", 60)}
SPLIT_AT = {"coffee": 48, "home": 30}
UNMATCHED = 1.0  # assignment cost of an ineligible (segment, video) pair; real matches are <~0.7
MIN_VALID = 0.8  # share of segment frames that must have a pose in the video at the offset


def legacy_folder_order(pickle_name):
    """Official video IDs in the order sorted(os.listdir()) lists 'video (k).avi' for k = 1..N."""
    a, b, n = FOLDER[pickle_name]
    names = sorted(f"video ({k}).avi" for k in range(1, n + 1))
    ks = [int(re.search(r"\((\d+)\)", x).group(1)) for x in names]
    return [f"le2i/{a if k <= SPLIT_AT[pickle_name] else b}/{k}" for k in ks]


def stored_segments():
    for name, pkl in PICKLES.items():
        with open(pkl, "rb") as fh:
            X, _ = pickle.load(fh)
        for s in csv.DictReader(open(SEGMENTS[name])):
            runs = s["label_runs"]
            if ">" not in runs and (runs.startswith("bend") or runs.startswith("Standing one foot")):
                continue  # team-recorded, not Le2i
            a, b = int(s["start"]), int(s["end"])
            frames = np.concatenate([X[a:b + 1, 0], X[b, 1:]])  # (n_windows + 29, 14, 3)
            yield {"pickle": name, "seg": int(s["seg"]), "start": a, "end": b, "runs": runs,
                   "xy": frames[:, :, :2].reshape(len(frames), -1).astype(np.float64)}


def video_features(path):
    """Per-frame (n_frames, 28) features of the dominant track present in each frame, plus a mask."""
    arrays, meta = load_extraction(path)
    n = meta["n_frames"]
    feats, mask = np.zeros((n, 28)), np.zeros(n, dtype=bool)
    if len(arrays["frame_index"]):
        tids, counts = np.unique(arrays["track_id"], return_counts=True)
        rank = {t: r for r, t in enumerate(tids[np.argsort(-counts)])}
        order = sorted(range(len(arrays["frame_index"])), key=lambda i: rank[arrays["track_id"][i]], reverse=True)
        norm = normalize_sequence(arrays["keypoints"], (meta["letterbox"]["size"],) * 2)[:, :, :2].reshape(-1, 28)
        for i in order:  # best-ranked track written last, so it wins per frame
            feats[arrays["frame_index"][i]] = norm[i]
            mask[arrays["frame_index"][i]] = True
    return meta, feats, mask


def best_offset(seg, feats, mask):
    m, n = len(seg), len(feats)
    if n < m:
        return np.inf, None
    mv = feats * mask[:, None]
    s2 = (seg ** 2).sum(1)
    ones = np.ones(m)
    t1 = fftconvolve((mv ** 2).sum(1), ones, "valid")
    t2 = sum(fftconvolve(mv[:, d], seg[::-1, d], "valid") for d in range(seg.shape[1]))
    t3 = fftconvolve(mask.astype(float), s2[::-1], "valid")
    cnt = np.rint(fftconvolve(mask.astype(float), ones, "valid"))
    sse = np.maximum(t1 - 2 * t2 + t3, 0)
    rmse = np.where(cnt >= MIN_VALID * m, np.sqrt(sse / np.maximum(cnt, 1) / seg.shape[1]), np.inf)
    o = int(np.argmin(rmse))
    return float(rmse[o]), o


def monotone_assignment(cost):
    """cost (S segments, V videos in folder order) -> video index per segment, non-decreasing, min total cost."""
    S, V = cost.shape
    # A segment with no eligible offset in a video (too few frames with a pose) costs UNMATCHED there:
    # worse than any real match, but bounded, so one unmatchable segment cannot drag its neighbours.
    c = np.where(np.isfinite(cost), cost, UNMATCHED)
    dp = np.full((S, V), np.inf)
    back = np.zeros((S, V), dtype=int)
    dp[0] = c[0]
    for s in range(1, S):
        run_min, run_arg = np.inf, 0
        for v in range(V):
            if dp[s - 1, v] < run_min:
                run_min, run_arg = dp[s - 1, v], v
            dp[s, v], back[s, v] = run_min + c[s, v], run_arg
    v = int(np.argmin(dp[-1]))
    out = [v]
    for s in range(S - 1, 0, -1):
        v = back[s, v]
        out.append(v)
    return out[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracted", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    videos = {}
    for p in sorted(args.extracted.glob("*.npz")):
        meta, feats, mask = video_features(p)
        videos[meta["video_id"]] = (meta, feats, mask)
    vid_ids = list(videos)
    segs = list(stored_segments())
    cost = np.full((len(segs), len(vid_ids)), np.inf)
    offset = np.full((len(segs), len(vid_ids)), -1)
    for i, seg in enumerate(segs):
        for j, vid in enumerate(vid_ids):
            cost[i, j], o = best_offset(seg["xy"], videos[vid][1], videos[vid][2])
            offset[i, j] = -1 if o is None else o
    np.savez_compressed(args.out.with_suffix(".cost.npz"), cost=cost, offset=offset, videos=np.array(vid_ids),
                        segments=np.array([f"{s['pickle']}:{s['seg']}" for s in segs]))

    rows = []
    for pk in PICKLES:
        idx = [i for i, s in enumerate(segs) if s["pickle"] == pk]
        order = [v for v in legacy_folder_order(pk) if v in videos]
        cols = [vid_ids.index(v) for v in order]
        assign = monotone_assignment(cost[np.ix_(idx, cols)])
        for i, a in zip(idx, assign):
            seg, j = segs[i], cols[a]
            ranked = np.argsort(cost[i])
            best, second = ranked[0], ranked[1]
            margin = cost[i, second] / cost[i, best] if cost[i, best] > 0 else np.inf
            if j == best and margin >= 1.2:
                conf = "high"
            elif j == best or cost[i, j] <= 1.1 * cost[i, best]:
                conf = "medium"
            else:
                conf = "low"
            o = int(offset[i, j])
            rows.append({"legacy_pickle": pk, "legacy_segment": seg["seg"], "window_start": seg["start"],
                         "window_end": seg["end"], "n_frames": len(seg["xy"]), "label_runs": seg["runs"],
                         "video_id": vid_ids[j], "scene": videos[vid_ids[j]][0].get("scene"),
                         "frame_start": o, "frame_end": o + len(seg["xy"]) - 1, "rmse": round(float(cost[i, j]), 4),
                         "content_best_video": vid_ids[best], "content_best_rmse": round(float(cost[i, best]), 4),
                         "content_margin": round(float(margin), 2), "confidence": conf})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({c: sum(r["confidence"] == c for r in rows) for c in ("high", "medium", "low")}))
    best_scene = {}
    for r in rows:
        best_scene[r["content_best_video"].split("/")[1]] = best_scene.get(r["content_best_video"].split("/")[1], 0) + 1
    print("content-only best matches by scene:", best_scene)


if __name__ == "__main__":
    main()
