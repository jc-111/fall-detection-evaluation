"""L1: fall-event evaluation of the rule baseline and the frozen legacy ST-GCN on official Le2i videos.

    python scripts/eval_l1.py --inventory data/le2i/inventory.csv --split data/le2i/legacy_split.csv \
        --out data/le2i/l1

Detectors (all converted to events with evaluation.events.frames_to_events):
  rule_r50     baselines.rule on the legacy_baseline (ResNet-50) extraction
  stgcn_r50    legacy checkpoint, "Fall Down" is the argmax for any track, legacy_baseline extraction
  stgcn_r101   same checkpoint on the legacy_r101 (ResNet-101) extraction
  rule_r101    rule on the ResNet-101 extraction (reference)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
from fall_detection.baselines.rule import flag_falls  # noqa: E402
from fall_detection.classifier import ActionClassifier  # noqa: E402
from fall_detection.config import REPO_ROOT, load_config  # noqa: E402
from fall_detection.data.extract import load_extraction  # noqa: E402
from fall_detection.data.windows import classify_extraction  # noqa: E402
from fall_detection.evaluation.events import Event, aggregate, evaluate_video, frames_to_events  # noqa: E402

CONFIGS = {"r50": REPO_ROOT / "configs" / "legacy_baseline.yaml", "r101": REPO_ROOT / "configs" / "legacy_r101.yaml"}


def stgcn_flags(arrays, classifier, size):
    out = classify_extraction(arrays, classifier, (size, size))
    fall = classifier.class_names.index("Fall Down")
    return out["frame_index"][out["scores"].argmax(1) == fall] if len(out["frame_index"]) else np.zeros(0, int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--extracted-root", type=Path, default=REPO_ROOT / "data" / "extracted")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    inv = {r["video_id"]: r for r in csv.DictReader(open(args.inventory))}
    status = {r["video_id"]: r["legacy_status"] for r in csv.DictReader(open(args.split))}
    cfgs = {k: load_config(v) for k, v in CONFIGS.items()}
    clf = ActionClassifier(cfgs["r50"].classifier)  # same checkpoint for both extractions

    per_video, unannotated = [], []
    for vid, r in inv.items():
        annotated = r["usable"] == "yes"
        gt = [Event(int(r["fall_start"]), int(r["fall_end"]))] if r["has_fall"] == "yes" else []
        res = {}
        for ext in ("r50", "r101"):
            path = args.extracted_root / cfgs[ext].name / f"{vid.replace('/', '__')}.npz"
            arrays, meta = load_extraction(path)
            fps, n = meta["fps"], meta["n_frames"]
            res[f"rule_{ext}"] = flag_falls(arrays, fps)
            res[f"stgcn_{ext}"] = stgcn_flags(arrays, clf, meta["letterbox"]["size"])
        for det, flagged in res.items():
            pred = frames_to_events(flagged, fps)
            if not annotated:  # no ground truth: keep predictions only, never score them
                unannotated.append({"video_id": vid, "scene": r["scene"], "legacy_status": status.get(vid, "uncertain"),
                                    "detector": det, "fps": fps, "n_frames": n, "duration_s": n / fps,
                                    "pred": [[p.start_frame, p.end_frame] for p in pred]})
                continue
            ev = evaluate_video(gt, pred, fps, n)
            per_video.append({"video_id": vid, "scene": r["scene"], "legacy_status": status.get(vid, "uncertain"),
                              "detector": det, "fps": fps, "n_frames": n,
                              "gt": [[g.start_frame, g.end_frame] for g in gt],
                              "pred": [[p.start_frame, p.end_frame] for p in pred], **ev})
        print(vid, {d: len(v) for d, v in res.items()}, flush=True)

    (args.out / "per_video.json").write_text(json.dumps(per_video, indent=0, default=float))
    (args.out / "unannotated_predictions.json").write_text(json.dumps(unannotated, indent=0, default=float))
    ua = defaultdict(lambda: [0, 0.0, 0])
    for u in unannotated:
        k = (u["detector"], u["scene"])
        ua[k][0] += len(u["pred"])
        ua[k][1] += u["duration_s"]
        ua[k][2] += bool(u["pred"])
    with open(args.out / "unannotated_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["detector", "scene", "videos", "videos_with_prediction", "predicted_events", "hours",
                    "predicted_events_per_hour"])
        for (d, sc), (npred, dur, nv) in sorted(ua.items()):
            nvid = sum(1 for u in unannotated if u["detector"] == d and u["scene"] == sc)
            w.writerow([d, sc, nvid, nv, npred, round(dur / 3600, 4), round(npred / (dur / 3600), 1)])
    groups = defaultdict(list)
    for v in per_video:
        for key in ((v["detector"], "all", "all"), (v["detector"], "all", v["scene"]), (v["detector"], v["legacy_status"], "all"),
                    (v["detector"], v["legacy_status"], v["scene"])):
            groups[key].append(v)
    summary = [{"detector": d, "legacy_status": s, "scene": sc, **aggregate(vs)} for (d, s, sc), vs in sorted(groups.items())]
    with open(args.out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        if s["scene"] == "all":
            print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()})


if __name__ == "__main__":
    main()
