"""Evaluate fall detectors on the manually annotated pilot videos (unseen Lecture room / Office).

    python scripts/eval_pilot.py [--out reports/pilot]

Same detectors, event rule and matching as L1 (scripts/eval_l1.py, evaluation/events.py), with ground
truth from annotations/le2i_pilot/annotations.json: human-verified entries only (each must carry the
review provenance written by scripts/annotate_falls.py). Refuses to run until every selected video is
verified, so a partial set is never reported as a result.

Videos the reviewer marked "uncertain" are excluded and listed. Metrics are reported twice:
"strict" (every prediction counts) and "visible_only" (predictions starting where the person is not
visible are not judged, and that time leaves the non-fall duration).

Testing modes (never ground truth, written only under git-ignored data/):
  --labels proposals   score against the UNVERIFIED proposals (pipeline smoke test only)
  --trial              allow a partial set of verified labels (workflow trial)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_l1 import CONFIGS, stgcn_flags  # noqa: E402
from fall_detection.annotation import load_annotations, load_proposals  # noqa: E402
from fall_detection.baselines.rule import flag_falls  # noqa: E402
from fall_detection.classifier import ActionClassifier  # noqa: E402
from fall_detection.config import REPO_ROOT, load_config  # noqa: E402
from fall_detection.data.extract import load_extraction  # noqa: E402
from fall_detection.evaluation.events import Event, aggregate, evaluate_video, frames_to_events  # noqa: E402

PILOT = REPO_ROOT / "annotations" / "le2i_pilot"
DETECTORS = [("rule_r50", "r50"), ("stgcn_r50", "r50"), ("stgcn_r101", "r101")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--annotations", type=Path, default=PILOT / "annotations.json")
    ap.add_argument("--labels", choices=["verified", "proposals"], default="verified")
    ap.add_argument("--trial", action="store_true")
    args = ap.parse_args()

    selected = json.loads((PILOT / "selection.json").read_text())["videos"]
    if args.labels == "proposals":
        ann = {v: {**p, "n_frames": None} for v, p in load_proposals(PILOT / "proposals.json").items()}
        out_default = REPO_ROOT / "data" / "le2i" / "pilot_PROVISIONAL_vs_proposals"
    else:
        ann = load_annotations(args.annotations)["videos"]
        unreviewed = [v for v, a in ann.items() if "review" not in a]
        if unreviewed:
            sys.exit(f"entries without human-review provenance: {unreviewed}")
        out_default = REPO_ROOT / ("data/le2i/pilot_trial" if args.trial else "reports/pilot")
    args.out = args.out or out_default
    missing = [v for v in selected if v not in ann]
    if missing and not (args.trial or args.labels == "proposals"):
        sys.exit(f"not all pilot videos are verified yet ({len(missing)} missing): {missing}")
    selected = [v for v in selected if v in ann]
    uncertain = [v for v in selected if ann[v]["has_fall"] == "uncertain"]
    selected = [v for v in selected if v not in uncertain]  # reviewer could not decide: not scored, listed
    if args.labels == "proposals" or args.trial:
        print("*** NOT A RESULT: " + ("scored against UNVERIFIED proposals" if args.labels == "proposals"
                                      else "partial trial") + " ***")

    cfgs = {k: load_config(v) for k, v in CONFIGS.items()}
    clf = ActionClassifier(cfgs["r50"].classifier)
    per_video, per_video_vis, table = [], [], []
    for vid in selected:
        a = ann[vid]
        gt = [Event(e["start"], e["end"]) for e in a["events"]]
        row = {"video_id": vid, "has_fall": a["has_fall"], "gt_events": [(g.start_frame, g.end_frame) for g in gt],
               "review": a.get("review", "proposal"), "not_visible": a.get("not_visible", [])}
        for det, ext in DETECTORS:
            arrays, meta = load_extraction(REPO_ROOT / "data" / "extracted" / cfgs[ext].name / f"{vid.replace('/', '__')}.npz")
            assert a["n_frames"] in (None, meta["n_frames"]), (vid, meta["n_frames"], a["n_frames"])
            flagged = flag_falls(arrays, meta["fps"]) if det.startswith("rule") else stgcn_flags(arrays, clf, meta["letterbox"]["size"])
            pred = frames_to_events(flagged, meta["fps"])
            ev = evaluate_video(gt, pred, meta["fps"], meta["n_frames"])
            vis = evaluate_video(gt, pred, meta["fps"], meta["n_frames"], ignore_spans=a.get("not_visible", []))
            per_video.append({"video_id": vid, "detector": det, "scene": vid.split("/")[1], **ev,
                              "pred": [[p.start_frame, p.end_frame] for p in pred]})
            per_video_vis.append({"video_id": vid, "detector": det, "scene": vid.split("/")[1], **vis})
            missed = ev["n_falls"] - ev["n_detected"]
            row[det] = ("correct" if not missed and not ev["n_false_alarms"] else
                        ", ".join(x for x in [f"missed {missed}" if missed else "", f"{ev['n_false_alarms']} false alarm(s)"
                                              if ev["n_false_alarms"] else ""] if x))
            row[f"{det}_pred_s"] = [round(p.start_frame / meta["fps"], 2) for p in pred]
            row[f"{det}_latency_s"] = [round(x, 2) for x in ev["latencies_s"]]
        table.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.labels == "proposals" or args.trial:
        (args.out / "NOT_GROUND_TRUTH.txt").write_text("Scored against unverified proposals or a partial trial. Not a result.\n")
    (args.out / "per_video.json").write_text(json.dumps(per_video, indent=1))
    with open(args.out / "per_video_outcomes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    summary = []
    for variant, rows in (("strict", per_video), ("visible_only", per_video_vis)):
        for det, _ in DETECTORS:
            for scope in ("all", "lecture_room", "office"):
                vs = [v for v in rows if v["detector"] == det and scope in ("all", v["scene"])]
                summary.append({"variant": variant, "detector": det, "scene": scope, **aggregate(vs)})
    (args.out / "excluded_uncertain.json").write_text(json.dumps(uncertain, indent=1))
    print("excluded as uncertain:", uncertain)
    with open(args.out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items()})
    for r in table:
        print(r["video_id"], r["has_fall"], r["gt_events"], "|", r["rule_r50"], "|", r["stgcn_r50"], "|", r["stgcn_r101"])


if __name__ == "__main__":
    main()
