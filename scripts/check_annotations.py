"""Reload and verify fall-event annotations, and render contact sheets of every marked interval.

    python scripts/check_annotations.py [--annotations annotations/le2i_pilot/annotations.json]

Checks: schema/validation on reload; frame count and sha256 against the inventory; frame count
against both extractions (the indices evaluation uses). Sheets (data/le2i/pilot_check/) show
start-1s, start, middle, end, end+1s with the frame number burned in, taken from the same
frame loader the annotation tool uses.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
from fall_detection.annotation import load_annotations, load_frames  # noqa: E402
from fall_detection.config import REPO_ROOT  # noqa: E402
from fall_detection.data.extract import load_extraction  # noqa: E402

EXTRACTIONS = ["legacy_baseline", "legacy_r101"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=REPO_ROOT / "annotations" / "le2i_pilot" / "annotations.json")
    ap.add_argument("--inventory", type=Path, default=REPO_ROOT / "data" / "le2i" / "inventory.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "le2i" / "pilot_check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load_annotations(args.annotations)  # validates every entry
    inv = {r["video_id"]: r for r in csv.DictReader(open(args.inventory))}
    problems, reviews = 0, {}
    for vid, a in data["videos"].items():
        r = inv[vid]
        checks = {"n_frames vs inventory": a["n_frames"] == int(r["n_frames"]),
                  "sha256 vs inventory": a["video_sha256"] == r["sha256"]}
        for ext in EXTRACTIONS:
            p = REPO_ROOT / "data" / "extracted" / ext / f"{vid.replace('/', '__')}.npz"
            if p.exists():
                checks[f"n_frames vs {ext}"] = load_extraction(p)[1]["n_frames"] == a["n_frames"]
        bad = [k for k, ok in checks.items() if not ok]
        problems += bool(bad)
        print(f"{vid}: has_fall={a['has_fall']} events={[(e['start'], e['end']) for e in a['events']]} "
              f"review={a.get('review')} not_visible={a.get('not_visible', [])} "
              f"checks={'OK' if not bad else 'FAILED ' + ', '.join(bad)}")
        reviews[a.get("review", "none")] = reviews.get(a.get("review", "none"), 0) + 1
        if not a["events"]:
            continue
        frames, fps = load_frames(REPO_ROOT / r["path"])
        sec = int(round(fps))
        for j, e in enumerate(a["events"]):
            idx = [max(e["start"] - sec, 0), e["start"], (e["start"] + e["end"]) // 2, e["end"], min(e["end"] + sec, len(frames) - 1)]
            tiles = [cv2.putText(frames[i].copy(), f"{tag} f{i}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                     for i, tag in zip(idx, ["-1s", "START", "mid", "END", "+1s"])]
            cv2.imwrite(str(args.out / f"{vid.replace('/', '__')}_ev{j}.jpg"), cv2.hconcat(tiles))
    print(f"{len(data['videos'])} videos, {problems} with failed checks; review outcomes {reviews}; sheets in {args.out}")


if __name__ == "__main__":
    main()
