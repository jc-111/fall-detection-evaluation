"""Render frame-numbered contact sheets of a video for visual fall inspection (no model output shown).

    python scripts/make_review_sheets.py le2i/office/9                    # overview, every 10th frame
    python scripts/make_review_sheets.py le2i/office/9 --range 200 320 --step 2   # dense strip

Frames come from fall_detection.annotation.load_frames, the same loader and numbering as the
annotation tool and the pipeline. Output: data/le2i/review_sheets/<video>_<start>-<end>_s<step>.jpg
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
from fall_detection.annotation import load_frames  # noqa: E402
from fall_detection.config import REPO_ROOT  # noqa: E402


def sheet(frames, fps, start, end, step, cols, scale):
    tiles = []
    for i in range(start, min(end, len(frames) - 1) + 1, step):
        t = cv2.resize(frames[i], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (t.shape[1], 14), (0, 0, 0), -1)
        cv2.putText(t, f"f{i}  {i / fps:.1f}s", (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_id")
    ap.add_argument("--range", nargs=2, type=int, default=None)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--inventory", type=Path, default=REPO_ROOT / "data" / "le2i" / "inventory.csv")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "le2i" / "review_sheets")
    args = ap.parse_args()

    r = {x["video_id"]: x for x in csv.DictReader(open(args.inventory))}[args.video_id]
    frames, fps = load_frames(REPO_ROOT / r["path"])
    s, e = args.range or (0, len(frames) - 1)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.video_id.replace('/', '__')}_{s}-{e}_s{args.step}.jpg"
    cv2.imwrite(str(path), sheet(frames, fps, s, e, args.step, args.cols, args.scale), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(path, f"({len(frames)} frames, {fps:.1f} fps)")


if __name__ == "__main__":
    main()
