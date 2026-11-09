"""Run the fall-detection pipeline on a video file.

    python scripts/run_video.py path/to/video.mp4 [--config configs/legacy_baseline.yaml]

Writes <out>/annotated.mp4, <out>/predictions.json and <out>/predictions.csv
(default <out> = outputs/<video name>/).
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
from fall_detection import DEFAULT_CONFIG, FallDetectionPipeline
from fall_detection.config import REPO_ROOT
from fall_detection.outputs import write_csv, write_json
from fall_detection.render import draw_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=None, help="output directory")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    args = ap.parse_args()

    out_dir = args.out or REPO_ROOT / "outputs" / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = FallDetectionPipeline(args.config)

    writer = None

    def on_frame(frame, frame_result):
        nonlocal writer
        if args.no_video:
            return
        if writer is None:
            h, w = frame.shape[:2]
            fps = cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FPS) or 25.0
            writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(draw_frame(frame, frame_result))

    t0 = time.time()
    result = pipeline.run_video(args.video, max_frames=args.max_frames, on_frame=on_frame)
    elapsed = time.time() - t0
    if writer is not None:
        writer.release()
    write_json(result, out_dir / "predictions.json")
    write_csv(result, out_dir / "predictions.csv")

    labels = Counter(p.label or "pending" for fr in result.frames for p in fr.persons)
    print(f"{result.frame_count} frames in {elapsed:.1f}s ({result.frame_count / max(elapsed, 1e-9):.1f} fps)")
    print("person-frames by label:", dict(labels.most_common()))
    print("outputs:", out_dir)


if __name__ == "__main__":
    main()
