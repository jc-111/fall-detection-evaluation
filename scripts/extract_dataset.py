"""Run the canonical extraction stage over every video of an inventory CSV.

    python scripts/extract_dataset.py data/le2i/inventory.csv --config configs/legacy_baseline.yaml

Writes data/extracted/<config name>/<video_id>.npz (keypoints + provenance, see
fall_detection.data.extract). Resumable: videos whose output exists are skipped.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
import torch  # noqa: E402

from fall_detection import FallDetectionPipeline  # noqa: E402
from fall_detection.config import REPO_ROOT, load_config  # noqa: E402
from fall_detection.data.extract import extract_video, save_extraction  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inventory", type=Path)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "extracted")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--only-scene", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    cfg = load_config(args.config)
    out_dir = args.out_root / cfg.name
    rows = [r for r in csv.DictReader(open(args.inventory))
            if not args.only_scene or r["scene"] == args.only_scene]
    todo = [r for r in rows if not (out_dir / f"{r['video_id'].replace('/', '__')}.npz").exists()]
    print(f"{cfg.name}: {len(rows) - len(todo)} done, {len(todo)} to extract -> {out_dir}", flush=True)
    pipeline = FallDetectionPipeline(cfg)
    t0, frames = time.time(), 0
    for i, r in enumerate(todo, 1):
        arrays, meta = extract_video(pipeline, REPO_ROOT / r["path"], r["video_id"])
        meta["scene"] = r["scene"]
        save_extraction(out_dir / f"{r['video_id'].replace('/', '__')}.npz", arrays, meta)
        frames += meta["n_frames"]
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {r['video_id']}: {meta['n_frames']} frames, {meta['elapsed_s']}s "
              f"(running {frames / el:.1f} fps)", flush=True)


if __name__ == "__main__":
    main()
