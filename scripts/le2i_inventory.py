"""Build data/le2i/inventory.csv from the extracted official Le2i release and audit its annotations.

    python scripts/le2i_inventory.py data/raw/le2i --out data/le2i/inventory.csv
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
from fall_detection.config import REPO_ROOT  # noqa: E402
from fall_detection.data.extract import file_sha256  # noqa: E402
from fall_detection.data.le2i import SCENES, find_videos, parse_annotation, probe_video, video_number  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for scene, vpath, apath in find_videos(args.root):
        info = probe_video(vpath)
        issues = []
        if info["header_frames"] != info["n_frames"]:
            issues.append(f"container says {info['header_frames']} frames, decoded {info['n_frames']}")
        ann = parse_annotation(apath) if apath else None
        if ann is None:
            has_fall, start, end, n_box, n_nonempty = "unknown", "", "", 0, 0
            issues.append("no annotation file")
        else:
            issues += ann.issues
            has_fall, start, end = ann.has_fall, ann.fall_start, ann.fall_end
            n_box, n_nonempty = ann.n_box_lines, len(ann.boxes)
            if n_box != info["n_frames"]:
                issues.append(f"{n_box} box lines vs {info['n_frames']} frames")
            if end is not None and end >= info["n_frames"]:
                issues.append(f"fall end {end} beyond last frame {info['n_frames'] - 1}")
            bad = [b for b in ann.boxes.values() if not (b[2] > b[0] and b[3] > b[1])]
            if bad:
                issues.append(f"{len(bad)} boxes with x2<=x1 or y2<=y1")
        usable = "yes" if has_fall in ("yes", "no") and not any("beyond" in i for i in issues) else "no"
        rows.append({
            "video_id": f"le2i/{SCENES[scene]}/{video_number(vpath)}", "scene": SCENES[scene],
            "room": SCENES[scene].rsplit("_0", 1)[0], "path": str(vpath.resolve().relative_to(REPO_ROOT)),
            "sha256": file_sha256(str(vpath)), **{k: info[k] for k in ("fps", "width", "height", "n_frames")},
            "duration_s": round(info["n_frames"] / info["fps"], 2),
            "annotation_path": str(apath.resolve().relative_to(REPO_ROOT)) if apath else "",
            "has_fall": has_fall, "fall_start": "" if start is None else start, "fall_end": "" if end is None else end,
            "box_lines": n_box, "frames_with_box": n_nonempty, "usable": usable, "issues": "; ".join(issues)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by = Counter((r["scene"], r["has_fall"]) for r in rows)
    for scene in SCENES.values():
        sr = [r for r in rows if r["scene"] == scene]
        print(f"{scene:15} videos={len(sr):3} fall={by[(scene, 'yes')]:3} no_fall={by[(scene, 'no')]:3} "
              f"unknown={by[(scene, 'unknown')]:3} frames={sum(r['n_frames'] for r in sr):6} "
              f"minutes={sum(r['duration_s'] for r in sr) / 60:5.1f} fps={sorted({round(r['fps'], 2) for r in sr})}")
    print("total videos", len(rows), "frames", sum(r["n_frames"] for r in rows))
    print("issues:")
    for r in rows:
        if r["issues"]:
            print(f"  {r['video_id']}: {r['issues']}")


if __name__ == "__main__":
    main()
