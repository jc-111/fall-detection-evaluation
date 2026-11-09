"""Le2i Fall Detection Dataset (official dataUBFC release, doi:10.25666/DATAUBFC-2024-04-09).

Layout found in the official FallDataset.zip (see docs/DATASETS.md):
  <Scene>/Videos/video (i).avi + <Scene>/Annotation_files/video (i).txt   (Coffee_room_01, Home_01, Home_02)
  Coffee_room_02/Annotations_files/...                                      (note the different spelling)
  Lecture room/video (i).avi, Office/video (i).avi                          (no annotation files)

Annotation file: line 1 = fall start frame, line 2 = fall end frame (both 1-based; "0","0" = no fall),
then one line per frame "frame, code, x1, y1, x2, y2" (1-based frame; box in 320x240 pixels;
all-zero box = no person). The README describes the box as height/width/centre, but the values
are corner coordinates (x2 > x1, y2 > y1 within 320x240). The second field ("code", values 1-8)
is undocumented and not used.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2

SCENES = {"Coffee_room_01": "coffee_room_01", "Coffee_room_02": "coffee_room_02", "Home_01": "home_01",
          "Home_02": "home_02", "Lecture room": "lecture_room", "Office": "office"}


@dataclass
class Annotation:
    fall_start: int | None = None  # 0-based frame index (converted from the file's 1-based numbers)
    fall_end: int | None = None
    has_fall: str = "unknown"  # "yes" | "no" | "unknown"
    boxes: dict = field(default_factory=dict)  # 0-based frame -> (x1, y1, x2, y2), only non-empty boxes
    n_box_lines: int = 0
    issues: list = field(default_factory=list)


def parse_annotation(path):
    lines = [ln.strip() for ln in Path(path).read_text(encoding="latin-1").splitlines() if ln.strip()]
    ann = Annotation()
    body = lines
    if len(lines) >= 2 and re.fullmatch(r"\d+", lines[0]) and re.fullmatch(r"\d+", lines[1]):
        s, e = int(lines[0]), int(lines[1])
        body = lines[2:]
        if s == 0 and e == 0:
            ann.has_fall = "no"
        elif 0 < s <= e:
            ann.has_fall, ann.fall_start, ann.fall_end = "yes", s - 1, e - 1
        else:
            ann.issues.append(f"invalid fall header {s},{e}")
    else:
        ann.issues.append("missing fall start/end header")
    for ln in body:
        p = ln.split(",")
        if len(p) != 6:
            ann.issues.append(f"malformed line: {ln[:30]}")
            continue
        ann.n_box_lines += 1
        frame, box = int(p[0]) - 1, tuple(int(v) for v in p[2:])
        if any(box):
            ann.boxes[frame] = box
    return ann


def video_number(path):
    return int(re.search(r"\((\d+)\)", Path(path).name).group(1))


def find_videos(root):
    """Yield (scene_dir_name, video_path, annotation_path or None) for the official layout."""
    root = Path(root)
    for scene in SCENES:
        sdir = root / scene
        vids = sorted((sdir / "Videos").glob("*.avi")) or sorted(sdir.glob("*.avi"))
        ann_dir = next((d for d in (sdir / "Annotation_files", sdir / "Annotations_files") if d.is_dir()), None)
        for v in sorted(vids, key=video_number):
            a = ann_dir / f"{v.stem}.txt" if ann_dir else None
            yield scene, v, (a if a and a.exists() else None)


def probe_video(path):
    """Decoded frame count (exact) plus container fps / size."""
    cap = cv2.VideoCapture(str(path))
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    header_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return {"fps": fps, "width": w, "height": h, "n_frames": n, "header_frames": header_count}
