"""Fall-event ground truth: file format, validation and frame-exact video loading.

Frame indices are 0-based positions in sequential decode order (`cv2.VideoCapture.read()` from the
start), which is exactly how the pipeline and the extraction number frames. The annotation tool
shows frames from `load_frames`, so a displayed frame number is the index the models use.

File (JSON):
  {"schema": "fall-events/v1", "frame_index": "...", "videos": {
     "<video_id>": {"has_fall": "yes"|"no", "events": [{"start": int, "end": int}], "note": str,
                    "n_frames": int, "fps": float, "video_sha256": str, "annotator": str, "updated": iso}}}
"""
import datetime as dt
import json
from pathlib import Path

import cv2

SCHEMA = "fall-events/v1"
FRAME_INDEX = "0-based, sequential cv2 decode order (same numbering as fall_detection extraction)"


def load_frames(path):
    """All frames of a video in decode order, plus fps. Sequential reading avoids seek inaccuracy."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def validate_entry(entry, n_frames):
    """Raise ValueError if an annotation entry is inconsistent.

    has_fall "uncertain" means the reviewer could not decide; such videos are excluded from scoring.
    `not_visible` lists [start, end] spans where the person cannot be seen (no fall verifiable there).
    """
    if entry.get("has_fall") not in ("yes", "no", "uncertain"):
        raise ValueError("has_fall must be 'yes', 'no' or 'uncertain'")
    events = entry.get("events", [])
    if entry["has_fall"] == "no" and events:
        raise ValueError("a no-fall video cannot have events")
    if entry["has_fall"] == "yes" and not events:
        raise ValueError("has_fall=yes needs at least one event")
    for span in entry.get("not_visible", []):
        if not (len(span) == 2 and 0 <= span[0] <= span[1] < n_frames):
            raise ValueError(f"not_visible span {span} outside 0..{n_frames - 1}")
    prev_end = -1
    for ev in sorted(events, key=lambda e: e["start"]):
        s, e = ev["start"], ev["end"]
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s <= e < n_frames):
            raise ValueError(f"event {ev} outside 0..{n_frames - 1} or start > end")
        if s <= prev_end:
            raise ValueError(f"event {ev} overlaps the previous event")
        prev_end = e


def load_annotations(path):
    path = Path(path)
    if not path.exists():
        return {"schema": SCHEMA, "frame_index": FRAME_INDEX, "videos": {}}
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"unexpected schema {data.get('schema')}")
    for vid, entry in data["videos"].items():
        validate_entry(entry, entry["n_frames"])
    return data


def set_entry(data, video_id, has_fall, events, n_frames, fps, video_sha256, note="", annotator="", **extra):
    """Store a human annotation. `extra` holds provenance such as review="accepted"|"edited"|"manual"."""
    entry = {"has_fall": has_fall, "events": sorted(({"start": int(s), "end": int(e)} for s, e in events),
                                                    key=lambda ev: ev["start"]),
             "note": note, "n_frames": int(n_frames), "fps": float(fps), "video_sha256": video_sha256,
             "annotator": annotator, "updated": dt.datetime.now().isoformat(timespec="seconds"), **extra}
    validate_entry(entry, n_frames)
    data["videos"][video_id] = entry
    return entry


PROPOSAL_SCHEMA = "fall-event-proposals/v1"


def load_proposals(path, n_frames_by_video=None):
    """Unverified label proposals (never ground truth). Validated for shape and range only."""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get("schema") != PROPOSAL_SCHEMA:
        raise ValueError(f"unexpected proposal schema {data.get('schema')}")
    for vid, p in data["videos"].items():
        if p.get("confidence") not in ("high", "medium", "low"):
            raise ValueError(f"{vid}: confidence must be high/medium/low")
        n = (n_frames_by_video or {}).get(vid, 10 ** 9)
        validate_entry(p, n)
    return data["videos"]


def save_annotations(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)  # atomic: a crash never leaves a half-written file
