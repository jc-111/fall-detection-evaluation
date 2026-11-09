"""Structured pipeline results and their JSON / CSV serialisation."""
import csv
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

SCHEMA_VERSION = 1


@dataclass
class PersonResult:
    track_id: int
    bbox: list  # [x1, y1, x2, y2] in original-frame pixels
    keypoints: list  # 13 x [x, y, confidence] in original-frame pixels (features.POSE_JOINTS order)
    label: Optional[str] = None  # None until the track has a full classifier window
    scores: Optional[dict] = None  # class name -> sigmoid score


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    persons: list = field(default_factory=list)


@dataclass
class VideoResult:
    video: str
    fps: float
    width: int
    height: int
    class_names: list
    config: dict
    frames: list = field(default_factory=list)

    @property
    def frame_count(self):
        return len(self.frames)

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "frame_count": self.frame_count, **asdict(self)}


def csv_columns(class_names):
    return (["frame_index", "timestamp_s", "track_id", "x1", "y1", "x2", "y2", "label"]
            + [f"score_{c}" for c in class_names])


def write_json(result, path):
    with open(path, "w") as fh:
        json.dump(result.to_dict(), fh)


def write_csv(result, path):
    """One row per (frame, visible tracked person). Frames without people have no rows."""
    cols = csv_columns(result.class_names)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for fr in result.frames:
            for p in fr.persons:
                scores = [round(p.scores[c], 5) for c in result.class_names] if p.scores else [""] * len(result.class_names)
                w.writerow([fr.frame_index, fr.timestamp_s, p.track_id, *p.bbox, p.label or ""] + scores)
