"""Structured output schema (no model files needed)."""
import csv
import json

from fall_detection.outputs import (SCHEMA_VERSION, FrameResult, PersonResult, VideoResult, csv_columns,
                                    write_csv, write_json)

CLASSES = ["Standing", "Fall Down"]


def make_result():
    kp = [[1.0, 2.0, 0.5]] * 13
    frames = [
        FrameResult(0, 0.0, []),
        FrameResult(1, 0.04, [PersonResult(3, [1, 2, 3, 4], kp)]),  # pending: no label yet
        FrameResult(2, 0.08, [PersonResult(3, [1, 2, 3, 4], kp, "Fall Down", {"Standing": 0.1, "Fall Down": 0.9})]),
    ]
    return VideoResult(video="v.mp4", fps=25.0, width=320, height=240, class_names=CLASSES,
                       config={"name": "test"}, frames=frames)


def test_json_schema(tmp_path):
    write_json(make_result(), tmp_path / "p.json")
    d = json.loads((tmp_path / "p.json").read_text())
    assert d["schema_version"] == SCHEMA_VERSION
    assert {"video", "fps", "width", "height", "class_names", "config", "frame_count", "frames"} <= set(d)
    assert d["frame_count"] == 3 and [f["frame_index"] for f in d["frames"]] == [0, 1, 2]
    person = d["frames"][2]["persons"][0]
    assert set(person) == {"track_id", "bbox", "keypoints", "label", "scores"}
    assert len(person["keypoints"]) == 13 and len(person["keypoints"][0]) == 3
    assert d["frames"][1]["persons"][0]["label"] is None and d["frames"][1]["persons"][0]["scores"] is None


def test_csv_schema(tmp_path):
    write_csv(make_result(), tmp_path / "p.csv")
    rows = list(csv.reader(open(tmp_path / "p.csv")))
    assert rows[0] == csv_columns(CLASSES)
    assert rows[0][:8] == ["frame_index", "timestamp_s", "track_id", "x1", "y1", "x2", "y2", "label"]
    assert len(rows) == 3  # header + one row per (frame, person); empty frame has no row
    assert rows[1][7] == "" and rows[1][8:] == ["", ""]
    assert rows[2][7] == "Fall Down" and float(rows[2][9]) == 0.9
