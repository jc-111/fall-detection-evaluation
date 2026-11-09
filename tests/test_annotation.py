"""Annotation format and frame-index correctness (no model files needed)."""
import cv2
import numpy as np
import pytest

from fall_detection.annotation import load_annotations, load_frames, save_annotations, set_entry, validate_entry


def write_indexed_video(path, n=40):
    """Frame k is a flat image of grey level 6*k, so the decoded frame reveals its own index."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for k in range(n):
        w.write(np.full((48, 64, 3), 6 * k, np.uint8))
    w.release()


def test_displayed_frame_index_equals_decode_index(tmp_path):
    p = tmp_path / "idx.avi"
    write_indexed_video(p)
    frames, fps = load_frames(p)
    assert len(frames) == 40 and fps == pytest.approx(25)
    assert [int(round(f.mean() / 6)) for f in frames] == list(range(40))
    # same numbering as the extraction/pipeline: plain sequential cap.read()
    cap, k = cv2.VideoCapture(str(p)), 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        assert np.array_equal(f, frames[k])
        k += 1
    assert k == 40


def test_annotation_round_trip(tmp_path):
    data = load_annotations(tmp_path / "a.json")
    set_entry(data, "v/1", "yes", [(30, 45), (5, 10)], 100, 25.0, "abc", note="two falls")
    set_entry(data, "v/2", "no", [], 50, 25.0, "def")
    save_annotations(data, tmp_path / "a.json")
    back = load_annotations(tmp_path / "a.json")
    assert back["videos"]["v/1"]["events"] == [{"start": 5, "end": 10}, {"start": 30, "end": 45}]
    assert back["videos"]["v/2"]["has_fall"] == "no" and back["videos"]["v/1"]["note"] == "two falls"


@pytest.mark.parametrize("entry", [
    {"has_fall": "maybe", "events": []},
    {"has_fall": "yes", "events": []},
    {"has_fall": "no", "events": [{"start": 1, "end": 2}]},
    {"has_fall": "yes", "events": [{"start": 5, "end": 4}]},
    {"has_fall": "yes", "events": [{"start": 1, "end": 100}]},
    {"has_fall": "yes", "events": [{"start": 1, "end": 10}, {"start": 10, "end": 20}]},
])
def test_invalid_entries_rejected(entry):
    with pytest.raises(ValueError):
        validate_entry(entry, 100)
