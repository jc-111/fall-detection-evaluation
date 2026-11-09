"""CPU end-to-end tests: synthetic video smoke test and a regression against the legacy replay."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import LEGACY_VIDEO_001
from fall_detection.outputs import write_json

pytestmark = pytest.mark.models


def test_cpu_smoke_on_synthetic_video(pipeline, tmp_path):
    path = tmp_path / "noise.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(12):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()

    seen = []
    result = pipeline.run_video(path, on_frame=lambda frame, fr: seen.append(fr.frame_index))
    assert result.frame_count == 12 == len(seen)
    assert [f.timestamp_s for f in result.frames] == sorted(f.timestamp_s for f in result.frames)
    assert (result.width, result.height) == (320, 240)
    write_json(result, tmp_path / "out.json")
    assert json.loads((tmp_path / "out.json").read_text())["frame_count"] == 12


@pytest.mark.skipif(not LEGACY_VIDEO_001.exists(), reason="legacy test video not present")
def test_matches_legacy_replay_on_001(pipeline):
    """Golden file = recovery/run_video.py (legacy legacy/web_app/detect code + runtime shims) on 001.avi."""
    golden = list(csv.DictReader(open(Path(__file__).parent / "data" / "golden_001_legacy_replay.csv")))
    result = pipeline.run_video(LEGACY_VIDEO_001)
    got = {(f.frame_index, p.track_id): p for f in result.frames for p in f.persons}
    assert set(got) == {(int(r["frame_index"]), int(r["track_id"])) for r in golden}
    for r in golden:
        p = got[(int(r["frame_index"]), int(r["track_id"]))]
        assert (p.label or "") == r["label"]
        if p.label:
            for c in result.class_names:
                assert abs(p.scores[c] - float(r[f"score_{c}"])) < 2e-4
