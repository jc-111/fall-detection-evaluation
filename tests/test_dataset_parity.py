"""Training/evaluation data must see exactly what the live pipeline sees."""
import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import LEGACY_VIDEO_001
from fall_detection.data.extract import extract_video, load_extraction, save_extraction
from fall_detection.data.windows import classify_extraction

pytestmark = pytest.mark.models


@pytest.mark.skipif(not LEGACY_VIDEO_001.exists(), reason="legacy test video not present")
def test_offline_windows_reproduce_live_pipeline(pipeline, tmp_path):
    arrays, meta = extract_video(pipeline, LEGACY_VIDEO_001, "legacy/001")
    save_extraction(tmp_path / "x.npz", arrays, meta)
    arrays, meta = load_extraction(tmp_path / "x.npz")
    assert meta["extractor"]["frame"]["border"] == "replicate" and meta["n_frames"] == 79
    assert len(meta["extractor"]["pose"]["weights_sha256"]) == 64

    offline = classify_extraction(arrays, pipeline.classifier, (meta["letterbox"]["size"],) * 2)
    live = pipeline.run_video(LEGACY_VIDEO_001)
    live_scores = {(f.frame_index, p.track_id): p.scores for f in live.frames for p in f.persons if p.scores}
    assert len(live_scores) == len(offline["frame_index"]) > 0
    names = pipeline.classifier.class_names
    for f, t, s in zip(offline["frame_index"], offline["track_id"], offline["scores"]):
        np.testing.assert_allclose(s, [live_scores[(int(f), int(t))][c] for c in names], atol=1e-6)

    golden = [r for r in csv.DictReader(open(Path(__file__).parent / "data" / "golden_001_legacy_replay.csv"))
              if r["label"]]
    assert len(golden) == len(offline["frame_index"])
