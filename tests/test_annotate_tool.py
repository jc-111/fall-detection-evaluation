"""Headless tests of the review tool's real interactive loop (OpenCV window calls are stubbed)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import annotate_falls as tool  # noqa: E402
from fall_detection.annotation import load_annotations, load_proposals  # noqa: E402

N, FPS = 60, 25
FALL = {"has_fall": "yes", "events": [{"start": 30, "end": 40}], "confidence": "medium", "note": "",
        "not_visible": [[0, 5]]}
NOFALL = {"has_fall": "no", "events": [], "confidence": "high", "note": "", "not_visible": [[50, 59]]}


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 48))
    for k in range(N):
        w.write(np.full((48, 64, 3), 4 * k, np.uint8))  # grey level reveals the frame index
    w.release()
    return {"path": str(p), "n_frames": str(N), "sha256": "x"}


def run(monkeypatch, tmp_path, video, keys, proposal, visible=1.0, clicks=None):
    shown, cb = [], {}
    seq = iter([ord(k) if isinstance(k, str) else k for k in keys])

    def wait(delay):
        if clicks and len(shown) in clicks:  # simulate a click on the overview before this key
            cb["f"](cv2.EVENT_LBUTTONDOWN, *clicks[len(shown)], 0, None)
        return next(seq)

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "moveWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *a, **k: visible)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, f: cb.update(f=f))
    monkeypatch.setattr(cv2, "imshow", lambda name, img: shown.append(img) if name == tool.MAIN else None)
    monkeypatch.setattr(cv2, "waitKeyEx", wait)
    out = tmp_path / "ann.json"
    res = tool.annotate_video("t/1", video, load_annotations(out), 0, 1, "tester", proposal, out_path=out)
    return res, load_annotations(out)["videos"].get("t/1"), shown


def frame_shown(img):
    return int(round(img[60: 48 * tool.SCALE - 10, 10: 64 * tool.SCALE - 10].mean() / 4))


def test_accept_during_autoplay(monkeypatch, tmp_path, video):
    res, entry, shown = run(monkeypatch, tmp_path, video, ["y"], FALL)
    assert res == "next" and entry["review"] == "accepted" and entry["events"] == [{"start": 30, "end": 40}]
    assert entry["proposal"]["confidence"] == "medium" and entry["not_visible"] == [[0, 5]]
    assert frame_shown(shown[0]) == 30 - FPS  # playback starts 1 s before the proposed start


def test_autoplay_then_rests_at_proposed_start(monkeypatch, tmp_path, video):
    ticks = [-1] * 60  # plays 5 -> min(40 + 25, 59) = 59, then rests at the proposed start (30)
    res, entry, shown = run(monkeypatch, tmp_path, video, ticks + ["y"], FALL)
    assert frame_shown(shown[-1]) == 30 and entry["review"] == "accepted"


def test_edit_boundaries(monkeypatch, tmp_path, video):
    keys = ["j", "d", "d", "s", "k", "a", "e", "w"]  # start 32, end 39
    res, entry, shown = run(monkeypatch, tmp_path, video, keys, FALL)
    assert entry["events"] == [{"start": 32, "end": 39}] and entry["review"] == "edited"
    assert frame_shown(shown[3]) == 32  # frame on screen when 's' was pressed


def test_reject_and_uncertain(monkeypatch, tmp_path, video):
    _, entry, _ = run(monkeypatch, tmp_path, video, ["n"], FALL)
    assert entry["has_fall"] == "no" and entry["review"] == "edited"
    _, entry, _ = run(monkeypatch, tmp_path, video, ["x"], FALL)
    assert entry["has_fall"] == "uncertain" and entry["review"] == "uncertain" and entry["events"] == []


def test_no_fall_proposal_plays_whole_video_fast(monkeypatch, tmp_path, video):
    res, entry, shown = run(monkeypatch, tmp_path, video, [-1, -1, "y"], NOFALL)
    assert [frame_shown(s) for s in shown] == [0, 4, 8]  # 4x playback from the first frame
    assert entry["has_fall"] == "no" and entry["review"] == "accepted" and entry["not_visible"] == [[50, 59]]


def test_space_pauses_autoplay(monkeypatch, tmp_path, video):
    _, _, shown = run(monkeypatch, tmp_path, video, [" ", -1, -1, "q"], FALL)
    assert len({frame_shown(s) for s in shown[1:]}) == 1


def test_overview_click_jumps(monkeypatch, tmp_path, video):
    # second tile of the overview = frame `step`; click it, then mark start there and save a fall
    step = max(FPS, -(-N // 70))
    keys = ["s", "]", "e", "w"]
    _, entry, _ = run(monkeypatch, tmp_path, video, keys, FALL, clicks={1: (tool.TILE_W + 5, 22 + 5)})
    assert entry["events"][0]["start"] == step


def test_nothing_saved_without_a_decision(monkeypatch, tmp_path, video):
    res, entry, _ = run(monkeypatch, tmp_path, video, ["d", "w", "q"], None)
    assert res == "quit" and entry is None
    res, entry, _ = run(monkeypatch, tmp_path, video, ["d"], FALL, visible=0.0)  # window closed
    assert res == "quit" and entry is None


def test_manual_annotation_without_proposal(monkeypatch, tmp_path, video):
    _, entry, _ = run(monkeypatch, tmp_path, video, ["]", "s", "]", "e", "w"], None)
    assert entry["events"] == [{"start": 25, "end": 50}] and entry["review"] == "manual"


def test_review_order_least_certain_first():
    props = {"a": {"confidence": "high"}, "b": {"confidence": "medium"}, "c": {"confidence": "low"}}
    assert tool.review_order(["a", "b", "c", "d"], props) == ["d", "c", "b", "a"]


def test_tool_never_reads_detector_output():
    src = (ROOT / "scripts" / "annotate_falls.py").read_text()
    for forbidden in ("extracted", "eval_", "PROVISIONAL", "classifier", "flag_falls", "per_video"):
        assert forbidden not in src


def test_pilot_proposals_file_is_valid():
    props = load_proposals(ROOT / "annotations" / "le2i_pilot" / "proposals.json")
    assert len(props) == 10 and all("not_visible" in p for p in props.values())
    assert sorted(v for v, p in props.items() if p["confidence"] != "high") == [
        "le2i/lecture_room/11", "le2i/lecture_room/2", "le2i/lecture_room/26"]
