"""Feature / skeleton contracts and deterministic preprocessing (no model files needed)."""
import os

import numpy as np
import pytest

from fall_detection.config import DEFAULT_CONFIG, load_config
from fall_detection.features import GRAPH_NODES, POSE_JOINTS, normalize_sequence, to_model_inputs
from fall_detection.frames import letterbox


def random_keypoints(t=30, seed=0):
    rng = np.random.default_rng(seed)
    kp = rng.uniform(20, 300, size=(t, 13, 3)).astype(np.float32)
    kp[..., 2] = rng.uniform(0, 1, size=(t, 13))
    return kp


def test_shape_contract():
    feats = normalize_sequence(random_keypoints(), (384, 384))
    assert feats.shape == (30, len(GRAPH_NODES), 3) == (30, 14, 3)
    pts, mot = to_model_inputs(feats)
    assert tuple(pts.shape) == (1, 3, 30, 14)
    assert tuple(mot.shape) == (1, 2, 29, 14)
    batch_pts, batch_mot = to_model_inputs(np.stack([feats, feats]))
    assert tuple(batch_pts.shape) == (2, 3, 30, 14) and tuple(batch_mot.shape) == (2, 2, 29, 14)


def test_rejects_wrong_joint_count():
    with pytest.raises(ValueError):
        normalize_sequence(np.zeros((30, 17, 3)), (384, 384))


def test_preprocessing_fixed_input():
    """Hand-computed expectation: per-frame min-max to [-1, 1]; confidence untouched; neck = mean of shoulders."""
    kp = np.zeros((1, 13, 3), dtype=np.float32)
    kp[0, :, 0] = np.arange(13) * 10 + 5  # x: 5..125
    kp[0, :, 1] = 100 - np.arange(13) * 5  # y: 100..40
    kp[0, :, 2] = np.linspace(0.1, 0.9, 13)
    out = normalize_sequence(kp, (200, 100))
    np.testing.assert_allclose(out[0, :13, 0], np.linspace(-1, 1, 13), atol=1e-6)
    np.testing.assert_allclose(out[0, :13, 1], np.linspace(1, -1, 13), atol=1e-6)
    np.testing.assert_allclose(out[0, :13, 2], kp[0, :, 2], atol=1e-7)
    np.testing.assert_allclose(out[0, 13], (out[0, 1] + out[0, 2]) / 2, atol=1e-7)
    assert len(POSE_JOINTS) == 13


def test_preprocessing_is_deterministic_and_input_safe():
    kp = random_keypoints()
    before = kp.copy()
    a = normalize_sequence(kp, (384, 384))
    b = normalize_sequence(kp, (384, 384))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(kp, before)  # caller's array is not modified


def test_normalisation_invariant_to_letterbox_scale_and_offset():
    """Per-frame scaling makes features independent of frame size / padding (recovery/DIAG_004_REPORT.md)."""
    kp = random_keypoints()
    moved = kp.copy()
    moved[..., 0] = kp[..., 0] * 1.2 + 7
    moved[..., 1] = kp[..., 1] * 1.2 + 48
    np.testing.assert_allclose(normalize_sequence(kp, (320, 240)), normalize_sequence(moved, (384, 384)), atol=1e-5)


def test_letterbox_geometry_round_trip():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    out, lb = letterbox(img, 384)
    assert out.shape == (384, 384, 3)
    assert (lb.ratio, lb.left, lb.top) == (1.2, 0, 48)
    np.testing.assert_allclose(lb.to_original([[0, 48], [384, 336]]), [[0, 0], [320, 240]])


def test_config_paths_do_not_depend_on_cwd(tmp_path):
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        cfg = load_config(DEFAULT_CONFIG)
    finally:
        os.chdir(cwd)
    for p in (cfg.detector.cfg_file, cfg.detector.weights, cfg.pose.weights, cfg.classifier.weights):
        assert p.is_absolute() and str(tmp_path) not in str(p)
