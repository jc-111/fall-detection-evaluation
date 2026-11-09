from pathlib import Path

import pytest

# The inherited upstream code is not redistributed, and the package imports it (e.g. pose_utils),
# so nothing can be tested until it has been fetched.
if not (Path(__file__).resolve().parents[1] / "third_party" / "human_falling_detect_tracks" / "pose_utils.py").exists():
    pytest.exit("inherited upstream code missing: run  python scripts/fetch_third_party.py  first", returncode=4)

from fall_detection.config import DEFAULT_CONFIG, REPO_ROOT, load_config  # noqa: E402

CONFIG = load_config(DEFAULT_CONFIG)
MODEL_FILES = [CONFIG.detector.cfg_file, CONFIG.detector.weights, CONFIG.pose.weights,
               CONFIG.classifier.weights]
MODELS_READY = all(p.exists() for p in MODEL_FILES)
LEGACY_VIDEO_001 = REPO_ROOT / "legacy" / "web_app" / "detect" / "video" / "001.avi"


def pytest_collection_modifyitems(config, items):
    if MODELS_READY:
        return
    skip = pytest.mark.skip(reason="model files missing: run python scripts/setup_models.py")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def config():
    return CONFIG


@pytest.fixture(scope="session")
def pipeline(config):
    from fall_detection import FallDetectionPipeline
    return FallDetectionPipeline(config)
