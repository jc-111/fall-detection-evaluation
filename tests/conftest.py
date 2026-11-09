import pytest

from fall_detection.config import DEFAULT_CONFIG, REPO_ROOT, load_config

# The inherited model code is not redistributed; fetch it with scripts/fetch_third_party.py.
THIRD_PARTY_READY = (REPO_ROOT / "third_party" / "human_falling_detect_tracks" / "pose_utils.py").exists()
collect_ignore = [] if THIRD_PARTY_READY else [
    "test_checkpoint.py", "test_pipeline_smoke.py", "test_dataset_parity.py"]

CONFIG = load_config(DEFAULT_CONFIG)
MODEL_FILES = [CONFIG.detector.cfg_file, CONFIG.detector.weights, CONFIG.pose.weights,
               CONFIG.classifier.weights]
MODELS_READY = all(p.exists() for p in MODEL_FILES)
LEGACY_VIDEO_001 = REPO_ROOT / "legacy" / "web_app" / "detect" / "video" / "001.avi"


def pytest_collection_modifyitems(config, items):
    if MODELS_READY and THIRD_PARTY_READY:
        return
    skip = pytest.mark.skip(reason="third-party code or model files missing: run scripts/fetch_third_party.py "
                                   "and scripts/setup_models.py")
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
