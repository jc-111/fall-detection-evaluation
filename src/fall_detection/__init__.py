"""Fall / posture detection pipeline built around inherited models (see third_party/)."""
from .config import DEFAULT_CONFIG, load_config
from .pipeline import FallDetectionPipeline

__all__ = ["FallDetectionPipeline", "load_config", "DEFAULT_CONFIG"]
