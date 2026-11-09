"""Pipeline configuration: one YAML file, resolved to absolute paths at load time."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FrameConfig:
    size: int = 384  # frames are letterboxed to size x size before detection and pose
    border: str = "replicate"  # "replicate" (deployed legacy web app) or "constant" (upstream)


@dataclass
class DetectorConfig:
    cfg_file: Path
    weights: Path
    conf_threshold: float = 0.45
    nms_threshold: float = 0.2
    expand_bbox: int = 10


@dataclass
class PoseConfig:
    weights: Path
    backbone: str = "resnet50"
    input_height: int = 224
    input_width: int = 160


@dataclass
class TrackerConfig:
    max_age: int = 1
    n_init: int = 3
    max_iou_distance: float = 0.7


@dataclass
class ClassifierConfig:
    weights: Path
    class_names: list = field(default_factory=list)
    window: int = 30


@dataclass
class PipelineConfig:
    name: str
    device: str
    frame: FrameConfig
    detector: DetectorConfig
    pose: PoseConfig
    tracker: TrackerConfig
    classifier: ClassifierConfig

    def to_dict(self):
        def plain(v):
            if isinstance(v, Path):
                return str(v.relative_to(REPO_ROOT)) if v.is_relative_to(REPO_ROOT) else str(v)
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            return v
        return plain(asdict(self))


def _resolve(models_dir, p):
    p = Path(p)
    return p if p.is_absolute() else (models_dir / p).resolve()


def load_config(path):
    """Load a pipeline config. Relative model paths are resolved against `models_dir`,
    which itself is resolved against the repository root, never the working directory."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    models_dir = Path(raw.get("models_dir", "models"))
    models_dir = models_dir if models_dir.is_absolute() else REPO_ROOT / models_dir
    det, pose, clf = raw["detector"], raw["pose"], raw["classifier"]
    return PipelineConfig(
        name=raw.get("name", path.stem),
        device=raw.get("device", "cpu"),
        frame=FrameConfig(**raw.get("frame", {})),
        detector=DetectorConfig(**{**det, "cfg_file": _resolve(models_dir, det["cfg_file"]),
                                   "weights": _resolve(models_dir, det["weights"])}),
        pose=PoseConfig(**{**pose, "weights": _resolve(models_dir, pose["weights"])}),
        tracker=TrackerConfig(**raw.get("tracker", {})),
        classifier=ClassifierConfig(**{**clf, "weights": _resolve(models_dir, clf["weights"])}),
    )


DEFAULT_CONFIG = REPO_ROOT / "configs" / "legacy_baseline.yaml"
