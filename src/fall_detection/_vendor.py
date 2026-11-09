"""The only place that makes inherited third-party code importable.

The vendored upstream code (third_party/human_falling_detect_tracks, see its UPSTREAM.md) uses
top-level imports such as `from Detection.Models import Darknet`. Rather than rewriting those
files, we put that directory on sys.path here. Project modules import third-party code only
after importing this module.
"""
import sys

from .config import REPO_ROOT

VENDOR_DIR = REPO_ROOT / "third_party" / "human_falling_detect_tracks"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
