"""Fetch the inherited model code from its upstream repository (it is NOT redistributed here).

    python scripts/fetch_third_party.py

Clones GajuuzZ/Human-Falling-Detect-Tracks at the pinned commit, copies the 38 files listed in
third_party/FILES.txt into third_party/human_falling_detect_tracks/, verifies each one against the
upstream git blob hash recorded in third_party/upstream_blobs_GajuuzZ.tsv, then applies the small
CPU-loading patch (third_party/cpu_loading.patch). Needs git and network access to github.com.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"
DEST = TP / "human_falling_detect_tracks"
REPO = "https://github.com/GajuuzZ/Human-Falling-Detect-Tracks.git"
COMMIT = "7ed2faa4d6147dfd576f58869b6c25545208af35"


def blob_hash(path):
    return subprocess.run(["git", "hash-object", "--no-filters", str(path)], check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    if (DEST / "pose_utils.py").exists():
        print(f"already present: {DEST} (delete it to re-fetch)")
        return
    files = (TP / "FILES.txt").read_text().split()
    expected = dict(line.rstrip("\n").split("\t") for line in (TP / "upstream_blobs_GajuuzZ.tsv").open()
                    if line.strip() and not line.startswith("#"))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--quiet", REPO, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "--quiet", COMMIT], check=True)
        bad = []
        for rel in files:
            src, dst = Path(tmp) / rel, DEST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if blob_hash(dst) != expected[rel]:
                bad.append(rel)
    if bad:
        shutil.rmtree(DEST)
        sys.exit(f"upstream files do not match the recorded hashes: {bad}")
    shutil.copy2(TP / "UPSTREAM.md", DEST / "UPSTREAM.md")
    subprocess.run(["git", "apply", str(TP / "cpu_loading.patch")], cwd=ROOT, check=False)
    patched = "map_location" in (DEST / "DetectorLoader.py").read_text()
    if not patched:  # fall back to plain patch(1) when not inside a git work tree
        subprocess.run(["patch", "-p1", "-i", str(TP / "cpu_loading.patch")], cwd=ROOT, check=True)
    print(f"fetched {len(files)} upstream files (hash-verified) and applied the CPU patch -> {DEST}")


if __name__ == "__main__":
    main()
