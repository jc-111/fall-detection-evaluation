"""Populate models/ from models/MANIFEST.tsv, verifying every file's sha256.

Weights are NOT in this repository. Download them yourself from the `source` column of
models/MANIFEST.tsv (the upstream author's Google Drive links), then point this script at the
folder you saved them in; files are matched by checksum, so their names do not matter:

    python scripts/setup_models.py --search ~/Downloads [--copy] [--include-optional]

The 9-class legacy ST-GCN used for the reported results is private and is not distributed; without
it use configs/upstream_7class.yaml. --source-root DIR instead links each file from its recorded
legacy location under DIR (only useful with the private project archive).
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_by_hash(root, wanted_sizes):
    found = {}
    for p in Path(root).expanduser().rglob("*"):
        if p.is_file() and not p.is_symlink() and p.stat().st_size in wanted_sizes:
            found.setdefault(sha256(p), p)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", type=Path, help="folder to search for the weight files (matched by sha256)")
    ap.add_argument("--source-root", type=Path, default=REPO_ROOT,
                    help="directory that legacy_source paths are relative to (private archive layout)")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    ap.add_argument("--include-optional", action="store_true")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(MODELS / "MANIFEST.tsv"), delimiter="\t")
            if r["required"] == "yes" or args.include_optional]
    by_hash = index_by_hash(args.search, {int(r["bytes"]) for r in rows}) if args.search else {}
    failed = False
    for row in rows:
        dst = MODELS / row["path"]
        if dst.exists() and sha256(dst) == row["sha256"]:
            print(f"ok       {row['path']}")
            continue
        src = by_hash.get(row["sha256"]) if args.search else args.source_root / row["legacy_source"]
        if src is None or not src.exists():
            print(f"MISSING  {row['path']}  (source: {row['source']})")
            failed = True
            continue
        if sha256(src) != row["sha256"]:
            print(f"BAD HASH {src}")
            failed = True
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if args.copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(os.path.relpath(src, dst.parent))
        print(f"{'copied' if args.copy else 'linked':8} {row['path']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
