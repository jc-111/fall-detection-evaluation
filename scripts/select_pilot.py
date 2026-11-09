"""Select the annotation-pilot videos from the unseen, unannotated Le2i scenes.

Selection uses only the inventory and a fixed seed; no model output is read.

    python scripts/select_pilot.py --inventory data/le2i/inventory.csv --out annotations/le2i_pilot/selection.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEED = 20261004
PER_SCENE = {"lecture_room": 5, "office": 5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inventory)))
    rng = np.random.default_rng(SEED)
    selected = []
    for scene, k in PER_SCENE.items():
        ids = sorted((r["video_id"] for r in rows if r["scene"] == scene), key=lambda v: int(v.rsplit("/", 1)[1]))
        selected += sorted(rng.choice(ids, size=k, replace=False).tolist(), key=lambda v: (v.split("/")[1], int(v.rsplit("/", 1)[1])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seed": SEED, "per_scene": PER_SCENE, "method": "numpy default_rng(seed).choice "
                                    "over video IDs sorted by number, per scene in the order listed; no model output used",
                                    "videos": selected}, indent=1))
    print("\n".join(selected))


if __name__ == "__main__":
    main()
