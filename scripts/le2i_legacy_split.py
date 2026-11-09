"""Mark every official Le2i video as seen / probably_seen / unseen / uncertain w.r.t. legacy training data.

    python scripts/le2i_legacy_split.py --inventory data/le2i/inventory.csv \
        --mapping data/le2i/legacy_mapping_r50.csv data/le2i/legacy_mapping_r101.csv --out data/le2i/legacy_split.csv

With several mappings (e.g. from two extractors) the statuses are combined conservatively: the
strongest evidence of having been seen wins, and "unseen" requires every mapping to agree.

Uses the segment mapping (scripts/map_legacy_segments.py) plus the builder's file order. Within each
legacy pickle, segments appear in the lexicographic order of that folder's videos. "Anchors" are
segments mapped with high confidence.
  seen           a high-confidence segment maps to the video
  probably_seen  a medium-confidence segment maps to it, or it lies between two anchors and the
                 unanchored segments between them are at least as many as the unanchored videos
  unseen         the order rules it out: it lies before the first anchor or between two anchors that
                 are adjacent in segment order (no segment left that could come from it), or its
                 scene's folder was not used by any legacy pickle (Lecture room, Office) and no
                 segment's content points there distinctively (margin >= 1.2 and RMSE <= 0.3)
  uncertain      otherwise
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_legacy_segments import PICKLES, legacy_folder_order  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inv = list(csv.DictReader(open(args.inventory)))
    results = [classify(inv, list(csv.DictReader(open(m)))) for m in args.mapping]
    rank = {"seen": 3, "probably_seen": 2, "uncertain": 1, "unseen": 0}
    out = []
    for r in inv:
        per = [res[r["video_id"]] for res in results]
        best = max(per, key=lambda x: rank[x[0]])
        out.append({"video_id": r["video_id"], "scene": r["scene"], "legacy_status": best[0], "reason": best[1],
                    "per_mapping": "|".join(x[0] for x in per), "segments": best[2]})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summary = defaultdict(lambda: defaultdict(int))
    for r in out:
        summary[r["scene"]][r["legacy_status"]] += 1
    for scene, c in summary.items():
        print(f"{scene:15}", dict(sorted(c.items())))
    print("unseen and annotated:", [r["video_id"] for r in out if r["legacy_status"] == "unseen"
                                    and r["scene"] not in ("lecture_room", "office")])


def classify(inv, mapping):
    """-> {video_id: (status, reason, segments)} for one mapping."""
    status, reason, segs_of = {}, {}, defaultdict(list)
    for m in mapping:
        segs_of[m["video_id"]].append(f"{m['legacy_pickle']}:{m['legacy_segment']}({m['confidence']})")

    for pk in PICKLES:
        order = legacy_folder_order(pk)
        pos = {v: i for i, v in enumerate(order)}
        rows = sorted((m for m in mapping if m["legacy_pickle"] == pk), key=lambda m: int(m["legacy_segment"]))
        for m in rows:
            v = m["video_id"]
            if m["confidence"] == "high":
                status[v], reason[v] = "seen", "high-confidence segment"
            elif m["confidence"] == "medium" and status.get(v) != "seen":
                status[v], reason[v] = "probably_seen", "medium-confidence segment"
        anchors = [(k, pos[m["video_id"]]) for k, m in enumerate(rows) if m["confidence"] == "high"]
        first_seg, first_pos = anchors[0]
        for p in range(first_pos):  # videos before the first anchor
            if order[p] not in status:
                status[order[p]], reason[order[p]] = (
                    ("unseen", "precedes the first segment's video in builder order") if first_seg == 0
                    else ("uncertain", "before the first anchor"))
        for (k1, p1), (k2, p2) in zip(anchors, anchors[1:]):
            gap_videos = [order[p] for p in range(p1 + 1, p2) if order[p] not in status]
            free = sum(1 for k in range(k1 + 1, k2) if rows[k]["confidence"] == "low")
            for v in gap_videos:
                if k2 == k1 + 1:
                    status[v], reason[v] = "unseen", "between adjacent anchored segments"
                elif free >= len(gap_videos):
                    status[v], reason[v] = "probably_seen", f"{free} unresolved segments for {len(gap_videos)} videos"
                else:
                    status[v], reason[v] = "uncertain", f"{free} unresolved segments for {len(gap_videos)} videos"
        for p in range(anchors[-1][1] + 1, len(order)):
            if order[p] not in status:
                status[order[p]], reason[order[p]] = "uncertain", "after the last anchor"

    legacy_scenes = {v.split("/")[1] for pk in PICKLES for v in legacy_folder_order(pk)}
    # a scene outside the legacy folders counts as pointed to only by a distinctive content match
    confident_elsewhere = {m["content_best_video"].split("/")[1] for m in mapping
                           if float(m["content_margin"]) >= 1.2 and float(m["content_best_rmse"]) <= 0.3}
    res = {}
    for r in inv:
        v = r["video_id"]
        if r["scene"] not in legacy_scenes:
            st = "unseen" if r["scene"] not in confident_elsewhere else "uncertain"
            res[v] = (st, "scene folder not used by legacy pickles; no confident content match", "")
        else:
            res[v] = (status.get(v, "uncertain"), reason.get(v, "no evidence"), ";".join(segs_of.get(v, [])))
    return res


if __name__ == "__main__":
    main()
