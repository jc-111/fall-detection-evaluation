"""Fall-event review tool (two OpenCV windows). Shows no detector output.

    python scripts/annotate_falls.py            # review the pilot videos (least certain first)
    python scripts/annotate_falls.py --redo     # revisit videos already reviewed
    python scripts/annotate_falls.py --video le2i/office/9
    python scripts/annotate_falls.py --no-proposals   # annotate from scratch
    python scripts/annotate_falls.py --smoke    # open/close the windows briefly (GUI check, saves nothing)

Each video opens with the UNVERIFIED proposal and plays it automatically: a proposed fall from 1 s
before to 1 s after, a proposed no-fall video in full at 4x. The OVERVIEW window shows the whole
video (one thumbnail per second; red = proposed fall, blue = person not visible); click a
thumbnail to jump there. Every decision is saved immediately and moves to the next video.

  y   ACCEPT the proposal           x   UNSURE (cannot decide; excluded from scoring)
  n   NO FALL                       s / e   set fall START / END here, then  w  save that fall
  g   replay the proposal           space   play / pause      + / -   speed      a / d   step a frame
  j / k   jump to proposed start / end
  [ / ]   -1 s / +1 s               v   jump to next not-visible span      f   add another fall (rare)
  u   undo last added fall          p   previous video        q   quit (decided videos are already saved)

Boundaries (same as the official Le2i ground truth):
  START = first frame where the person visibly loses balance and starts an uncontrolled descent
  END   = first frame where the person is on the ground and the fall motion has stopped
  Deliberately sitting, bending, kneeling or lying down is not a fall. In blue (not visible) spans
  nothing can be verified: if a fall could be hidden there, press x instead of accepting "no fall".
"""
import argparse
import csv
import getpass
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # usable without installing the package
from fall_detection.annotation import (load_annotations, load_frames, load_proposals,  # noqa: E402
                                       save_annotations, set_entry)
from fall_detection.config import REPO_ROOT  # noqa: E402

PILOT = REPO_ROOT / "annotations" / "le2i_pilot"
LEFT, RIGHT = {2424832, 63234, 65361}, {2555904, 63235, 65363}
SCALE = 2
SPEEDS = [0.5, 1, 2, 4, 8]
MAIN, OVER = "review", "overview"
TILE_W, TILE_H, OVER_COLS = 80, 60, 10
ORANGE, RED, BLUE, YELLOW, WHITE, GREY = (0, 140, 255), (0, 0, 255), (255, 120, 0), (0, 255, 255), (255, 255, 255), (170, 170, 170)


def in_spans(i, spans):
    return any(s <= i <= e for s, e in spans)


def overview(frames, fps, proposal):
    """Whole-video thumbnail grid. Returns (image, list of frame indices per tile)."""
    n = len(frames)
    step = max(int(round(fps)), math.ceil(n / 70))
    idx = list(range(0, n, step))
    events = [(e["start"], e["end"]) for e in proposal["events"]] if proposal else []
    hidden = proposal.get("not_visible", []) if proposal else []
    tiles = []
    for i in idx:
        t = cv2.resize(frames[i], (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        if in_spans(i, hidden):
            t = cv2.addWeighted(t, 0.45, np.full_like(t, BLUE), 0.55, 0)
        if in_spans(i, events) or any(abs(i - s) < step / 2 for s, _ in events):
            cv2.rectangle(t, (0, 0), (TILE_W - 1, TILE_H - 1), RED, 3)
        cv2.rectangle(t, (0, 0), (TILE_W, 12), (0, 0, 0), -1)
        cv2.putText(t, f"{i / fps:.0f}s f{i}", (2, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.28, WHITE, 1)
        tiles.append(t)
    while len(tiles) % OVER_COLS:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + OVER_COLS]) for r in range(0, len(tiles), OVER_COLS)])
    head = np.zeros((22, grid.shape[1], 3), np.uint8)
    cv2.putText(head, "OVERVIEW - click to jump   red = proposed fall   blue = person not visible", (6, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    return np.vstack([head, grid]), idx


def overview_with_playhead(base, idx, i):
    img = base.copy()
    k = max(j for j, f in enumerate(idx) if f <= i)
    r, c = divmod(k, OVER_COLS)
    cv2.rectangle(img, (c * TILE_W, 22 + r * TILE_H), ((c + 1) * TILE_W - 1, 22 + (r + 1) * TILE_H - 1), YELLOW, 2)
    return img


def render(frame, i, n, fps, vid, pos, total, marks, events, decision, proposal, speed, message=""):
    img = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    w = img.shape[1]
    hidden = proposal.get("not_visible", []) if proposal else []
    if in_spans(i, hidden):
        cv2.rectangle(img, (0, 0), (w - 1, img.shape[0] - 1), BLUE, 6)
        cv2.putText(img, "PERSON NOT VISIBLE (cannot verify)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
    panel = np.zeros((232, w, 3), np.uint8)
    x = lambda f: int(10 + (w - 20) * f / max(n - 1, 1))  # noqa: E731
    cv2.rectangle(panel, (10, 8), (w - 10, 22), (80, 80, 80), -1)
    for s, e in hidden:
        cv2.rectangle(panel, (x(s), 8), (max(x(e), x(s) + 1), 22), BLUE, -1)
    if proposal:
        for ev in proposal["events"]:
            cv2.rectangle(panel, (x(ev["start"]), 24), (max(x(ev["end"]), x(ev["start"]) + 2), 29), ORANGE, -1)
    for s, e in events:
        cv2.rectangle(panel, (x(s), 8), (max(x(e), x(s) + 2), 22), RED, -1)
    for m in (marks.get("start"), marks.get("end")):
        if m is not None:
            cv2.line(panel, (x(m), 4), (x(m), 26), YELLOW, 2)
    cv2.line(panel, (x(i), 2), (x(i), 31), WHITE, 1)
    medium = proposal and proposal["confidence"] != "high"
    prop = ("PROPOSAL (unverified): " + ("FALL " + str([(e["start"], e["end"]) for e in proposal["events"]])
                                         if proposal["has_fall"] == "yes" else "NO FALL") +
            f"   [{proposal['confidence']} confidence]") if proposal else "no proposal"
    note = ("  note: " + proposal.get("note", ""))[:88] if proposal else ""
    lines = [(f"[{pos + 1}/{total}] {vid}    speed {speed}x", WHITE, 0.45),
             ("!! MEDIUM-CONFIDENCE PROPOSAL - check carefully !!" if medium else "", RED, 0.5),
             (f"FRAME {i} / {n - 1}    t = {i / fps:6.2f} s", YELLOW, 0.45),
             (prop, ORANGE, 0.42),
             (note, ORANGE, 0.36),
             (f"your marks: start={marks.get('start')} end={marks.get('end')}   added falls: {events}   "
              f"decision: {decision or 'none yet'}", WHITE, 0.42),
             (message, (0, 220, 0), 0.42),
             ("START = balance lost   END = on ground, motion stopped   sitting/bending = not a fall", GREY, 0.36),
             ("y accept  x unsure  n no fall  s/e + w correct  g replay  space play  +/- speed  q quit", GREY, 0.36)]
    for k, (t, c, sz) in enumerate(lines):
        cv2.putText(panel, t, (10, 50 + 21 * k), cv2.FONT_HERSHEY_SIMPLEX, sz, c, 1 if sz < 0.5 else 2)
    return np.vstack([img, panel])


def place_windows():
    """Side by side and in front: video on the left, overview to its right (fits a 1440-px-wide screen)."""
    cv2.moveWindow(MAIN, 10, 30)
    cv2.moveWindow(OVER, 10 + 640 + 20, 30)
    for name in (MAIN, OVER):
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)


def window_closed():
    try:
        return cv2.getWindowProperty(MAIN, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def annotate_video(vid, info, data, pos, total, annotator, proposal=None, out_path=None):
    """Interactive loop for one video. Returns "next", "prev" or "quit"."""
    out_path = out_path or PILOT / "annotations.json"
    frames, fps = load_frames(REPO_ROOT / info["path"])
    n = len(frames)
    if n != int(info["n_frames"]):
        raise RuntimeError(f"{vid}: decoded {n} frames, inventory says {info['n_frames']}")
    prev = data["videos"].get(vid, {})
    events = [(e["start"], e["end"]) for e in prev.get("events", [])] if prev.get("has_fall") == "yes" else []
    decision = prev.get("has_fall")
    sec = int(round(fps))
    p_start = proposal["events"][0]["start"] if proposal and proposal["events"] else None
    p_end = proposal["events"][0]["end"] if proposal and proposal["events"] else None
    hidden = proposal.get("not_visible", []) if proposal else []
    marks = {"start": p_start, "end": p_end} if p_start is not None and not prev else {}
    # auto-play on open: proposed fall with 1 s context, or a proposed no-fall video in full at 4x
    if p_start is not None:
        i, playing, play_until, speed_i, rest_at = max(p_start - sec, 0), True, min(p_end + sec, n - 1), 1, p_start
    elif proposal:
        i, playing, play_until, speed_i, rest_at = 0, True, n - 1, 3, None
    else:
        i, playing, play_until, speed_i, rest_at = 0, False, None, 1, None
    over_base, over_idx = overview(frames, fps, proposal)
    click = {"frame": None}

    def on_click(event, xx, yy, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and yy >= 22:
            k = ((yy - 22) // TILE_H) * OVER_COLS + xx // TILE_W
            if k < len(over_idx):
                click["frame"] = over_idx[k]

    cv2.namedWindow(MAIN, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(OVER, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(OVER, on_click)
    place_windows()
    message = "playing the proposal..." if proposal else ""

    def save(has_fall, evs, review):
        extra = {"review": review, "not_visible": hidden,
                 "not_visible_source": "proposal (approximate), shown to reviewer" if hidden else "none"}
        if proposal:
            extra["proposal"] = {k: proposal[k] for k in ("has_fall", "events", "confidence")}
        entry = set_entry(data, vid, has_fall, evs, n, fps, info["sha256"], "", annotator, **extra)
        save_annotations(data, out_path)
        print(f"  saved {vid}: {entry['has_fall']} {[(e['start'], e['end']) for e in entry['events']]} review={review}")

    def same_as_proposal(has_fall, evs):
        return bool(proposal) and has_fall == proposal["has_fall"] and \
            evs == [(e["start"], e["end"]) for e in proposal["events"]]

    while True:
        speed = SPEEDS[speed_i]
        cv2.imshow(MAIN, render(frames[i], i, n, fps, vid, pos, total, marks, events, decision, proposal, speed, message))
        cv2.imshow(OVER, overview_with_playhead(over_base, over_idx, i))
        key = cv2.waitKeyEx(max(1, int(1000 / (fps * speed))) if playing else 30)
        if window_closed():
            return "quit"
        if click["frame"] is not None:  # a click on the overview jumps there (a key in the same cycle still counts)
            i, playing, play_until, rest_at, click["frame"] = click["frame"], False, None, None, None
        if key == -1:
            if playing:
                i = min(i + (max(1, int(speed)) if speed >= 1 else 1), n - 1)
                if i >= (play_until if play_until is not None else n - 1):
                    playing, play_until = False, None
                    if rest_at is not None:
                        i, rest_at = rest_at, None
                    message = "decide: y accept / x unsure / n no fall / s,e then w to correct"
            continue
        ch = chr(key & 0xFF) if 0 <= key & 0xFF < 128 and key not in LEFT | RIGHT else ""
        if ch and ch in "+=-":  # speed change keeps playing
            speed_i = min(speed_i + 1, len(SPEEDS) - 1) if ch in "+=" else max(speed_i - 1, 0)
            continue
        was_playing = playing
        playing, play_until, rest_at = False, None, None
        message = ""
        if ch == " ":
            playing = not was_playing
        elif ch == "d" or key in RIGHT:
            i = min(i + 1, n - 1)
        elif ch == "a" or key in LEFT:
            i = max(i - 1, 0)
        elif ch == "]":
            i = min(i + sec, n - 1)
        elif ch == "[":
            i = max(i - sec, 0)
        elif ch == "g" and p_start is not None:
            i, playing, play_until, speed_i = max(p_start - sec, 0), True, min(p_end + sec, n - 1), 1
        elif ch == "g" and proposal:
            i, playing, play_until = 0, True, n - 1
        elif ch == "j" and p_start is not None:
            i = p_start
        elif ch == "k" and p_end is not None:
            i = p_end
        elif ch == "v" and hidden:
            nxt = [s for s, _ in hidden if s > i] or [hidden[0][0]]
            i = nxt[0]
        elif ch == "y" and proposal:
            evs = [(e["start"], e["end"]) for e in proposal["events"]]
            save(proposal["has_fall"], evs, "accepted")
            return "next"
        elif ch == "x":
            save("uncertain", [], "uncertain")
            return "next"
        elif ch == "n":
            save("no", [], "accepted" if same_as_proposal("no", []) else ("edited" if proposal else "manual"))
            return "next"
        elif ch == "s":
            marks["start"] = i
        elif ch == "e":
            marks["end"] = i
        elif ch in ("f", "w"):
            s, e = marks.get("start"), marks.get("end")
            pending = s is not None and e is not None
            if pending and s > e:
                message = "start is after end - fix with s / e"
                continue
            if pending and any(not (e < a or s > b) for a, b in events):
                message = "overlaps a fall already added"
                continue
            if pending:
                events, marks = sorted(events + [(s, e)]), {}
            if ch == "f":
                message = f"fall added: {events}" if pending else "set s and e first"
                continue
            if not events:
                message = "nothing to save: set s and e (or press y / n / x)"
                continue
            save("yes", events, "accepted" if same_as_proposal("yes", events) else ("edited" if proposal else "manual"))
            return "next"
        elif ch == "u" and events:
            events.pop()
        elif ch == "p":
            return "prev"
        elif ch == "q":
            return "quit"


def review_order(videos, proposals):
    """Least certain proposals first, so they get reviewed while attention is fresh."""
    rank = {"low": 0, "medium": 1, "high": 2}
    return sorted(videos, key=lambda v: (rank.get(proposals.get(v, {}).get("confidence"), -1), videos.index(v)))


def smoke(videos, inv, proposals):
    """Open the real windows for every video, play a few frames, close. Saves nothing."""
    for k, v in enumerate(videos):
        frames, fps = load_frames(REPO_ROOT / inv[v]["path"])
        base, idx = overview(frames, fps, proposals.get(v))
        cv2.namedWindow(MAIN, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(OVER, cv2.WINDOW_AUTOSIZE)
        place_windows()
        for i in range(0, len(frames), max(1, len(frames) // 8)):
            cv2.imshow(MAIN, render(frames[i], i, len(frames), fps, v, k, len(videos), {}, [], None, proposals.get(v), 1, "GUI check"))
            cv2.imshow(OVER, overview_with_playhead(base, idx, i))
            cv2.waitKey(15)
        visible = cv2.getWindowProperty(MAIN, cv2.WND_PROP_VISIBLE) >= 1 and cv2.getWindowProperty(OVER, cv2.WND_PROP_VISIBLE) >= 1
        print(f"  {v}: windows visible={visible}  main={render(frames[0], 0, len(frames), fps, v, 0, 1, {}, [], None, proposals.get(v), 1).shape[:2]}  overview={base.shape[:2]}")
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=REPO_ROOT / "data" / "le2i" / "inventory.csv")
    ap.add_argument("--video", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-proposals", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--annotator", default=getpass.getuser())
    args = ap.parse_args()

    inv = {r["video_id"]: r for r in csv.DictReader(open(args.inventory))}
    videos = [args.video] if args.video else json.loads((PILOT / "selection.json").read_text())["videos"]
    proposals = {} if args.no_proposals else load_proposals(PILOT / "proposals.json",
                                                            {v: int(r["n_frames"]) for v, r in inv.items()})
    if args.smoke:
        smoke(review_order(videos, proposals), inv, proposals)
        return
    data = load_annotations(PILOT / "annotations.json")
    todo = videos if (args.redo or args.video) else [v for v in videos if v not in data["videos"]]
    todo = review_order(todo, proposals)
    print(__doc__.split("    python")[0].strip())
    print(__doc__[__doc__.index("  y   ACCEPT"):])
    medium = [v for v in todo if proposals.get(v, {}).get("confidence") in ("medium", "low")]
    print(f"{len(videos) - len(todo)} already reviewed, {len(todo)} to do. Medium-confidence (shown first): {medium}")
    k = 0
    while 0 <= k < len(todo):
        res = annotate_video(todo[k], inv[todo[k]], data, k, len(todo), args.annotator, proposals.get(todo[k]))
        if res == "quit":
            break
        k += 1 if res == "next" else -1 if k > 0 else 0
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    done = [v for v in videos if v in data["videos"]]
    print(f"reviewed {len(done)}/{len(videos)} -> {PILOT / 'annotations.json'}")
    if len(done) == len(videos):
        print("All pilot videos reviewed. Tell Claude: 'review finished'.")


if __name__ == "__main__":
    main()
