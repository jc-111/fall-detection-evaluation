# Unseen-scene pilot: label sourcing, proposals and review workflow

**Status: 0 of 10 pilot videos have verified ground truth.** All 10 have **unverified proposals**. No detector result on these videos has been reported, and none may be until a human has verified the labels.

## 1. Do trustworthy annotations already exist? No.

| Source checked | Finding |
|---|---|
| Official dataUBFC release (doi:10.25666/DATAUBFC-2024-04-09), downloaded and inventoried | `Lecture room/` and `Office/` contain videos only, with no `Annotation_files` (`docs/DATASETS.md`) |
| [AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow #10](https://github.com/AdrianNunez/Fall-Detection-with-CNNs-and-Optical-Flow/issues/10) | The maintainer says he could not find labels for these scenes either and used the rest of the dataset |
| Projects such as [Jeyaditya/…Fall_detector_FA2](https://github.com/Jeyaditya/1000406_Jeyaditya_AIY2_Machine_learning_Fall_detector_FA2) | Treat Lecture room/Office as unannotated, and assume they are fall-free. That is contradicted by our inspection: falls occur in at least 5 of the 10 pilot videos |
| [Kaggle mirror](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) | Describes the same official format; no evidence of extra annotations for these scenes |
| [seanphan/le2i-sentinel-frames](https://huggingface.co/datasets/seanphan/le2i-sentinel-frames) | Coffee room and Home only |

No source provides per-video fall frames for these scenes whose identity could be verified against our files, so none is usable as independent ground truth.

## 2. Proposals by independent visual inspection

**Method:**
- An LLM vision model (Claude) inspected frame-numbered contact sheets rendered by `scripts/make_review_sheets.py`.
- The sheets use the same frame loader and numbering as the annotation tool and the pipeline.
- Each video got an overview (every 15 frames, about 0.6 s), then dense strips (every 2–3 frames) around candidate falls and around bending or fast-motion moments.

**Independence:** no rule-baseline or ST-GCN output for these videos was viewed before the proposals were written. They were committed (`5f8d574`) before any detector was run against them.

**Calibration, done blind before proposing** (`calibration.json`):
- **Selection:** 9 officially annotated Coffee/Home videos chosen with a seed, none previously viewed.
- **Protocol:** readings written to disk, with their hashes recorded, before the official labels were opened.

| Check | Result |
|---|---|
| Fall / no-fall agreement | **9 / 9** (one marked "uncertain" and still correct) |
| Fall start error | −4 … +7 frames (≤ 0.28 s) |
| Fall end error | −29 … +4 frames |

The end-frame errors are a **definition mismatch**: I marked first ground contact, while official Coffee-room ends are at "motion stopped". The proposals therefore use the official convention. **That corrected convention was not re-tested blind.**

**Proposals** (`annotations/le2i_pilot/proposals.json`):

| Video | Proposal | Confidence | Note |
|---|---|---|---|
| lecture_room/2 | fall 375–405 | medium | slow slumping collapse; partly controlled-looking |
| lecture_room/10 | fall 341–380 | high | trips over the mat |
| lecture_room/11 | fall 241–283 | medium | lurches, kneels, collapses; onset of the uncontrolled phase ambiguous |
| lecture_room/23 | no fall | high | bends to pick up a paper |
| lecture_room/26 | no fall | medium | handles a bag; out of view f945–1200 |
| office/3 | fall 345–375 | high | stands from chair, falls onto the mat |
| office/9 | fall 64–100 | high | falls backwards onto the mat |
| office/21 | no fall | high | fast exit with motion blur, upright |
| office/29 | no fall | high | stands at the table throughout |
| office/30 | no fall | high | sits and works, leaves |

## 3. Three kinds of label, kept separate

| Kind | Where | May be used for |
|---|---|---|
| **Verified ground truth** | `annotations/le2i_pilot/annotations.json`. Written only by the tool after a human accepts or edits a video; every entry carries `review` = accepted / edited / manual and a copy of the proposal it started from | the pilot evaluation (`scripts/eval_pilot.py`, which refuses entries without review provenance) |
| **Automated proposals** | `proposals.json` (status: UNVERIFIED) | speeding up review; pipeline smoke tests (`eval_pilot.py --labels proposals`, written only to git-ignored `data/` with a NOT_GROUND_TRUTH marker) |
| **Uncertain** | proposals with confidence medium/low (3 videos), plus anything the reviewer notes as ambiguous | shown first in review; flagged in the final report |

## 4. Review workflow, and what was tested

**Workflow update (2026-11-09):**
- **Two windows:** the video panel, plus a clickable whole-video overview with one thumbnail per second. Red marks the proposed fall; blue marks spans where the person is not visible.
- **Auto-play on open:** a proposed fall plays with 1 s of context either side; a proposed no-fall video plays in full at 4×.
- **One key per decision, saved immediately:** `y` accept, `x` unsure (excluded from scoring), `n` no fall, `s`/`e` then `w` to correct.
- **Medium-confidence proposals:** shown first under a red banner.
- **Detector output:** never loaded. A test checks the tool's source for this.
- **Not-visible spans:** proposed spans are shown and saved with each decision. The evaluation reports both a strict variant and a visible-only variant, where predictions starting in not-visible spans are not judged and that time leaves the non-fall duration. A "no fall" decision covers visible frames only.
- **Live GUI check on this Mac (OpenCV Cocoa backend):** `annotate_falls.py --smoke` opened both windows for all 10 videos, and all reported visible. No labels were saved.

**One command:**

```bash
.venv/bin/python scripts/annotate_falls.py
```

- It opens the least certain videos first.
- Each video starts 1 s before the proposed fall, with the proposal drawn in orange.
- `g` replays it, `y` accepts and saves, and `s`/`e`/`f` or `n` correct it.
- `+` speeds up playback for skimming the no-fall videos.

**Evidence that it works:**
- **Tests, 35 passing:**
  - The real key-handling loop is driven headlessly: accepting a proposal, editing its boundaries, rejecting a proposed fall, accepting no-fall, quitting without a decision, and manual annotation without a proposal.
  - The frame on screen when a key is pressed is the index that gets saved, checked with frames that encode their own index.
- **End-to-end trial on a real pilot video (`office/9`), simulated in a scratch file:** tool, then `check_annotations.py` (frame count and sha256 match the inventory and both extractions), then the trial evaluation. The tool frame and the checker sheet were inspected visually.
- **Full-set smoke test** of `eval_pilot.py` against the proposals ran for all 10 videos. Results are not reported.

## 5. What remains uncertain

**Limitations that must appear in the pilot evaluation report:**
- **Calibration of no-fall calls is thin.** The blind calibration had 7 fall videos but only 2 no-fall videos (both called correctly). Agreement on no-fall videos is therefore barely measured, while half the pilot proposals are no-fall.
- **Confirmation bias.** The reviewer sees pre-filled proposals, so accepting may be biased toward them. Each saved entry records `review` = accepted / edited / uncertain, and the evaluation report will give the acceptance rate. The labels are human-verified proposals, not independent annotations.

- **The proposals come from a single vision model.** Its calibration covers 9 videos, with fall boundaries tested blind only under the old end convention.
- **Boundaries for the three medium-confidence videos** depend on where "uncontrolled descent" begins in slow, partly controlled-looking collapses.
- **No-fall calls cover visible frames only.** `lecture_room/26`, `office/29` and `office/30` have stretches where the person is out of view.
- **Reviewer anchoring:** a human checking a proposal may be biased toward accepting it. The tool records accepted vs edited so this can be audited, and the medium-confidence cases come first.
