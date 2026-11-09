# Pilot evaluation: fall detection on unseen Le2i scenes (10 videos)

**Question:** on Lecture room and Office videos the legacy model never trained on, do the rule baseline and the legacy ST-GCN (R50, R101) behave differently, and is it worth annotating the remaining 50 videos?

**Short answer:**
- With 5 falls and about 4 minutes of non-fall video, **the pilot cannot statistically separate the detectors.** All 95% intervals overlap widely.
- It does show three things worth acting on:
  - **The legacy ST-GCN's R101 advantage seen on its own training scenes (L1) does not carry over.**
  - **Pose/tracking coverage failures** cause at least one miss.
  - **The labels were accepted too quickly** to count as independent verification (§1).

**Inputs:**
- Labels: `annotations/le2i_pilot/annotations.json`.
- Detectors: the same detectors, event rule and matching as L1 (`scripts/eval_pilot.py`), run on the canonical extractions `15efdb82c39c` (R50, 224×160) and `7bf0c80fee72` (R101, 320×256).
- Artefacts in this folder: `summary.csv`, `per_video.json`, `per_video_outcomes.csv`, `review_provenance.json`, `excluded_uncertain.json`.

## 1. Annotation validation

**Passed:**

| Check | Result |
|---|---|
| Completeness | 10/10 selected videos have an entry |
| Frame count and sha256 vs inventory | 10/10 match |
| Frame count vs both extractions | 10/10 match |
| Schema and validation on reload | events within range, start ≤ end, no overlaps, no event inside a not-visible span |
| Review provenance | 10/10 carry `review` and the originating proposal |
| Uncertain labels | none (0 excluded) |
| Not-visible spans | present on 9/10 videos, carried from the proposals (approximate, ±15 frames) |

**Needs your judgement: the review was very fast.**
- All 10 proposals were **accepted unchanged**: 0 edits, 0 "unsure".
- The ten saves span **59 s**.
- For **8 of the 9** videos where it can be measured, the time since the previous save was shorter than that proposal's own auto-play. For example, `office/29` needs 12.2 s at 4× but was saved 5 s after the previous video (`review_provenance.json`).
- So most decisions were made before the proposal finished playing, presumably from the overview thumbnails.
- I did not change any label. These labels are treated here as **human-accepted proposals after rapid review**:
  - The fall/no-fall calls are plausibly confirmed: the thumbnails show the fall or the absence of one.
  - The boundaries are effectively the proposer's.
- This matters less than it might: **shifting every annotated start and end by ±0.5 s changes 0 of 135 detector outcomes**, so the results below depend on the fall/no-fall calls, not on boundary precision.

## 2. Results

**Denominators:**
- **Falls:** 5 falls in 5 videos (Lecture room 3, Office 2); 5 no-fall videos.
- **Non-fall time:** video time outside each fall's matching window [start − 1 s, end + 3 s].
  - **Strict:** 4.24 min.
  - **Visible-only:** 3.43 min. 0.81 min (19%) was excluded because the person was not visible.
- **Rates:** with these denominators, one false alarm corresponds to 14–18 per hour.

**All 10 videos**

| Detector | Recall | Recall 95% CI | Predictions | False alarms | FA / h, strict (4.24 min) | FA / h, visible-only (3.43 min) | Median latency |
|---|---|---|---|---|---|---|---|
| rule (R50 pose) | 3/5 = 0.60 | [0.15, 0.95] | 3 | 0 | 0 (95% upper bound 52) | 0 | 0.48 s |
| legacy ST-GCN, R50 | **5/5 = 1.00** | [0.48, 1.00] | 6 | 1 | 14.2 (95% CI 0.4–79) | 17.5 | 0.68 s |
| legacy ST-GCN, R101 | 3/5 = 0.60 | [0.15, 0.95] | 4 | 1 | 14.2 | 17.5 | 0.76 s |

- **Strict vs visible-only:** both false alarms fell in visible frames, so excluding unobserved time changes only the denominator, not the counts.
- **Unobserved stretches:** no prediction started in a not-visible span. Those stretches are *not* counted as confirmed no-fall evidence in the visible-only variant.

**By scene** (strict)

| Scene | Falls | Rule | ST-GCN R50 | ST-GCN R101 |
|---|---|---|---|---|
| Lecture room (5 videos, 2.03 min non-fall) | 3 | 2/3, 0 FA | 3/3, 1 FA | 2/3, 0 FA |
| Office (5 videos, 2.20 min non-fall) | 2 | 1/2, 0 FA | 2/2, 0 FA | 1/2, 1 FA |

**By video** (✓ = correct; detection time in seconds)

| Video | Label (proposal confidence) | Rule | ST-GCN R50 | ST-GCN R101 |
|---|---|---|---|---|
| lecture_room/2 | fall 375–405 (medium) | ✓ 15.6 s | ✓ 15.8 s | **missed** |
| lecture_room/10 | fall 341–380 (high) | ✓ 14.1 s | ✓ 14.3 s | ✓ 14.4 s |
| lecture_room/11 | fall 241–283 (medium) | **missed** | ✓ 11.0 s | ✓ 11.2 s |
| lecture_room/23 | no fall (high) | ✓ | ✓ | ✓ |
| lecture_room/26 | no fall (medium) | ✓ | **FA at 14.3 s** | ✓ |
| office/3 | fall 345–375 (high) | ✓ 14.3 s | ✓ 14.4 s | **missed** |
| office/9 | fall 64–100 (high) | **missed** | ✓ 3.2 s | ✓ 3.3 s |
| office/21 | no fall (high) | ✓ | ✓ | ✓ |
| office/29 | no fall (high) | ✓ | ✓ | ✓ |
| office/30 | no fall (high) | ✓ | ✓ | **FA at 16.7 s** |

**Per fall:** ST-GCN R50 detects all 5. The rule and R101 each detect 3, different ones. Only 1 of the 5 falls (`lecture_room/10`) is detected by all three.

## 3. Failure analysis

| Error | Evidence | Classification |
|---|---|---|
| R101 misses `office/3` | Pose kept in only **30%** of frames in the fall window with R101, against **100%** with R50. Tracking runs on pose-derived boxes, so the different pose model broke the track | **pose/tracking failure** |
| R101 misses `lecture_room/2` | Coverage 89%; labels "Lying Down"/"Walking"; highest Fall score 0.22 (R50: 0.82) | **classifier error** on a slow collapse; the proposal was medium-confidence (possible label ambiguity) |
| Rule misses `lecture_room/11` | Largest hip drop 0.9 torso lengths in 1 s, with tilt 29° (thresholds 1.0 and 50°) | **rule threshold / slow fall**; medium-confidence proposal (kneel-then-collapse, possible label ambiguity) |
| Rule misses `office/9` | Largest hip drop only 0.47 torso lengths: a backward fall seen from above, with little 2D drop | **rule geometry limitation** |
| R50 false alarm `lecture_room/26` (f358) | Window labels around it: Stand up / Walking / Sit down / bend while the person bends at the table with a bag | **model error** (bending mistaken for a fall) |
| R101 false alarm `office/30` (f418) | Labels Sitting / Sit down; the person is sitting down at the desk | **model error** (sitting down mistaken for a fall) |
| (no miss) `lecture_room/10` | Pose coverage only 39–46% in the fall window, yet both ST-GCN variants detected it | low pose coverage as a recurring risk |
| (observation) `office/9` | Both ST-GCN variants label much of the window "Standing one foot", a class learned only from team-recorded clips | the source-confounded class shows up on unseen scenes |

**Possible annotation ambiguity** touches 2 of the 6 errors: `lecture_room/2` (R101 miss) and `lecture_room/11` (rule miss), the two medium-confidence slow collapses. No error depends on the exact boundaries (±0.5 s robustness check).

## 4. Comparison with L1 (annotated Coffee room / Home)

| | L1: seen scenes (127 videos, 96 falls, 19.4 min non-fall) | Pilot: unseen scenes (10 videos, 5 falls, 4.24 min non-fall) |
|---|---|---|
| Rule recall | 0.69 (66/96) | 0.60 (3/5) |
| ST-GCN R50 recall / FA | 0.90 (86/96), 3 FA | 1.00 (5/5), 1 FA |
| ST-GCN R101 recall / FA | 0.92 (88/96), 1 FA | 0.60 (3/5), 1 FA |

- In L1 the legacy model had **trained on almost every video**, so its numbers were optimistic. The rule never trained on anything.
- The pilot is the first test on scenes the legacy model never saw.
- The point estimates move as a train/test gap would predict for R101. But with 5 falls, **none of these differences is statistically significant.**

## 5. What the pilot supports, and what it cannot establish

**It supports:**
- **The workflow works mechanically:** proposal → human acceptance → validated labels → evaluation, reproducibly (a rerun is byte-identical; 40 tests pass).
- **R101 is not better on unseen scenes,** and its pose output can break tracking (`office/3`). The L1-based case for switching to R101 is not confirmed.
- **The detectors fail on different falls** (only 1/5 detected by all three), so they are not interchangeable.
- **Pose/tracking coverage is a real failure source,** independent of the classifier.

**It cannot establish:**
- **Any ranking of the detectors.** The recall intervals [0.15, 0.95] and [0.48, 1.00] overlap. The false-alarm rates rest on 0–1 events in 4 minutes (intervals up to about 80 per hour).
- **Independence of the labels.** All labels started as one model's proposals, and the human review accepted all of them faster than the proposals played back. This is the confirmation-bias risk realised: these are human-accepted proposals, not independent annotations.
- **No-fall label reliability.** Blind calibration of the proposal method included only **2 no-fall videos**, and 5 of the 10 pilot labels are no-fall.
- **Generalisation beyond two rooms,** with 3 and 2 falls.

## 6. Recommendation

1. **Expand annotation to the remaining 50 Lecture room / Office videos — yes, but with a stronger review protocol.** Without more falls the detectors can't be compared at all: 30 falls would narrow a recall interval to roughly ±0.15. The protocol needs:
   - **(a)** A **blind subset:** about 10 of the 50 annotated from scratch, with no proposal shown. Comparing them with the proposals measures proposal accuracy directly, including for no-fall videos.
   - **(b)** **Enforced viewing** in the tool: `y` enabled only after the proposal or video has played through; time spent recorded.
   - **(c)** Ideally a **second reviewer** on a sample.
2. **Investigate pose/tracking coverage next, in parallel, cheaply.** It caused the clearest attributable miss. A coverage report per fall across all 190 extractions needs no new labels.
3. **Defer event-logic changes and a simple classifier until after the expansion.** Both false alarms were brief misfires during bending or sitting down, which an event rule might suppress. Tuning that on 2 events would be fitting noise. A feature classifier needs the larger labelled set to be trained and tested on unseen scenes honestly.
4. **Don't switch the default to R101** on current evidence.
