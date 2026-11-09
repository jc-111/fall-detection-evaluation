# Improvement backlog and project direction

> Note for this public copy: paths under `legacy/`, `recovery/` and `LiCode/` refer to the private project archive and are not part of this repository.

Ordering follows dependency: nothing in **ML** or **Product** can be judged before the **Data** and evaluation items it depends on.

**Columns:**
- **Need:** **R** = required for a credible system; **O** = optional.
- **Evidence before starting:** what must be known first.

## Data

| # | Item | Problem it solves | Need | Evidence before starting |
|---|---|---|---|---|
| D1 | Obtain Le2i Home + Coffee-room videos and annotations; confirm the segment → video map with the existing matcher | No raw data, so no honest split, no event labels and no re-extraction | R | Licence or terms acceptable for portfolio use |
| D2 | Rebuild the dataset from raw video through `src/fall_detection` (same detector/pose/features as inference), with per-frame labels and provenance per sample | Train/inference pose mismatch; unknown boxes; silent frame skipping | R | D1 |
| D3 | Recover per-frame posture labels from the stored windows (stride-1 window means can be inverted) or relabel | Label provenance and possible frame-shift bug | R | D1 map confirmed; compare against a hand-checked sample |
| D4 | Split definitions by video and by room (one room held out); never by window | 97.8% legacy figure was leakage | R | D1 |
| D5 | Fall-event annotations (start/end, on-floor interval) plus explicit non-fall videos | No product-level ground truth | R | D1 (Le2i ships fall frames [unverified offline]) |
| D6 | Dataset gate: confidence-only and source-only classifiers must not predict source within a class | Prevents the bend/one-foot confound recurring | R (cheap) | D2 |
| D7 | New recordings for bend / one-foot under the same cameras, rooms and people as other classes | The only route to keep these classes honestly | O (until decided) | A product decision that the classes matter; a recording protocol |

## ML

| # | Item | Problem it solves | Need | Evidence before starting |
|---|---|---|---|---|
| M1 | Evaluation harness: per-class window metrics, plus event recall, false alarms per hour and detection latency, all per split | Nothing can be compared today | R | D4, D5 |
| M2 | Simple baselines on the same keypoints: rule-based fall (hip-velocity + torso angle + on-floor time); logistic/GBM on window features | Shows whether ST-GCN earns its complexity | R | M1 |
| M3 | Retrain the inherited ST-GCN on D2 features with the D4 splits | Honest version of the existing model | R | D2, D4, M1 |
| M4 | Confidence-channel ablation (keep / drop / per-window rescale), as a standing experiment | Measured shortcut (`recovery/CONFIDENCE_REPORT.md`) | R | M3 |
| M5 | Pose backbone comparison (AlphaPose R50 / R101 vs a modern single-stage pose model) under the fixed harness | Legacy stack is slow (pose ≈ 60% of runtime), CUDA-oriented, partly unmaintained | O | M1–M3 show pose quality or latency is the bottleneck |
| M6 | Alternative temporal models (small TCN/GRU) | Possibly simpler or better | O | M2/M3 gap analysis |
| M7 | Score calibration | Thresholds for alerts need meaningful scores | O → R once P1 exists | M3 |

## Product logic

| # | Item | Problem it solves | Need | Evidence before starting |
|---|---|---|---|---|
| P1 | Fall-event state machine (fall-like transition followed by on-floor for N seconds, with cooldown) on top of per-frame scores | Today the output is labels, not events | R | M1 event metrics to tune against |
| P2 | Temporal smoothing / hysteresis of labels | Flicker between classes (e.g. 002: bend ↔ one-foot) | R | M1 |
| P3 | False-alarm handling: confusers such as lying on a sofa, sitting fast, bending | Product credibility | R | D5 non-fall videos; M1 false alarms per hour |
| P4 | Alert interface (webhook or email), stub first | The original product promise | O | P1 validated |
| P5 | Multi-person tracking (`max_age` > 1, re-identification) | Legacy setting is effectively single-person | O | Target scenario defined |

## Presentation

| # | Item | Problem it solves | Need | Evidence before starting |
|---|---|---|---|---|
| U1 | Uploaded-video demo showing annotated video, label timeline and events (any lightweight front end over `run_video`) | A way to demonstrate the system | O (strongly useful) | P1, so the demo shows events rather than raw labels |
| U2 | Event timeline / summary view | Readable results | O | P1 |
| U3 | Browser live camera | Real-time demo | O | U1; latency budget from M5 |
| U4 | History dashboard | Legacy feature | O | A product need |

## Engineering

| # | Item | Problem it solves | Need | Evidence before starting |
|---|---|---|---|---|
| E1 | CI running the no-model tests on every push; model tests in a job with cached weights | Keeps refactors safe | R (cheap) | Where weights will be hosted (E4) |
| E2 | Execute `docs/MIGRATION.md` (legacy trees → `legacy/`) | Cluttered root | R | Decision on `LiCode/` |
| E3 | Licence and attribution decision (GPL-3.0 tracker/detector; upstream declares no licence) | Blocks publishing | R before publishing | `recovery/LICODE_PROVENANCE_REPORT.md` |
| E4 | Weight distribution (release asset / model hub) instead of legacy symlinks | Others can't run the baseline | R before sharing | E3 |
| E5 | Performance: batch pose crops, MPS/GPU option, profiling | 11–14 fps CPU | O | A target latency |
| E6 | Replace deprecated APIs in vendored code (`scipy.misc`, `torch.autograd.Variable`) | Future dependency upgrades | O | A dependency upgrade that actually breaks |

## What could make this genuinely our own project

The detector, pose estimator, tracker and ST-GCN are inherited, and should be credited as such. The original value lies around them.

| Rank | Contribution | Technical value | Portfolio value | Effort |
|---|---|---|---|---|
| 1 | **Leakage-free, event-level evaluation of skeleton fall detection** (D4, D5, M1): video/room splits; event recall, false alarms per hour and latency; per-room reporting | High | High: shows ML methodology and turns the legacy 97.8% into a cautionary before/after story | Medium |
| 2 | **A train/inference-consistent data pipeline with provenance and a shortcut audit** (D2, D3, D6, M4): one extraction path, per-sample metadata, an automated confidence/source leakage gate | High | High: rare in student projects; builds directly on our forensic findings | Medium |
| 3 | **Fall-event layer with evaluated false-alarm behaviour** (P1–P3, M7) | High | High: product thinking, measurable | Medium |
| 4 | **Baselines vs ST-GCN study** (M2, M3, M6): does the graph network beat a rule or a GBM on the same keypoints? | Medium–high | High: an honest, quantitative result, whichever way it goes | Low–medium |
| 5 | **Pose-backbone study** (M5) under the fixed harness, including the measured ResNet-50/101 confidence effect | Medium | Medium | Medium |
| 6 | **Demo** (U1/U2) showing events on an uploaded video | Low | High for first impressions | Low |
| 7 | **Custom-class data collection protocol** (D7) | Medium | Medium | High (recording people) |

**Recommended direction:** position the project as **"a reproducible, leakage-aware fall-event detection system built on an inherited pose/action stack"**. Concretely, items 1 → 2 → 4 → 3, then a small demo (6).

This uses what is already unusually strong here, the forensic evidence of leakage, pose-model mismatch and the confidence shortcut. It turns that into methodology anyone can verify. Model swaps (M5, M6) are then decided by measured results rather than taste.

The single blocking prerequisite is **D1**, obtaining the Le2i videos and annotations.
