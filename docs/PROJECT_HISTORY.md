# Project history: what was inherited and what was found

This repository started from an inherited student project: a fall-detection web app built on GajuuzZ/Human-Falling-Detect-Tracks, with a retrained 9-class ST-GCN. The original code, a web UI and the training data are kept in a **private archive** and are not published. Documents here that mention `legacy/…` or `recovery/…` paths refer to that archive.

## Forensic findings about the inherited system

1. **The reported 97.8% accuracy was measured on training data.** The legacy evaluation reused the training windows, so it is not evidence of generalisation.

2. **Train/inference pose-model mismatch.** The stored Le2i training features match the AlphaPose FastPose **ResNet-101** (320×256) extractor. The deployed app ran **ResNet-50** (224×160). Both are evaluated here (`configs/legacy_baseline.yaml` = R50 as deployed; `configs/legacy_r101.yaml` = R101 as trained).

3. **Confidence-channel shortcut risk.** Two classes ("Standing one foot", "bend") came only from team-recorded clips, extracted with a different pose model. Joint-confidence features separate the data source almost perfectly, so source, pose model and class are confounded. The evidence is consistent with a shortcut but cannot prove one.

4. **Training lineage.**
   - Segment-to-video matching (`scripts/map_legacy_segments.py`) shows the legacy model trained on almost every annotated Le2i Coffee room/Home video.
   - Only `coffee_room_01/1` is unseen.
   - Lecture room and Office were never used for training, because the official release has no labels for them.

5. **Nothing ran as shipped.** The inference path was rebuilt as `src/fall_detection`. It is tested against a replay of the legacy behaviour (`tests/data/golden_001_legacy_replay.csv`).

## What this repository adds

- A clean, tested CPU pipeline around the inherited models.
- Reproducible keypoint extraction.
- A simple rule baseline.
- Event-level evaluation: fall events, a matching window, and false alarms per hour of non-fall time.
- The L1 evaluation on official Le2i labels (`reports/l1/`).
- A 10-video annotation pilot on the unseen scenes, with a review tool and its evaluation (`reports/pilot/`).

No model was retrained.
