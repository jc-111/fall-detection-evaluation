# Data and evaluation plan (next phase; nothing here is implemented yet)

> Note for this public copy: paths under `legacy/`, `recovery/` and `LiCode/` refer to the private project archive and are not part of this repository.

**Architecture freeze:** `src/fall_detection/` stays as it is unless an ML or product requirement exposes a concrete problem.

**The one requirement-driven change already foreseen** is this:
- Today, extracting poses for a frame (letterbox → detect → pose → track) and classifying it happen together in `FallDetectionPipeline.process_frame`.
- The dataset builder must reuse *exactly* that extraction stage.
- So the plan is to expose it as `pipeline.extract(video) -> per-frame track records`, with `run_video` = `extract` + classify. It is a split, not a rewrite.
- It is guarded by a new parity test: windows built by the dataset code must equal the classifier inputs the live pipeline produces on the same video.

Evidence tags:
- **[K]** general knowledge of the Le2i dataset, to be verified on acquisition;
- **[V]** verified in this repo.

## 1. Target flow

```
raw video (+ sha256, source metadata)
  → canonical extraction  = the inference pipeline's own letterbox/detector/pose/tracker, one config (extractor_id)
  → per-frame records     = frames table: keypoints, confidences, boxes, track IDs (+ labels/events joined in)
  → temporal windows      = derived on demand by features.normalize_sequence (never stored as the only copy)
  → splits                = assigned per video / take / subject / room (never per window)
  → train / eval datasets = windows filtered by split
```

Project-owned code would live in `src/fall_detection/data/`:
- `extract.py`, which calls the pipeline's extraction stage;
- `labels.py`, joining per-frame labels and fall events;
- `windows.py`;
- `splits.py`.

Evaluation would live in `src/fall_detection/evaluation/`. Raw videos, extracted tables and artefacts stay out of git, under `data/` (ignored) with a hashed manifest, the same way `models/` works.

## 2. Processed-sample schema

The data is stored as four tables plus one config record. A "sample" for training is a window plus the metadata joined from these tables.

**`videos` (one row per source video)**

| Field | Type | Notes |
|---|---|---|
| `video_id` | str | stable, e.g. `le2i/home_01/video_12` |
| `dataset` | str | `le2i`, `custom_2026`, … |
| `source_path`, `sha256` | str | raw-file identity |
| `scene`, `room` | str | e.g. `home_01`; **room is a split axis** |
| `camera_id`, `take_id` | str or null | multi-view recordings of one take share `take_id` |
| `subject_ids` | list[str] or null | null = unknown (never guessed) |
| `fps`, `width`, `height`, `n_frames` | num | |
| `has_fall` | bool or null | from annotations |
| `licence_note` | str | terms the video is used under |

**`extractors` (one row per extraction config)**

| Field | Notes |
|---|---|
| `extractor_id` | hash of all fields below |
| `detector`, `detector_weights_sha256`, `detector_conf/nms/expand` | |
| `pose_model`, `pose_backbone`, `pose_weights_sha256`, `pose_input_hw` | |
| `frame_size`, `frame_border` | the letterbox setting that changed scores in the legacy study |
| `tracker_params` | max_age, n_init, max_iou_distance |
| `code_commit`, `package_versions` | git SHA of this repo; torch/opencv/numpy versions |

**`frames` (one row per video × frame × visible track)**

| Field | Type | Notes |
|---|---|---|
| `video_id`, `frame_index`, `timestamp_s` | | `frame_index` is the decoder index; no silent skipping |
| `extractor_id` | str | every keypoint knows which model produced it |
| `track_id` | int or null | null row = frame processed, nobody detected (kept explicitly) |
| `bbox_xyxy` | 4 × float | original-frame pixels |
| `box_source` | str | `detector`, `kalman`, or `gt` (Le2i boxes, diagnostics only) |
| `keypoints_xy` | 13 × 2 float | original-frame pixels, `features.POSE_JOINTS` order |
| `keypoint_conf` | 13 × float | raw extractor confidences, stored separately from xy |
| `pose_score` | float | |
| `action_label` | str or null | per-frame label; null = unlabelled |
| `label_source` | str | `le2i_recovered`, `manual_v1`, … |
| `fall_event_id` | str or null | set while the frame is inside an annotated fall |
| `subject_id` | str or null | if the track can be tied to a known subject |

**`events` (one row per annotated fall)**

| Field | Notes |
|---|---|
| `event_id`, `video_id` | |
| `start_frame`, `end_frame` | annotated fall onset and impact/end (Le2i provides start/end frames [K]) |
| `on_floor_until_frame` | optional, for "still down" logic |
| `annotation_source` | `le2i_official`, `manual_v1` |

**`windows` (derived, regenerable; never the only copy)**

| Field | Notes |
|---|---|
| `video_id`, `track_id`, `start_frame`, `end_frame`, `extractor_id` | |
| `label` | label at `end_frame`. This is causal and matches inference, which labels the current frame |
| `label_purity` | share of frames with that label (lets evaluation drop ambiguous windows) |
| `fall_event_id` | if the window overlaps an event |
| `split` | joined from `splits` by `video_id` |

**`splits`**
- One row per `video_id`: split name plus the grouping key used (`subject` > `take` > `video`).
- Protocols:
  - `room_holdout`: one whole room/scene is test;
  - `video_kfold`: grouped k-fold.

**Rules that address the legacy failures:**
1. **One extraction path:** training windows are built only through `features.normalize_sequence` on keypoints produced by the inference extraction stage. A parity test enforces it.
2. **Leakage:** stride-1 windows are fine for training, but splits are assigned per video, take or subject, never per window. Evaluation never includes a video that appears in training.
3. **Confidence:** raw confidences are stored separately and tagged with `extractor_id`. Mixing extractor IDs within one dataset is rejected.
4. **Dataset gate:** before training, a confidence-only classifier must not predict source or extractor within a class. Target AUC ≈ 0.5; the legacy data scored 0.99. Ablation with and without the confidence channel is a standing experiment.
5. **No baked-in processing:** labels are stored hard, per frame. Smoothing is a training-time choice, never baked into the data.
6. **Custom classes:** recorded with the same cameras, rooms and people as other classes, inside mixed sequences, and labelled per frame. Every class must come from ≥ 2 sources, rooms or subjects.

## 3. Evaluation ladder

All rungs are scored on the **same test videos, the same splits, and the same event layer**. Rung order is also implementation order.

| Rung | Model | Needs training? | What it establishes |
|---|---|---|---|
| 1 | **Rule-based fall detector** on keypoints: downward hip-centre velocity normalised by body height, torso angle from vertical, and on-floor duration | thresholds tuned on training videos only | The floor any learned model must beat for fall events |
| 2 | **Simple classifier on pose features** (logistic regression and gradient boosting): window statistics of normalised joints, velocities, torso angle, bbox aspect. Run with and without confidence features | yes (cheap) | Whether a graph network is needed at all; how much confidence matters |
| 3 | **Recovered legacy ST-GCN** (frozen checkpoint), with the ResNet-50 config (deployed) and the ResNet-101 config (training-consistent for Le2i) | no | What the inherited model really does on data it never saw |
| 4 | **Retrained temporal model** (same ST-GCN architecture first, then alternatives) on the new dataset | yes (later) | Gain from consistent features and honest splits |

**Constraint on rung 3:** the legacy checkpoint was trained on Le2i Coffee-room and Home. It can only be evaluated honestly on videos outside those. If available, those are other Le2i scenes [K: Le2i is reported to include Office and Lecture-room scenes] or new recordings. Results on Coffee-room and Home are reported only as "seen during training (optimistic)".

### Metrics: frame / action classification

**Unit:** each frame where a confirmed track has a full window. The prediction is the label of the window ending at that frame, against the per-frame ground truth.

**Report:**
- per-class precision, recall and F1 with support;
- macro-F1 (primary);
- the confusion matrix, both as counts and row-normalised;
- everything per split and per room.

Accuracy is reported only as a secondary number, never alone.

Rung 1 predicts coarse classes (fall / lying / upright). It is scored on that coarse mapping, applied identically to every rung.

### Metrics: fall-event detection

- **Event layer (shared by rungs 1–4):** a predicted fall event starts when the fall score (or rule) stays above threshold for *k* consecutive frames. A cooldown follows. Thresholds are chosen on training or validation videos only.
- **Matching:** a predicted event matches a ground-truth event if its start lies in [GT start − 1 s, GT end + 3 s]. Each GT event matches at most one prediction. The tolerances are fixed before seeing test results.
- **Event recall:** matched GT events / GT events.
- **Event precision:** matched predictions / predictions.
- **False alarms per hour:** unmatched predictions / total test-video hours. This includes non-fall videos, which is why `has_fall = false` videos are mandatory.
- **Detection latency:** predicted start − GT start, for matched events. Report median and 90th percentile, in seconds.
- **Operating points:** recall at fixed false-alarm rates (e.g. ≤ 1/h and ≤ 5/h), plus the full recall-vs-false-alarm curve.
- **Uncertainty:** 95% bootstrap confidence intervals, resampling **videos**, not windows.

## 4. What acquiring Le2i unlocks

**Files needed** (format details are [K] and must be verified on arrival):

| Item | Why |
|---|---|
| Video files for the **Coffee_room_01/02** and **Home_01/02** scenes | Confirm the 214-segment map; re-extract the legacy training data with one pipeline (expected ≈ 70 + 60 videos, 320×240 at 25 fps [K]) |
| Videos for the **other scenes** (Office, Lecture room) if present | The only honest test set for the legacy checkpoint (rung 3) |
| **Annotation files** per video: fall start/end frames plus per-frame person boxes | Event ground truth; box diagnostics. Upstream's builder parsed them as `frame_idx, class, xmin, ymin, xmax, ymax` after dropping the header rows [V: `create_dataset_2.py`] |
| Dataset **documentation / terms of use** | Licence for portfolio use; actor and camera info (subject IDs if listed) |
| Not part of Le2i [K] | Per-frame 7-class posture labels. These must be recovered from the legacy pickles via the segment map, or relabelled |

**On arrival:**
- record the sha256 of every file;
- check counts per scene, fps and resolution against the expectations above;
- confirm the annotation format before writing the parser.

## 5. First experiment after Le2i arrives: "L1, ground truth and a first honest number"

No retraining. The only fitting is rule-threshold tuning on training videos.

1. **Confirm lineage:** run the existing segment matcher over every Coffee/Home video, then record which videos the legacy checkpoint trained on, and any frame skips.
   - Pass condition: every one of the 134 Le2i segments maps to one video with a clear margin, as `004.avi` did.
2. **Canonical extraction:** run the frozen pipeline's extraction stage (`legacy_baseline` config) over all Le2i videos and write the `videos`, `frames` and `events` tables.
3. **Recover posture labels** for Coffee/Home from the stored windows via the confirmed map, by inverting the stride-1 window means. Hand-check a random sample of 20 transitions against the video.
4. **Evaluate:**
   - Rung 1 (rule) and rung 3 (legacy ST-GCN, both the ResNet-50 and ResNet-101 configs).
   - **Events** on the held-out scenes: recall, precision, false alarms per hour, latency, with confidence intervals.
   - **Posture classification** on Coffee/Home, labelled "seen during training" for rung 3.
5. **Decisions this experiment informs:**
   - whether the legacy ST-GCN beats a rule on unseen scenes;
   - whether the ResNet-101 config helps when measured on events;
   - whether the recovered posture labels are trustworthy enough to train rungs 2 and 4, or need relabelling.
