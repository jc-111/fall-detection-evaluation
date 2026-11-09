# L1: fall-event evaluation on the official Le2i dataset

> Note for this public copy: paths under `legacy/`, `recovery/` and `LiCode/` refer to the private project archive and are not part of this repository.

**Question:** on genuinely unseen Le2i videos, does the recovered legacy model detect falls better than a simple rule?

**Answer: the official data cannot answer this.**
- Only **one** annotated official video was not in the legacy training data.
- The two genuinely unseen scenes, Lecture room (27 videos) and Office (33 videos), have **no official annotations**.

On the annotated videos, all of which the legacy model trained on except one, the legacy ST-GCN beats the rule by a wide margin: fall recall 0.89–0.91 vs 0.66–0.68, at the cost of 1–2 false alarms. But that is a training-set comparison, and it says nothing about generalisation.

| | |
|---|---|
| Scope | Diagnostic only: no retraining, no pose-model replacement, no UI |
| Data | Official dataUBFC release, doi:10.25666/DATAUBFC-2024-04-09, CC BY-NC-SA 3.0 (`docs/DATASETS.md`) |
| Code | Extraction at clean commits; the code that affects extraction is unchanged across `8dfba43`…`927edfa` (verified with `git diff -- src third_party configs`) |

Derived artefacts in this folder:
- `event_summary.csv`, `unannotated_summary.csv`;
- `legacy_mapping_r50.csv`, `legacy_mapping_r101.csv`, `legacy_split.csv`;
- `extractors.json`.

Raw videos, per-video annotation values and keypoints stay in git-ignored `data/`.

## 1. Official dataset inventory

190 videos, 75,911 frames, 50.9 minutes, all 320×240. Full details are in `docs/DATASETS.md`; the inventory itself is `data/le2i/inventory.csv`.

| Scene | Videos | Fall | No fall | Unknown | fps | Annotations |
|---|---|---|---|---|---|---|
| coffee_room_01 | 48 | 47 | 0 | 1 | 25 | yes |
| coffee_room_02 | 22 | 12 | 8 | 2 | 25 | yes (folder `Annotations_files`) |
| home_01 | 30 | 30 | 0 | 0 | 24 | yes |
| home_02 | 30 | 7 | 23 | 0 | 24 | yes |
| lecture_room | 27 | ? | ? | 27 | 25 | **none** |
| office | 33 | ? | ? | 33 | 25 | **none** |

**Inconsistencies found:**
- The record says 191 videos; the archive has 190.
- The README says 25 fps; Home videos are 24 fps.
- The README says boxes are "height, width, centre"; the values are corner coordinates.
- 3 annotation files lack the fall header.
- Most Home files have 1–9 more box lines than frames, and `coffee_room_01/35` lists every frame twice.
- **3 pairs of byte-identical duplicate videos**, each with slightly different annotations: `home_01/2` = `/3`, `home_01/20` = `/21`, `home_02/49` = `/50`.
- Annotated fall start and end were checked visually: with 1-based numbering converted to 0-based, the start is the collapse onset and the end is reaching the floor.

## 2. Legacy-to-original video mapping

**Method** (`scripts/map_legacy_segments.py`):
- **Content match:** every one of the 134 stored legacy Le2i segments is compared with every official video. The measure is the masked RMSE of per-frame normalised skeletons (`features.normalize_sequence`) at the best frame offset.
- **Order constraint, inferred from the legacy builder:** it concatenated videos in the `sorted(os.listdir())` order of one folder per pickle: "Coffee_room" (videos 1–70) and "Home" (videos 1–60).
- **Assignment:** order-preserving dynamic programming. Confidence comes from agreement with the content-only best match.
- **Cross-check:** run independently on the ResNet-50 and ResNet-101 extractions.

**Results:**
- **ResNet-101 mapping:** 108 high, 10 medium, 16 low-confidence segments.
- **ResNet-50 mapping:** 105 / 16 / 13.
- **Agreement between the two:** 119 of 134 assignments.
- **Coffee pickle (71 segments):** 69 videos in exact lexicographic order (10, 11, … 19, 2, 20, … 9). Videos 68 and 70 were each split into two segments. **Video 1 is absent:** no segment comes closer to it than RMSE 0.54, while true matches are 0.08–0.30 (median of high-confidence matches 0.16).
  - This corrects the earlier lineage hypothesis, which assumed segment 0 was video 1.
- **Home pickle (63 segments):** most segments map consistently with the order. Two stretches (segments 5–10 and 13–17, Home_01 videos 14–21) have weak content evidence, because pose coverage is poor in those videos.
- **No segment maps distinctively to Lecture room or Office.** Their best matches have RMSE 0.36–0.48 with margins of at most 1.08, which is noise. Neither folder was used by a legacy pickle.

The frame range of each segment within its video is in the mapping CSVs (`frame_start`, `frame_end`).

## 3. Seen / unseen split

`scripts/le2i_legacy_split.py` combines both mappings conservatively: the stronger "seen" evidence wins, and "unseen" requires both to agree.

| Scene | Seen | Probably seen | Uncertain | Unseen |
|---|---|---|---|---|
| coffee_room_01 | 47 | – | – | **1** (`video 1`) |
| coffee_room_02 | 22 | – | – | – |
| home_01 | 17 | 11 | 2 | – |
| home_02 | 22 | 8 | – | – |
| lecture_room | – | – | – | **27** |
| office | – | – | – | **33** |

**Unseen *and* annotated: one video (`coffee_room_01/1`).** The intended primary test, Lecture room and Office, is verified unseen but has no official ground truth.

## 4. Detectors

All detectors see the same canonical extraction (`src/fall_detection`, letterbox 384 with replicate padding), recorded in `extractors.json`:

| Extraction | extractor_id | Pose model | Crop | Throughput (CPU, M1 Pro) |
|---|---|---|---|---|
| `legacy_baseline` | `15efdb82c39c` | AlphaPose FastPose ResNet-50 | 224×160 | 15 fps alone, ~13 fps in parallel; 92 min for all 190 videos |
| `legacy_r101` | `7bf0c80fee72` | AlphaPose FastPose ResNet-101 | 320×256 | 7.4 fps alone, ~8 fps; 155 min |

**Rule baseline** (`baselines/rule.py`, thresholds fixed a priori and never tuned):
- A track's frame is flagged when its hip centre has dropped ≥ 1 torso length compared with about 1 s earlier, **and** its torso is tilted ≥ 50° from vertical.
- Torso length is the track's median over the previous 2 s.
- Frames with mean shoulder/hip confidence < 0.2 are skipped.

**Legacy ST-GCN:** the frozen 9-class checkpoint. A frame is flagged when "Fall Down" is the argmax for any confirmed track. Its windows are identical to the live pipeline's; a parity test enforces this.

**Event rule (all detectors):** flagged frames merge into one event when less than 2 s apart. The detection time is the first flagged frame.

**Matching:**
- A prediction matches a fall if it starts in [fall start − 1 s, fall end + 3 s].
- Each fall matches at most one prediction.
- Every other prediction is a false alarm, including repeat alarms.
- False alarms per hour are computed over non-fall time, i.e. video time outside those windows.
- The unit is whole videos.

## 5. Legacy-model evaluation

**All 127 annotated videos** (96 falls, 31 no-fall videos, only **0.32 h of non-fall time**). 95% confidence intervals come from 2,000 bootstrap resamples of videos.

| Detector | Recall [95% CI] | Precision [95% CI] | False alarms | FA/h [95% CI] | Latency median / p90 |
|---|---|---|---|---|---|
| rule (ResNet-50) | 0.69 [0.60, 0.78] | 1.00 [1.00, 1.00] | 0 | 0.0 [0, 0] | 0.37 / 0.56 s |
| rule (ResNet-101) | 0.66 [0.56, 0.74] | 1.00 | 0 | 0.0 | 0.37 / 0.59 s |
| legacy ST-GCN (ResNet-50, deployed) | 0.90 [0.84, 0.95] | 0.97 [0.92, 1.00] | 3 | 9.3 [0, 20.8] | 0.60 / 0.92 s |
| legacy ST-GCN (ResNet-101) | 0.92 [0.86, 0.97] | 0.99 [0.96, 1.00] | 1 | 3.1 [0, 10.5] | 0.36 / 0.56 s |

**With duplicates removed** (124 videos, 94 falls): rule 0.68 / 0.66 recall with 0 false alarms; ST-GCN 0.89 (2 false alarms) and 0.91 (1 false alarm).

**Paired per fall** (deployed configurations):

| | ST-GCN detects | ST-GCN misses |
|---|---|---|
| Rule detects | 62 | 4 |
| Rule misses | 24 | 6 |

With ResNet-101: 62 / 1 / 26 / 7.

**Per scene** (all annotated videos; full table in `event_summary.csv`):

| Scene | Falls | Rule R50 recall | ST-GCN R50 recall / FA | ST-GCN R101 recall / FA |
|---|---|---|---|---|
| coffee_room_01 | 47 | 0.66 | 1.00 / 0 | 1.00 / 0 |
| coffee_room_02 | 12 | 0.75 | 0.92 / 1 | 1.00 / 0 |
| home_01 | 30 | 0.70 | 0.73 / 2 | 0.80 / 1 |
| home_02 | 7 | 0.71 | 0.86 / 0 | 0.71 / 0 |

**The one unseen annotated video (`coffee_room_01/1`, one fall):** both ST-GCN configurations detect it, with latency 0.92 s (ResNet-50) and 0.44 s (ResNet-101). The rule misses it: the hip drop was 0.47 torso lengths. One video is anecdote, not evidence.

**Unannotated unseen scenes (truth unknown, so these are not scores):**

| | Lecture room (27 videos, 11.5 min) | Office (33 videos, 11.0 min) |
|---|---|---|
| Videos flagged by: rule R50 / ST-GCN R50 / ST-GCN R101 | 11 / 19 / 13 | 10 / 18 / 18 |
| Predicted events per hour: rule R50 / ST-GCN R50 / ST-GCN R101 | 57 / 99 / 68 | 54 / 103 / 114 |

Every video the rule flags is also flagged by ST-GCN R50. Whether ST-GCN's extra flags are falls the rule missed or false alarms **cannot be determined without annotations**.

## 6. Event-level results: what they do and do not show

- **Not a generalisation result.** On videos the legacy model was trained on, it detects about 89–92% of falls versus about 66–69% for an untrained rule. Training-set exposure inflates exactly this kind of comparison. Only n = 1 unseen annotated video exists.
- **False-alarm rates are barely measured.** Le2i's annotated videos are short and fall-centred: 0.32 h of non-fall time in total, with no long daily-activity footage. The ST-GCN rates (3–9 per hour) rest on 1–3 events, and the rule's 0 per hour on zero events. The confidence intervals span 0 to about 20 per hour.
- **ResNet-101 vs ResNet-50 for the legacy model:** similar recall (0.92 vs 0.90; they disagree on only 8 falls), fewer false alarms (1 vs 3), and lower latency (median 0.36 s vs 0.60 s). This is consistent with the earlier finding that the checkpoint's Le2i training features came from ResNet-101, but it is again on training videos.
- **Latency:** all detectors fire within about 0.4–0.9 s of the annotated fall onset, so latency doesn't distinguish them here.

## 7. Failure cases

**Shared failure: pose extraction under occlusion.**
- Six falls are missed by both the rule and the deployed ST-GCN, mostly in `home_01/10–18`.
- There the person falls behind the table or chairs and is detected in under 50% of frames. Example: `home_01/15`, where the person falls behind furniture and crawls.
- No classifier can fix missing poses. This affects every option that keeps the current detector and pose stack.

**Rule misses (30 with ResNet-50):**
- **16:** hip drop below 1 torso length within 1 s. Eight of these are near the threshold (0.8–0.97): slow or partly sideways falls, or falls toward the camera.
- **5:** torso tilt below 50° at the drop.
- **9:** pose coverage under 50%.
- The rule was not tuned. A threshold sweep on these videos would be tuning on the evaluation set, so it wasn't done.

**Legacy ST-GCN misses** (ResNet-50: 10 falls; ResNet-101: 8): concentrated in `home_01/10–18`, the occlusion cases above.

**Legacy ST-GCN false alarms** (ResNet-50: 3; ResNet-101: 1):
- 2 of the 3 ResNet-50 alarms are **repeat alarms for the same fall**: one during the fall, one while the person lies on the floor (`home_01/2`, and its duplicate `/3`). A product would suppress these with an event state; the strict scoring counts them.
- The remaining one (`coffee_room_02/68`, 39.1 s) is in a no-fall stretch.

**Dataset issues that affect any evaluation:**
- duplicate videos;
- no ground truth for 60 of 190 videos;
- 3 videos lacking fall headers (excluded);
- very little non-fall time.

## 8. Implications for the next model decision

1. **The decisive question is still open.** Answering it needs **fall annotations for Lecture room and Office**: 60 videos, 22.5 minutes. Each needs a fall start/end frame, or an explicit "no fall", in the same format as the official files. This is the smallest action that unlocks an honest legacy-vs-rule comparison. It is a labelling task for you to approve: who labels, and whether a second annotator checks a sample. I haven't done it, because guessed labels would defeat the purpose.
2. **Don't promote the legacy ST-GCN on these numbers.** They are training-set numbers.
   - Its apparent advantage over the rule (about +24 falls) could be real, since the rule is crude, or it could be memorisation.
   - The paired table at least shows that the ST-GCN finds nearly everything the rule finds and more, *on seen data*.
3. **If the legacy model is kept for now, use the ResNet-101 extraction.** On the evidence so far it has the same or better recall, fewer false alarms and faster detection, and it matches how the Le2i training features were made. Two costs: half the CPU throughput, and the earlier warning that the bend and one-foot classes become unreliable.
4. **Le2i alone cannot measure false alarms.** Product-relevant false-alarm rates need long non-fall recordings (daily activity). Plan to record or obtain them before any alerting work.
5. **Pose coverage under occlusion is a shared ceiling.** That favours eventually evaluating a different pose or detector stack (backlog M5), but only after (1), using the same harness.
6. **Retraining remains blocked on the same point.** Any retrained model needs a held-out scene with ground truth. With Lecture room and Office annotated, the clean protocol is: train on Coffee/Home (with the legacy segment map available for label recovery), test on Lecture/Office.
