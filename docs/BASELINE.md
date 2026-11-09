# The canonical baseline

> Note for this public copy: paths under `legacy/`, `recovery/` and `LiCode/` refer to the private project archive and are not part of this repository.

## Execution path

`scripts/run_video.py` calls `FallDetectionPipeline.run_video()`, which runs these steps for each frame:

1. **`frames.letterbox`:** BGR → RGB, resize to 384 on the longer side, pad to 384×384 (edge replication).
2. **`detector.PersonDetector`:** Tiny-YOLOv3 one-class person boxes.
3. **`tracking.PersonTracker.predict`:** Kalman-predicted boxes of existing tracks are added as extra candidate boxes.
4. **`pose.PoseEstimator`:** SPPE FastPose ResNet-50 at 224×160 on every box, then pose-NMS merges overlapping poses. 13 joints: eyes and ears dropped.
5. **`tracking.PersonTracker.update`:** IoU/Kalman matching. Tracks confirm after 3 hits; `max_age=1`.
6. **For each confirmed track seen in this frame, once it holds 30 keypoint frames:** `features.normalize_sequence` (per-frame x/y min-max to [-1, 1], plus a neck node), then `features.to_model_inputs` (points plus frame-difference motion), then `classifier.ActionClassifier`. That gives 9 sigmoid scores, and the label is their argmax.
7. **`outputs`:** one `FrameResult` per frame, with coordinates mapped back to original-frame pixels. Written as JSON (full) and CSV (one row per frame × person).

## Verified equivalence with the legacy app

The labels and scores are identical to `recovery/run_video.py`, a replay of the legacy `legacy/web_app/detect/main.py` on CPU. This was checked on all four legacy test videos:
- 1,026 person-frames;
- labels match 100%;
- scores agree to within the CSV rounding (5e-5).

`tests/test_pipeline_smoke.py::test_matches_legacy_replay_on_001` guards this. A one-setting change, such as padding with black borders as upstream does, shifts scores by about 0.01 and fails the test.

## Deliberate differences from the legacy web app

| Legacy (`legacy/web_app/detect`) | Baseline | Effect on predictions |
|---|---|---|
| Hard-coded CUDA | CPU by default (`device` in config) | none |
| Weight paths relative to the working directory | resolved from `models_dir` against the repo root | none |
| Threaded `CamLoader` (can drop frames or end early) | synchronous `cv2.VideoCapture` | none on files |
| `sklearn.utils.linear_assignment_` (removed API) | upstream's `scipy.optimize.linear_sum_assignment` | none observed (verified parity) |
| Wall-clock "action durations" in module globals | frame-index timestamps in the result | n/a |
| Image size passed to normalisation as (h, w) | (w, h) | none: 384×384 is square, and normalisation is scale-invariant |
| Models reloaded per HTTP request; DB writes every poll | models load once per pipeline | n/a |
| Annotated frames at 768×768 (2× letterbox) | original resolution | cosmetic |

## Known limitations (inherited, unchanged on purpose)

- **Pose-model mismatch:** the 9-class checkpoint's Le2i training features match ResNet-101 extraction, while inference uses ResNet-50. Walking is often labelled "bend" as a result (`recovery/DIAG_004_REPORT.md`). Switching to ResNet-101 is a config change, but it is deliberately **not** made (see `recovery/CONFIDENCE_REPORT.md` §E).
- **Confidence shortcut:** "bend" and "Standing one foot" come from single-source footage separable by confidence alone, and the classifier relies on it (`recovery/CONFIDENCE_REPORT.md`).
- **No event logic:** the output is per-frame labels. There is no fall event, smoothing or alert.
- **Not evaluated:** no held-out evaluation exists (`recovery/DATASET_REPORT.md`).
- **Single-person bias:** `max_age=1` drops tracks after one missed frame.
