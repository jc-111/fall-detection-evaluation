# Vendored: GajuuzZ/Human-Falling-Detect-Tracks

This directory is **third-party code, not part of this project's own work.** It holds the subset of the upstream repository that the fall-detection pipeline needs at inference time.

| | |
|---|---|
| Upstream | https://github.com/GajuuzZ/Human-Falling-Detect-Tracks |
| Commit | `7ed2faa4d6147dfd576f58869b6c25545208af35` (2021-10-02, "Fix an error in "CamLoader_Q"") |
| Upstream licence | none declared in the repository (checked 2026-10-03) |
| Verification | Each file was byte-compared against upstream's published git blob hash (`git hash-object --no-filters`; hashes in `recovery/provenance/upstream_blobs_GajuuzZ.tsv`). All 38 matched before patching. |

**Upstream itself is built from other projects, each under its own terms:**

| Path | Original project | Licence (as stated by that project) |
|---|---|---|
| `SPPE/`, `pPose_nms.py` | AlphaPose (pytorch-AlphaPose, Jeff-sjtu / MVIG-SJTU) | MIT, see `SPPE/LICENSE` |
| `Actionsrecognition/` | ST-GCN, yysijie/st-gcn (Yan, Xiong & Lin, AAAI 2018) | BSD-2-Clause |
| `Track/` | Deep SORT, nwojke/deep_sort | GPL-3.0 |
| `Detection/` | PyTorch-YOLOv3, eriklindernoren/PyTorch-YOLOv3 | GPL-3.0 |

These licence entries are factual pointers, not a licensing assessment. See `recovery/LICODE_PROVENANCE_REPORT.md` §F before redistributing.

**Not vendored** (not needed for inference): `main.py`, `App.py`, `CameraLoader.py`, `ActionsEstLoader.py`, `Visualizer.py`, `Data/`, `Actionsrecognition/train.py`, `sample1.gif`, and the stray `SPPE/src/utils/dataset/.coco.py.swp`.

## Local patches

The patches are kept minimal and listed here. `git log -- third_party/` shows the pristine import commit followed by each patch.

1. **CPU-capable weight loading** (`SPPE/src/main_fast_inference.py`, `DetectorLoader.py`, `PoseEstimateLoader.py`):
   - remove hard-coded `.cuda()`;
   - pass `map_location` to `torch.load`;
   - let `SPPE_FastPose` take an explicit `weights_file`, because upstream's default is a working-directory-relative path.
   - The numerical behaviour is unchanged.

## How it is imported

This code uses top-level imports (`from Detection.Models import ...`). `src/fall_detection/_vendor.py` puts this directory on `sys.path` in exactly one place. Project code imports these modules only through the adapters in `src/fall_detection/`.
