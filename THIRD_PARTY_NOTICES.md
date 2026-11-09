# Third-party components and licensing

**Not redistributed here:**
- the inherited model code;
- the model weights;
- the Le2i videos.

This repository contains only original code and data we can share. The rest has to be obtained from its source.

| Component | Source | Licence | In this repo? |
|---|---|---|---|
| Fall-detection pipeline code (Tiny-YOLOv3 loader, SPPE FastPose, tracker, two-stream ST-GCN, utilities) | [GajuuzZ/Human-Falling-Detect-Tracks](https://github.com/GajuuzZ/Human-Falling-Detect-Tracks) @ `7ed2faa4` | **No licence declared** (all rights reserved by default) | **No.** `scripts/fetch_third_party.py` clones it. We ship only file hashes (`third_party/upstream_blobs_GajuuzZ.tsv`), the file list, and our own CPU-loading patch (`third_party/cpu_loading.patch`) |
| Code that upstream code derives from | AlphaPose SPPE (MIT), ST-GCN by yysijie (BSD-2-Clause), deep_sort by nwojke (GPL-3.0), PyTorch-YOLOv3 by eriklindernoren (GPL-3.0) | as listed | No (comes with the fetched upstream code) |
| Model weights: Tiny-YOLO, FastPose R50/R101, 7-class ST-GCN | upstream author's Google Drive links (`models/MANIFEST.tsv`) | No licence declared | **No.** Download them yourself |
| 9-class ST-GCN behind the reported results | retrained by the original student project | not licensed for redistribution | **No (private)** |
| Le2i Fall Detection Dataset | dataUBFC, [doi:10.25666/DATAUBFC-2024-04-09](https://doi.org/10.25666/DATAUBFC-2024-04-09) | CC BY-NC-SA 3.0 | **No** videos or frames. Only video identifiers, frame counts and hashes |

## Licences of the material in this repository

- **Original code** (`src/`, `scripts/`, `tests/`, `configs/`, `third_party/cpu_loading.patch`): MIT. See `LICENSE`.
- **Annotations and reports** (`annotations/`, `reports/`, `docs/`): these describe and derive from the Le2i dataset. They are therefore shared under **CC BY-NC-SA 3.0**:
  - attribution to Le2i / dataUBFC;
  - non-commercial use only;
  - share-alike.

## Note on combining

The fetched upstream code includes GPL-3.0-derived parts (the tracker and the YOLO utilities). Our MIT code imports them at run time.

If you **distribute** a combined work, the GPL-3.0 terms apply to it. The upstream repository's lack of a licence also means its code has no grant of redistribution rights. That is why it is fetched rather than included here.

This is not legal advice.
