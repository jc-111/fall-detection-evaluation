# Datasets

> Note for this public copy: paths under `legacy/`, `recovery/` and `LiCode/` refer to the private project archive and are not part of this repository.

## Le2i Fall Detection Dataset (official release)

| | |
|---|---|
| Record | dataUBFC, "Fall Detection Dataset", https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html |
| DOI | **10.25666/DATAUBFC-2024-04-09** (https://doi.org/10.25666/DATAUBFC-2024-04-09) |
| Creators | Julien Dubois, Johel Miteran (contributors listed: Imen Charfi, Rached Tourki); Université Bourgogne Europe, ImViA (formerly Le2i, UMR 6306) |
| Licence | **CC BY-NC-SA 3.0** (http://creativecommons.org/licenses/by-nc-sa/3.0/), as stated on the record: attribution, **non-commercial**, share-alike |
| File | `FallDataset.zip`, 9,608,701,260 bytes, sha256 `01df91e32a034d9897d1f9cd5e0211c70ab54289d27e8490db5a699384c17fc8` (downloaded 2026-10-04) |
| Local path | `data/raw/le2i/` (git-ignored; never commit the raw data) |

**Citation requested by the dataset README:**

> I. Charfi, J. Miteran, J. Dubois, M. Atri, R. Tourki, "Optimised spatio-temporal descriptors for real-time fall detection: comparison of SVM and Adaboost based classification", *Journal of Electronic Imaging* 22(4), 2013. doi:10.1117/1.JEI.22.4.041106

**Dataset citation:** Julien Dubois, Johel Miteran (2014): Fall Detection Dataset. dataUBFC. doi:10.25666/DATAUBFC-2024-04-09

### Structure as actually found

These details differ from the README and from the legacy code's assumptions:

```
FallDataset.zip -> Coffee_room_01.zip, Coffee_room_02.zip, Home_01.zip, Home_02.zip,
                   Lecture_room.zip, Office.zip, README.txt   (zips nested; macOS `unzip` cannot read
                                                               the outer ZIP64 archive, Python's zipfile can)
Coffee_room_01/{Videos, Annotation_files}/video (i).{avi,txt}
Coffee_room_02/{Videos, Annotations_files}/...      <- different folder spelling
Home_01/, Home_02/ {Videos, Annotation_files}/...
Lecture room/video (i).avi                          <- no annotation files
Office/video (i).avi                                <- no annotation files
```

**Inventory** (`data/le2i/inventory.csv`, built by `scripts/le2i_inventory.py`; frame counts are decoded frames):

| Scene | Videos | Fall | No fall | Unknown | Frames | Minutes | fps |
|---|---|---|---|---|---|---|---|
| coffee_room_01 | 48 | 47 | 0 | 1 | 14,428 | 9.6 | 25 |
| coffee_room_02 | 22 | 12 | 8 | 2 | 13,141 | 8.8 | 25 |
| home_01 | 30 | 30 | 0 | 0 | 7,175 | 5.0 | 24 |
| home_02 | 30 | 7 | 23 | 0 | 7,322 | 5.1 | 24 |
| lecture_room | 27 | – | – | 27 | 17,293 | 11.5 | 25 |
| office | 33 | – | – | 33 | 16,552 | 11.0 | 25 |
| **total** | **190** | **96** | **31** | **63** | **75,911** | **50.9** | |

All videos are 320×240.

### Annotation format and issues

**Format** (`src/fall_detection/data/le2i.py`):
- Line 1 is the fall start frame and line 2 the fall end frame, both 1-based; `0, 0` means no fall.
- Then one line per frame: `frame, code, x1, y1, x2, y2`.

**Facts checked by inspection:**
- The README says the boxes are "height, width, centre", but the values are corner coordinates.
- The `code` field (values 1–8) is undocumented. Its values don't line up with the fall intervals, so it is not used.
- Annotated fall start is the onset of the collapse, and fall end is when the person reaches the floor. This was checked visually after converting to 0-based indices.

**Discrepancies:**
- **Video count:** the record states 191 videos; the archive contains 190.
- **Frame rate:** the README states 25 fps for all videos; Home videos are 24 fps.
- **Unannotated scenes:** Lecture room and Office have **no annotations** (no fall ground truth, no boxes).
- **Missing fall header** in `coffee_room_01/26`, `coffee_room_02/50` and `coffee_room_02/52` (has_fall = unknown; excluded from event evaluation).
- **Box-line counts:** most Home files have 1–9 more box lines than decoded frames. `coffee_room_01/35` lists every frame twice (471 lines, 231 frames). Its fall header is valid.
- **Inverted boxes:** a few boxes have x2 ≤ x1 or y2 ≤ y1 (`home_02/36`: 65, `home_02/51`: 24, plus 1 each in `coffee_room_02/68` and `home_01/8`).
- **Audio:** some videos carry an audio stream with broken headers (decoder warnings only).

### Use in this project

- **Research use only, non-commercial**, with attribution.
- Derived data (extracted keypoints, `data/extracted/`) counts as an adaptation and inherits **share-alike**.
- Do not redistribute the raw videos; point to the DOI instead.
