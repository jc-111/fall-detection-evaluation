# Le2i unseen-scene annotation pilot

Manual fall-event ground truth for 10 videos from the two Le2i scenes the legacy model never saw
and the official release does not annotate (Lecture room, Office).

- `selection.json`: the 10 videos, chosen with a fixed seed from the inventory only (`scripts/select_pilot.py`).
- `proposals.json`: UNVERIFIED fall-event proposals from visual inspection (see `reports/pilot/PILOT_PROPOSAL_REPORT.md`).
  Not ground truth.
- `annotations.json`: human-reviewed labels (all 10 proposals were accepted unchanged after a rapid review, so treat them as
  human-accepted proposals, not independent annotations; see `reports/pilot/PILOT_EVAL_REPORT.md` §1), written only by `scripts/annotate_falls.py` after a human accepts
  or edits a video (each entry records `review` and the proposal it started from); format in `src/fall_detection/annotation.py`.
  Frame indices are 0-based in sequential decode order (the numbering the pipeline uses).
- Definitions (as the official Le2i ground truth): **start** = first frame of visible loss of balance /
  uncontrolled descent; **end** = first frame the person is on the ground and the fall motion has stopped. Deliberately sitting or
  lying down is not a fall. Several falls per video are allowed. No posture classes are annotated.
- The annotation tool shows no model output.

Workflow: `annotate_falls.py` (review mode: accept `y` or correct each proposal) → `check_annotations.py` (reload, consistency checks, contact sheets of every
marked interval) → `eval_pilot.py` (refuses to run until all 10 videos are annotated).

Licence: these labels describe videos of the Le2i Fall Detection Dataset (doi:10.25666/DATAUBFC-2024-04-09,
CC BY-NC-SA 3.0) and are shared under the same terms: attribution, non-commercial, share-alike.
