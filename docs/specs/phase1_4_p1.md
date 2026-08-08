# Phase 1.4 — P1, the appearance sanity probe

The prompt below is the spec handed to the agent executing this block. It is
kept in the repo so the reasoning survives the chat it was pasted into.

---

I am executing a geospatial AI research project: probing frozen EO
foundation-model representations for dynamics/forecastability on
GreenEarthNet minicubes, tile 32UNU (Allgäu/Upper Swabia). Act as a
senior ML research engineer.

Standing constraints (true for every block in this project):
- No pretrained model is ever fine-tuned. Frozen encoders, eval() and
  torch.no_grad(), always.
- Every array's shape is printed and asserted.
- Never re-implement a function that already exists in the repo - import
  it. Two files are canonical and must never be duplicated: data/ndvi.py
  (the target) and probes/cv.py (the splits).
- Any number produced outside probes/cv.py does not exist.
- Runtime: CPU is sufficient. sklearn on cached arrays.

REPO STATE. Clone https://github.com/benj2o/terradreamer and read
README.md, docs/DECISIONS.md, log.md, RUNBOOK.md, then the module
docstrings of probes/cv.py and encoders/manifest.py. What exists:

- manifest: (264, 21) rows, one per RETAINED frame. 20 cubes, tile
  ['32UNU'], year [2018]. Columns include cube_id, tile, year, timestamp,
  original_axis_index, pixel_bbox, clear_frac, per-cube and PER-GRID-CELL
  landcover_stratum (esawc_lc), per-cell elevation (cop_dem), and in-cube
  E-OBS weather (8 vars: tg/tn/tx/rr/pp/fg/hu/qq) joined on
  original_axis_index.
- 100 .npz = 20 cubes x 5 encoders, schema v3:
    raw_features            D=35     not a network
    imagenet_vit_b16        D=1536   concat(cls_last, patch_mean_last)
    dinov2_vitb14           D=3840   published linear-probe recipe
    satlas_s2_swinb_rgb     D=1024   single-image, EO-native
    satlas_s2_swinb_mi_rgb  D=1024   MULTI-IMAGE positive control
  Each carries pooled [T_kept, D] fp32, grid [T_kept, 16, D_grid] fp16,
  grid_clear_frac [T_kept, 16], named feature variants (cls_last,
  patch_mean_last, cls_last4_concat where applicable), and
  window_span_days.
- probes/cv.py: six fold modes. On THIS subset (one tile, one year)
  `cube`, `spatial_block` and `temporal` run; `year`, `tile` and
  `crossed` correctly RAISE. Use folds(...) and join_embeddings(...);
  do not hand-roll a split anywhere.
- REPLICATION_STRATA = (cropland, tree_cover, grassland). built_up has 5
  cells and bare_sparse 1 across 320 - report them, never replicate on them.

P1 - appearance sanity probe

GOAL: measure how accessible month and season are from a single frame's
embedding, for every encoder, WITH the raw-feature baseline in the same
table. This probe is EXPECTED to succeed. It is calibration, not a
finding: month is confounded with appearance (greenness, snow, sun angle),
so a high score establishes only that appearance is trivially present,
which is what makes P2 and P3 interpretable. The surprise worth reporting
is the opposite - an EO foundation model that FAILS P1, which would
suggest aggressive appearance-invariance training.

DELIVERABLE: probes/p1_appearance.py plus one results CSV.

Task definition:
- Targets: month (multiclass) and season (4-way), from the manifest
  timestamp. NOTE: this subset spans roughly March-December 2018 only, so
  month is ~10 classes, not 12, and is imbalanced. Print the realised
  class distribution before training anything and derive chance level
  from it - do not assume 1/12.
- Estimators: multinomial logistic regression and ridge classifier.
- Feature sets, each a separate row in the results table:
    (a) pooled embedding, per encoder
    (b) grid embedding flattened per CELL (each of the 16 cells becomes
        its own row, inheriting the frame's month label). This raises the
        row count from 264 to ~4224 and is the primary defence against
        p >> n: pooled DINOv2 is D=3840 against 264 rows.
    (c) raw_features pooled - the mandatory baseline
    (d) DEGENERATE CONTROL: [clear_frac, window_span_days] alone, no
        embedding. If this decodes month above chance, then cloud
        retention itself is seasonal and part of every other row's score
        is retention, not representation. This row is not optional.

Protocol, non-negotiable:
- Splits come from probes/cv.py. Primary: mode="cube", k=5. Robustness:
  LOCO and mode="spatial_block". Report all three; they must agree in
  ORDERING even if not in absolute value.
- NESTED CV for the regularisation strength. Tune alpha/C by an inner
  split on the TRAINING folds only, never on the test fold. This is the
  single easiest way to produce an inflated number here, so make the
  inner loop explicit and print the selected alpha per outer fold.
- Standardisation fit on train only, applied to test. Never fit on the
  full array.
- Metrics: balanced accuracy and macro-F1 (the classes are imbalanced),
  plus a most-frequent-class dummy for the floor. Report mean and the
  SPREAD ACROSS FOLDS, not just the mean - with 20 cubes the fold-to-fold
  variation is the honest uncertainty, and a mean without it is not
  reportable.

Encoder-specific caveat that must be handled, not ignored:
- satlas_s2_swinb_mi_rgb aggregates 8 RETAINED frames, whose lookback
  spans a variable number of days (measured on this subset: min 0,
  median 55, max 105). Its embedding at time t therefore summarises up to
  three months of history, so a single "month" label is ill-defined for
  it in a way it is not for the single-image encoders. Report MI's P1
  score, but label it explicitly as not directly comparable, and
  additionally report it conditioned on window_span_days. Do not quietly
  put it in the same column as the SI encoders and rank them.

FIGURE 1 (the latent clock): PCA of embeddings coloured by month, POOLED
ACROSS CUBES spanning the tile's 16 distinct time windows. It cannot come
from a single cube: ~29 frames over ~150 days is a fragment of an annual
cycle, so one cube gives an arc, not a loop. Produce it for every encoder
on the same axes scale so they are visually comparable.

TESTS:
- Assert every results row has a matching raw_features row for the same
  fold mode and feature set. A table without the baseline is not a table.
- Assert the degenerate control row is present.
- Assert no test-fold index appears in its own fold's training set, by
  re-deriving it from the manifest independently of probes/cv.py.
- Assert the inner tuning loop never sees test indices: pass a deliberately
  poisoned test fold and confirm the selected alpha is unchanged.
- Assert chance level was computed from the realised class distribution,
  not hard-coded.

EXIT TEST: results CSV covering 5 encoders x 4 feature sets x 3 fold
modes, each with balanced accuracy, macro-F1, dummy floor, and
across-fold spread; Figure 1 rendered for all five encoders; the
degenerate control reported; notebook runs clean end to end in a fresh
runtime; full test suite green.

Then update README.md phase status, log.md (the realised class
distribution, chance level, per-encoder scores with spread, selected
alphas), and docs/DECISIONS.md (why grid-cell rows are the primary
feature set; why MI is reported separately).

Generate the code exactly to this spec. After the code, list the three
most likely silent failure modes and the print statement that would
expose each.

---

## Where the spec's assumptions met the data

Two figures in the spec were estimates and are corrected by measurement.
Recorded here so the spec is not read later as if it had been right.

- **"roughly March–December 2018, so month is ~10 classes."** The realised
  subset is **8 months, April–November**. March and December have cube
  windows but no frame that survives `clear_frac > 0.5`. Season is realised
  **3-way**, not 4: there is no DJF frame at all. This is exactly why the spec
  forbids a hard-coded chance level, and the probe derives `1/K` from the
  labels it is handed.
- **"~29 frames over ~150 days" per cube.** 29 is the count on the ORIGINAL
  time axis; after the clear-fraction filter a cube retains 10–16 frames
  (median 13, 264 total). The conclusion the figure supports is unchanged and
  strengthened: one cube is an even smaller fragment of the annual cycle than
  the spec assumed, so Figure 1 pooling across the 16 windows is not optional.

One thing the spec does not mention and the results made load-bearing:
`raw_features` is computed over **all four bands including B8A**, and includes
NDVI statistics, while all four network encoders are **RGB-only**. Any row where
the baseline beats a frozen encoder is a statement about the input as much as
about the representation. See `docs/DECISIONS.md`.
