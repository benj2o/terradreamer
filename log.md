# Working log

Running record of measurements and adopted definitions. Reverse chronological.
Decisions and their rationale live in [docs/DECISIONS.md](docs/DECISIONS.md).

## 2026-08-19 (later): the `loco` top-up on lxhalle -- the table is COMPLETE at 1540 rows, and the conclusion does not depend on the fold mode

The 308 `loco` rows rule 4 dropped locally, computed on TUM's lxhalle
(AMD EPYC 9554P, 64 cores) and merged back. **The extreme-tile P3 table is no
longer a subset.**

```
local  (Mac, 7 workers)   1232 rows  cube + spatial_block   6.40 h
server (lxhalle, 64 wk)    308 rows  loco                   1.95 h
merged                    1540 rows  = the full grid
```

`scripts/merge_loco.py` refused-or-passed on identity, not on trust: no
`fold_mode` overlap, identical `n_cubes` (342), `cubes_excluded` (the same 6),
`plausibility_screen` and `tile`. Library versions were pinned to the Mac's
(numpy 2.0.2 / pandas 2.3.3 / sklearn 1.6.1 / scipy 1.13.1) so the two halves
are the same experiment, not merely similar ones.

### `loco` vs `cube`, `cube_mean` / `nested_cv` / `+base`, skill vs persistence

```
view                        D=5     D=25    D=50   D=100   | loco-cube at D=5
raw_features               +0.491  +0.656  +0.728  +0.834  |  +0.008
imagenet_vit_b16_cir       -0.259  +0.497  +0.726  +0.861  |  +0.108
satlas_s2_swinb_mi_rgb_cir +0.180  +0.499  +0.666  +0.846  |  +0.112
dinov2_vitb14              -0.464  +0.436  +0.717  +0.860  |  +0.170
satlas_s2_swinb_mi_rgb     +0.078  +0.424  +0.676  +0.837  |  +0.118
```

`loco` reads **uniformly slightly higher** than `cube` (+0.002 to +0.170,
largest at D=5). That is the expected direction and not a finding: a `loco`
fold trains on 341 cubes against ~274 for 5-fold `cube`, so it fits marginally
better. The ORDERING is unchanged and `raw_features` still leads at D=5.

### THE ROBUSTNESS RESULT: the crossover is not an artefact of the fold mode

Separably better / worse than persistence, all rows, per horizon:

```
          cube (of 228)     spatial_block (of 76)   loco (of 76)
D=5      41 / 158           8 / 64                  10 / 59
D=25    109 /  73          41 / 32                 43 / 32
D=50    169 /  35          43 / 16                 55 / 17
D=100   189 /  33          44 /  8                 60 / 16
```

**All three fold modes show the same shape**: persistence dominates at D=5,
parity around D=25, models dominate from D=50. The headline -- that frozen
encoders do not beat persistence at the horizons early warning would use --
survives leave-one-cube-out, which is the strictest of the three.

### Compute, measured

`loco` at 342 cubes: **23.0 s/row, 304 evaluate calls, 116.7 min on 64
workers**. Against the local projection of 10.12 h on 7 workers, that is
consistent (~5.2x fewer worker-seconds at 9x the workers). A local attempt had
measured **344 s/row** and was heading for ~29 h; the server did the same work
in under two hours on a machine already at load 48.

The 4 `horizon_only` rows carry no `evaluate()` call, so 304 counted rows IS
the complete 308-row table. A run that stops at "row 304" has finished, not
failed -- worth knowing before anyone panics at a vanished process again.

## 2026-08-19: Extreme-tile P3 on 32UQC -- no frozen encoder beats persistence until 100 days on stressed land, and the crossover moves a full horizon step LATER

The nine-view forecastability probe on the 2018 heat/drought tile, same four
Tier-1 corrections, plausibility screen applied, read against the 115-cube
32UNU Tier-1 table. Embeddings built on Colab (T4); everything below is CPU.

**This is a SUBSET table.** `fold_mode=loco` was dropped by rule 4 -- see the
compute note at the end. All nine encoder views survive; that was the point.

### Roster: four separate exclusions, 348 -> 342

```
348   non-overlapping cubes, 32UQC, extreme split
-2    P4 fill block (zero-reflectance B04/B8A), the pilot's own exclusion
-3    encode-time plausibility: implausible valid pixels (>1.2) at
      1.38e-04 / 1.44e-04 / 4.10e-04 of valid pixels, tolerance 1e-04
      -- bright cloud leaking THROUGH the mask
-1    cached mask != isfinite(canonical NDVI) on 2923 px in 1 of 21 frames
342   fitted
```

The last two are NEW and neither was pre-authorised. The 3 encode failures are
the frozen encoders' own guard firing; **32UNU had none of these.** The 1 mask
mismatch is the zero-reflectance fill block again, on a cube the P4 pilot's
61-grid-cell criterion did not catch -- and the cached mask reproduces
BIT-IDENTICALLY from `cube_masks` locally, so nothing is stale: `cube_masks`
and `cube_ndvi` genuinely disagree. P4 fitted 346, P3 fits 342.

### Geometry

```
                        32UQC              32UNU (Tier-1)
cubes fitted            342                115
rows retained  D=5      2162               510      (4.2x)
               D=25     1879               484      (3.9x)
               D=50     1963               445      (4.4x)
               D=100    1137               196      (5.8x)
table rows              1232 (subset)      1540 (full)
held-out predictions    10 289 818         173 310
wall                    6.40 h / 7 workers 2.90 h
rgb cache               1715 .npz, 1093 MB
cir cache               1372 .npz, 1084 MB
masks                   343 .npz, 2.1 MB
```

### THE HEADLINE: `cube_mean` / `cube` folds / `nested_cv` / `+base`, skill vs persistence

```
view                          D=5     D=25    D=50   D=100  |  32UNU D=5   D=100
raw_features                +0.483  +0.633  +0.724  +0.821  |    +0.583  +0.585
imagenet_vit_b16_cir        -0.367  +0.484  +0.711  +0.847  |    +0.071  +0.370
dinov2_vitb14               -0.634  +0.412  +0.709  +0.844  |    -0.219  +0.483
dinov2_vitb14_cir           -0.576  +0.430  +0.705  +0.835  |    -0.336  +0.524
satlas_s2_swinb_mi_rgb_cir  +0.068  +0.455  +0.657  +0.838  |    -0.108  +0.292
satlas_s2_swinb_mi_rgb      -0.040  +0.377  +0.660  +0.835  |    -0.161  +0.506
satlas_s2_swinb_rgb_cir     -0.177  +0.461  +0.651  +0.821  |    +0.025  +0.277
satlas_s2_swinb_rgb         -0.233  +0.372  +0.671  +0.800  |    -0.238  +0.426
imagenet_vit_b16            -0.563  +0.370  +0.638  +0.765  |    -0.110  +0.335
```

**At D=5 the hand-crafted `raw_features` is the ONLY view with positive skill**
(+0.483); every network is negative, two of them separably worse than
persistence. Every view is higher than its 32UNU counterpart at D>=25 -- but so
is persistence's own difficulty, and the comparison that matters is the paired
one below, not the level.

### Separably better / worse than persistence, all rows, per horizon

```
        32UQC (of 308)        32UNU (of 385)
D=5     49 better 222 worse   84 better 196 worse
D=25   150 better 105 worse   60 better 146 worse
D=50   212 better  51 worse  203 better  98 worse
D=100  233 better  41 worse  170 better  90 worse
```

### THE TRIGGER TABLE: Peirce skill, `extreme_low`, vs persistence

```
                    32UNU 2026-08-16          32UQC 2026-08-19
D      persistence  best fc  sep +/-    persistence  best fc  sep +/-
5        +0.585     +0.511   0 / 4        +0.834     +0.681   0 / 8
25       +0.300     +0.512   0 / 0        +0.627     +0.492   0 / 8
50       +0.102     +0.380   2 / 0        +0.223     +0.281   0 / 0
100      +0.087     +0.342   0 / 0        +0.026     +0.236   2 / 0
```

**The crossover SURVIVES and MOVES ONE FULL HORIZON STEP LATER.** On 32UNU the
first horizon at which any frozen view is separably better than persistence is
**50 d**; on stressed land it is **100 d**. And persistence's dominance is not
merely longer but harder: at D=25 32UNU was at parity (0 separably worse),
while on 32UQC **all eight views are separably worse**. At D=100 the two that
win are `dinov2_vitb14` (+0.21) and `satlas_s2_swinb_rgb_cir` (+0.20).

Do not soften this: **for early warning at the horizons an operator would
actually use (5-25 d), every frozen representation is separably WORSE than
persistence on the extreme tile.**

### `_cir` vs `_rgb`, paired, same folds

```
twin                         D=5     D=25    D=50   D=100    separable at
imagenet_vit_b16_cir       +0.024  +0.047  +0.074  +0.155    25, 50, 100
satlas_s2_swinb_rgb_cir    +0.007  +0.037  -0.020  +0.040    25, 100
dinov2_vitb14_cir          +0.007  +0.007  -0.005  -0.016    never
satlas_s2_swinb_mi_rgb_cir +0.013  +0.032  -0.004  +0.004    never
```

Only `imagenet_vit_b16` gains consistently from the NIR band swap; the largest
single effect in the run is its +0.155 at D=100. DINOv2 gains nothing anywhere.
The NIR headline is therefore **encoder-specific, not a general property**.

### `weather_only` GAINS, and that is a warning not a result

```
        32UQC   32UNU
D=5    +0.368  +0.266
D=25   +0.491  +0.329
D=50   +0.611  +0.318
D=100  +0.575  +0.454
```

Separably better than persistence at ALL FOUR horizons. Read against the P4
measurement that **~93% of a typical windowed weather feature is recoverable
from the DATE alone on this tile** (32UNU: ~61%), this is mostly calendar
fitting, and it is the single strongest reason not to read the D>=50 numbers
above as evidence of weather-driven forecast skill.

### Compute note -- and a measurement failure worth recording

Rule 4 dropped `loco`. The clean two-point calibration (343 cubes, quiet
machine) measured `cube` +0.71, `loco` **+1.31**, `spatial_block` -0.52,
totalling 12.10 h against a 12 h budget, with `loco` alone 10.12 h of it.

A SECOND calibration, run while Spotlight was reindexing the freshly copied
2 GB cache, measured `loco` at 33.5 s/row at 20 cubes and 5.3 at 40 -- exponent
**-2.65**, projecting 0.00 h, and declared the full table affordable. It was
about to run `loco` at 342 cubes. **A negative exponent is not a measurement,
it is a corrupted one**: no fold mode gets cheaper per row as the tile grows.
`project()` now rejects negative exponents, falls back to linear growth from
the worse point, and says so. Note the clean run's `spatial_block` -0.52 was
the same defect, and it is why that run's 1.98 h projection was optimistic
against the 6.40 h actual.

Wall-clock figures from this run are contaminated by machine load and should
not be quoted as compute costs. The scientific numbers are unaffected:
contention changes how long a fit takes, not what it returns.

## 2026-08-17: Extreme-tile P4 pilot on 32UQC -- the ceiling does NOT rise on stressed land, and the confounding gets worse

The 2018 heat/drought check the paper's Limits paragraph promises. One extreme
tile, Stage A only, same screen, fold modes and estimators as the screened
115-cube 32UNU run, so the two tables sit side by side. **`weather_full8` ran on
both**, so the comparison needs no footnote.

### Geometry

```
                        32UQC              32UNU (screened)
cubes                   346 (of 348)       115
frames                  6990               1580
frames/cube             20.2               13.7
cell_mean rows          110 949            25 013
year                    2018 only          2018 only
day-of-year span        32-327 (295 d)     235 d
window per cube         2018-01-28..11-23  150 d
weather block           weather_full8      weather_full8
implausible frames      2/6990 (0.03%)     3/1580 (0.19%)
  cell_mean dropped     -24 rows           -36 rows
```

Stage B **deferred**, same reason as 32UNU: single-year, 0 cubes span a year
boundary. Reported, never gated on -- Stage A is the within-season proxy regime
either way, which is the regime 32UNU's number was computed in.

### THE HEADLINE: `cell_mean` / HGB / `weather_full8` / Stage A

```
mode/kind          32UQC R2   obs margin  DOY margin | 32UNU R2  obs     doy
cube/weather         +0.111      +0.086      +0.096  |  +0.116  +0.120  +0.086
loco/weather         +0.078      +0.056      +0.063  |  +0.096  +0.117  +0.066
cube/doy             +0.015                          |  +0.030
loco/doy             +0.015                          |  +0.030
cube weather 95% CI [+0.031, +0.191], 5 folds, effective n 346 CUBES
loco weather 95% CI [+0.037, +0.118], 346 folds
```

**The ceiling is LOWER on the stressed tile, not higher** -- `cube` -0.005,
`loco` -0.018 -- and the margin over the observation-process control falls by
more (-0.034 and -0.061). The pilot's hypothesis, that a heat/drought geography
would make the weather-attributability ceiling larger, is **not supported**.

What did improve is the control-beating RATE, and it is worth stating because it
cuts the other way:

```
                                          32UQC     32UNU
weather rows at or below the obs control   14/54     23/54
weather rows at or below the DOY control   14/54     29/54
weather rows whose CI includes zero        25/54     24/54
permutation control, max r2_vs_clim        -0.002    -0.012
```

Three times the cubes tightened `loco` (CI width 0.081 against 0.134) and did
**not** move the fraction of weather rows whose interval still spans zero.

### THE FINDING THAT MATTERS MORE THAN THE HEADLINE

`print_doy_weather_collinearity`, measured before anything is fitted:

```
                                             32UQC      32UNU
distinct days of year                        47         41
day-of-year span                             295 d      235 d
max cubes sharing one date                   346        114
across-cube spread / total spread (median)   0.07       0.39
=> weather recoverable from the DATE alone   ~93%       ~61%
```

Every row satisfies `doy % 5 == 2` on both tiles: one Sentinel-2 orbit lattice,
so day-of-year is close to a 47-level categorical. On 32UQC only **7%** of a
typical windowed weather feature's spread is ACROSS CUBES at a given date. 346
cubes packed into one MGRS tile read an E-OBS grid too coarse to tell them
apart, so on any given date they get nearly the same weather.

**This is why the improved DOY margin must not be read as improved separation.**
The DOY control is 6 harmonics -- 13 smooth features. It cannot fit a 47-level
categorical, so it understates what the date alone can do, and it understates it
*harder here* (93% recoverable) than on 32UNU (61%). The +0.096 DOY margin is
weather beating a smooth function of timing, not weather beating timing.

### The two cubes that were dropped, and why

`cube_frame_targets` refused 2 of 348 cubes: a grid cell had clear pixels and no
finite NDVI. Diagnosed, not worked around -- **all 57 720 "clear" pixels in the
61 offending cells (0.055% of 111 968) carry exactly-zero reflectance in B04 and
B8A**, a no-data fill block the published `s2_dlmask`/SCL conjunction does not
flag. `finite_valid_mask` cannot demote them: the bands are finite, just zero.
`data.ndvi.ndvi`'s `|B8A+B04| < 1e-12` guard correctly returns NaN. The
assertion was right and caught a fill region before it was averaged into a
target.

The cubes are excluded WHOLE, nothing filled, `n_cubes_excluded_fill` and the
reason on every CSV row. The proper fix -- a zero-reflectance rule beside
`finite_valid_mask` -- is a SHARED path that P1/P2/P3 and the published 32UNU
tables read through, so it was not touched. Both offenders were heavily clouded
(1 and 7 retained frames): 8 frames of 6998 lost.

### Wall clock, and the cost model that was wrong

**Projected 4.7 CPU-hours, actual 7.1** (`run_stage_a` 427.4 min against 281.2
projected, +52%; 445.4 min end to end on 7 workers). The projection was measured
on this tile at 20 and 40 cubes with a per-fold-mode power law:

```
fold mode        20c     40c   exponent   projected   
cube             33s     50s     0.59       0.05 h
loco            116s    384s     1.72       4.39 h     <- dominates
spatial_block    32s     71s     1.17       0.25 h
naive linear-in-cubes projection, for contrast:  1.26 h
```

The memo's **1.3-1.9 CPU-hours by linear scaling was low by about 4x**, and the
measured extrapolation was still low by another 1.5x -- `loco`'s exponent grows
with the extrapolation range because leave-one-cube-out grows its fold COUNT and
its per-fold training set together. Anything sized off linear-in-cubes on this
tile should be multiplied by 4 to 6.

### Artefacts

```
data/scaled_32UQC/p4_extreme_results.csv   270 rows x 105 cols  (invariants pass)
data/scaled_32UQC/p4_extreme_run.log
notebooks/runs/2026-08-17_p4_extreme_32UQC_346cubes.txt
data/scaled_32UQC/raw/                     348 cubes, 2.2 GB
```

No 32UNU artefact touched.

### The go/no-go this was run to answer

The bar, stated against the pilot's own hypothesis: the extreme tile justifies
the gated slim P3 (Aug 18-21) if the ceiling is **clearly higher** than 32UNU's
+0.116/+0.096 **and** the margin over the observation and DOY controls grows
rather than shrinks -- concretely, `cube` at or above roughly +0.15 with an
observation margin at or above 32UNU's own +0.120.

Measured: **+0.111 / +0.078, observation margin +0.086 / +0.056.** Lower on
both axes, on a tile where 93% of the weather is recoverable from the date.
**NO-GO** on the rationale that a heat/drought geography changes the ceiling
story. See DECISIONS.md, 2026-08-17.

## 2026-08-16: Tier-1 trigger metrics -- persistence WINS at short lead, and the two curves cross

The R-squared table re-sliced into a threshold-crossing table. No new fits, no
new geography: the same held-out predictions, scored on "did the forecast fire
when the anomaly crossed the line".

The predictions did not exist as an artefact -- `p3_tier1_results.csv` holds
aggregated fold statistics only -- so P3 gained an opt-in
`emit_predictions=True` that writes one row per held-out observation. It is
opt-in because the published run did not have it, and it is verified free: all
**424 rows** this run shares with `p3_tier1_results.csv` are **bit-identical on
36 scoring columns**.

### Artefacts

```
data/scaled_32UNU/p3_tier1_predictions.csv      173 310 rows x 22 cols, 50.6 MB
data/scaled_32UNU/p3_tier1_triggers.csv            848 rows x 63 cols
data/scaled_32UNU/p3_tier1_subset_results.csv      424 rows x 138 cols  (SUBSET)
```

Scope of the re-run: `cube_mean`, `fold_mode in {cube, spatial_block}`, pooled,
`alpha_rule in {nested_cv, not_a_ridge}`, 4 horizons, 9 encoder views, all
model kinds. `loco` and `cell_mean` are excluded on purpose -- a contingency
table over a two-row LOCO fold is not a contingency table.

**Wall clock. Estimated 8.7 min, actual 15.5 min** (`run_p3` 14.5 min +
manifest 29 s, 7 workers). The estimate extrapolated the fit loop and did not
cost `add_paired_separability` or assembling the 173 k-row CSV. Both are
minutes, not the 173.7 min of the full Tier-1 stack, which is dominated by
`loco` (115 folds) and `cell_mean` (8043-row views).

### The threshold, and where it is fitted

p4's severity rule imported unchanged -- `SEVERITY_QUANTILES` 10th / 30th
percentile of the anomaly from a day-of-year curve -- with the FIT moved inside
the fold. `severity_reference_anomaly` fits on everything and is right to: it
LABELS held-out rows after the fact. A threshold a forecast is SCORED against
cannot, so `_trigger_reference` fits curve and quantiles on the fold's training
rows only, and `assert_thresholds_are_train_fitted` refuses a file where they
could have come from the full sample.

Realised event rates confirm the line lands where it should (nominal 0.10 /
0.30):

```
Delta   extreme_low   low     threshold (median over folds, extreme_low)
   5d       0.112     0.306        -0.103
  25d       0.103     0.306        -0.102
  50d       0.106     0.301        -0.107
 100d       0.117     0.306        -0.171
```

### THE RESULT: persistence decays with lead time and the embeddings do not

`cube_mean` / `cube` folds / `extreme_low` / linear / `nested_cv` / `+base`.
Peirce skill score (hit rate minus false-alarm rate), model vs persistence on
the same rows and the same line:

```
Delta   persistence   best encoder                       paired diff [95% CI]
   5d      +0.585      satlas_mi_rgb_cir  +0.511         -0.074 [-0.335, +0.186]
  25d      +0.300      satlas_mi_rgb_cir  +0.512         +0.212 [-0.082, +0.505]
  50d      +0.102      satlas_mi_rgb_cir  +0.380         +0.278 [+0.121, +0.435]  SEPARABLE
 100d      +0.087      imagenet_vit_cir   +0.342         +0.255 [-0.301, +0.811]
```

Persistence's own contingency table shows why: its hit rate falls
`0.632 -> 0.360 -> 0.213 -> 0.087` across the four horizons while its
false-alarm rate stays near zero. **At 100 d persistence essentially never
fires**, so beating it there is a low bar.

Separable cells among the 32 forecast rows per horizon (cube folds,
`extreme_low`):

```
Delta   separably WORSE than persistence   numerically ahead
   5d              23 / 32                       0 / 32
  25d               6 / 32                      11 / 32
  50d               5 / 32                      18 / 32
 100d               0 / 32                      21 / 32
```

So the two curves **cross somewhere between 25 and 50 days**. At 5 d the result
is unambiguous and it is the paper's thesis in its strongest form: **no encoder
beats persistence at the lead time an early-warning trigger would actually run
on, and 23 of 32 are separably worse.** Past 25 d the sign flips, but the wins
are fragile -- only 2 of 32 cells are separable at 50 d under `cube` folds
(`satlas_mi_rgb_cir`, `dinov2_cir`), a DIFFERENT pair is separable under
`spatial_block`, and nothing at all is separable at 100 d.

Across the whole 848-row trigger table, separable-from-persistence rates are
hit rate 33%, false-alarm rate 47%, CSI 36%, Peirce 38% -- in both directions.

### What this does and does not license

Licensed: "at 5-day lead, frozen EO embeddings do not beat persistence for
bottom-decile anomaly crossings on this tile, and mostly lose to it."

NOT licensed: "foundation models are unnecessary at any lead time." The 50-100 d
crossover is real in sign and unstable in significance; it is a scope boundary,
not a finding. Two of the three `_cir` twins are among the long-horizon leaders,
which is suggestive and nothing more at these event counts (47 and 23).

## 2026-08-13: P2 + P4 screened re-runs on 32UNU -- trust fix, not a new story

Same three implausible frames P3 already dropped (`3/1580`). Published
`p2_scaled_results.csv` / `p4_scaled_results.csv` untouched. Cite
`*_screened_results.csv` for like-for-like with P3.

### P4 (`scripts/rerun_p4_screened.py`, 36.2 min / 7 workers)

Stdout: `notebooks/runs/2026-08-13_p4_screened_32UNU_115cubes.txt`.
Table: `data/scaled_32UNU/p4_screened_results.csv` (270 rows; invariants pass).
Geometry: cube_mean −3, cube_p90 −3, cell_mean −36.

Stage A `cell_mean` / HGB / `weather_full8` (screened | unscreened):

```
mode/kind          screened R2   obs margin   DOY margin | unscreened R2
cube/weather            +0.116       +0.120       +0.086 |        +0.130
loco/weather            +0.096       +0.117       +0.066 |        +0.085
```

### P2 (`scripts/rerun_p2_screened.py`, 113.0 min / 7 workers)

Stdout: `notebooks/runs/2026-08-13_p2_screened_32UNU_115cubes.txt`.
Table: `data/scaled_32UNU/p2_screened_results.csv` (600 rows; invariants pass).
Geometry: **6 pairs dropped** → **1459** pairs scored.

`cube_mean` / pooled / linear / cube folds (screened | was unscreened):

```
SIGN
  raw_features              +0.808 (+0.742) | was +0.785 (+0.720)
  raw_rgb_only              +0.759 (+0.693) | was +0.695 (+0.630)
  dinov2_vitb14             +0.604 (+0.537) | was +0.606 (+0.541)

MAGNITUDE
  dinov2_vitb14             +0.128 (-0.017) | was +0.105 (+0.043)
  raw_rgb_only              +0.119 (-0.026) | was +0.092 (+0.029)
  gap_days control          +0.145          | was +0.063
```

Gate K2 still passes for 3/4 networks (satlas_mi still not separable vs raw).
**Verdict unchanged:** sign recoverable, magnitude not; `raw_*` still above every
network on sign. Screen is a consistency fix with P3, not a claim mover.

## 2026-08-12: Tier 1 -- P3 re-run under four corrections. NO encoder separably beats the band-matched baseline, and three cloud frames were carrying the old result

`scripts/rerun_p3_tier1.py`, local CPU, 7 workers, 115 cubes / tile 32UNU /
2018. Verbatim stdout at
`notebooks/runs/2026-08-12_p3_tier1_32UNU_115cubes.txt`; table at
`data/scaled_32UNU/p3_tier1_results.csv`. Nothing fine-tuned; no encoder
imported.

```
tests                 522 passed, 5 skipped, 0 failed
manifest              REBUILT FRESH: 1580 retained frames, 115 cubes, 2018
                      weather join re-derived from the cubes: 0 rows off their day
encoder views         9 = 5 rgb + 4 cir (Phase 1.9 cache, same weights)
twin caches distinct  max relative difference 0.288 / 0.947 / 0.296 / 0.306
run_p3                1540 rows x 153 cols, 173.7 min on 7 workers
invariants            all THIRTEEN PASS, re-checked on the CSV
effective n           CUBES: 115 / 114 / 115 / 94 at Delta = 5 / 25 / 50 / 100
```

Four things changed at once and each is a column, so any pair of rows in the
table is still comparable: `alpha_rule` (fixed alpha = D vs a penalty selected
by nested CV on the training fold), `feature_base` (with and without a shared
[NDVI(t), weather] block), `plausibility_screen` (applied), and 4 new
colour-infrared encoder views. Every "X beats Y" is now the PAIRED per-fold
difference with a delete-one-fold jackknife interval.

### THE SCREEN WAS THE BIGGEST SINGLE EFFECT, AND IT MOVED THE BASELINE MOST

`p4_ceiling.cube_frame_targets`' `frame_plausible` flags **3 of 1580** retained
frames. Dropping every forecast row that touches one costs **8 / 5 / 5 / 0**
rows at Delta = 5 / 25 / 50 / 100 -- 518 -> 510, 489 -> 484, 450 -> 445, 196
unchanged. Under otherwise the PUBLISHED protocol (fixed alpha, no shared base),
pooled out-of-fold R-squared, `cube_mean` / cube folds / ridge:

```
                              D=5              D=25             D=50             D=100
raw_features + weather     +0.863 (+0.672)  +0.746 (+0.704)  +0.611 (+0.368)  +0.628 (+0.628)
raw_rgb_only + weather     +0.744 (+0.464)  +0.677 (+0.566)  +0.580 (+0.329)  +0.588 (+0.588)
dinov2_vitb14              +0.593 (+0.486)  +0.539 (+0.529)  +0.508 (+0.432)  +0.554 (+0.554)
satlas_s2_swinb_mi_rgb     +0.541 (+0.412)  +0.590 (+0.597)  +0.628 (+0.454)  +0.569 (+0.569)
PERSISTENCE                +0.690 (+0.169)  +0.542 (+0.344)  +0.124 (-0.009)  +0.129 (+0.129)
                                    (2026-08-12 published value in brackets)
```

**Persistence at 5 days goes +0.169 -> +0.690.** Its R-squared was not low
because NDVI moves in five days; it was low because three cloudy frames of 1580
sat in its residual and carried 71% of its sum of squares. Removing eight rows
of 518 moved it by 0.52. The band-matched baseline gains 0.28 at the same cell
and the networks gain 0.02-0.11, so **the screen shrinks every gap the old table
reported, and it shrinks the ones the old headline rested on the most.**

### HEADLINE: `cube_mean`, ridge, cube folds, penalty TUNED, shared base

```
                              D=5      D=25     D=50     D=100    D (with base)
raw_features + weather      +0.871   +0.743   +0.625   +0.639      122
raw_rgb_only + weather      +0.853   +0.727   +0.589   +0.556       80   <- BAND-MATCHED
[NDVI(t), weather] ALONE    +0.773   +0.693   +0.522   +0.406       17   <- no image at all
imagenet_vit_b16_cir        +0.712   +0.670   +0.524   +0.451     4625
satlas_s2_swinb_rgb_cir     +0.698   +0.647   +0.514   +0.370     3089
satlas_s2_swinb_mi_rgb      +0.640   +0.667   +0.626   +0.570     1041  [si_comparable=False]
imagenet_vit_b16            +0.656   +0.622   +0.542   +0.421     4625
satlas_s2_swinb_rgb         +0.616   +0.620   +0.559   +0.500     3089
dinov2_vitb14               +0.622   +0.577   +0.512   +0.550    11537
dinov2_vitb14_cir           +0.586   +0.551   +0.517   +0.586    11537
--------------------------------------------------------------
persistence                 +0.690   +0.542   +0.124   +0.129
proxy climatology           +0.026   +0.125   +0.136   -0.164
observation control         +0.014   +0.056   +0.072   -0.024
permutation                 -0.014   -0.002   -0.007   -0.024
horizon-alone control       -0.006   -0.006   -0.007   -0.023
                              effective n = 115 / 114 / 115 / 94 CUBES
```

**Two numbers and no image beat every frozen network at 5 and 25 days.** The
`[NDVI(t), weather]` row is 17 columns -- current NDVI plus 16 weather
aggregates -- and it reaches +0.773 and +0.693, above all nine encoder views.
The three controls are still where they should be: the observation control never
exceeds +0.072, the horizon-alone control is negative at every horizon, and the
permutation null is at -0.002 to -0.024.

### THE RESULT, PAIRED: NO ENCODER SEPARABLY BEATS THE BAND-MATCHED BASELINE

Paired per-fold difference (encoder minus `raw_rgb_only`, same folds, same
held-out rows) with a delete-one-fold jackknife interval:

```
Delta =   5 d   ALL NINE encoder views are SEPARABLY BELOW it, -0.141 to -0.267
Delta =  25 d   4 of 9 separably below (-0.107 to -0.176); none above
Delta =  50 d   0 of 9 separable in either direction
Delta = 100 d   3 of 9 separably below; none above
```

The only row that is separably ABOVE the band-matched baseline anywhere is
`raw_features` -- **+0.017 `[+0.008, +0.027]`** at 5 days and **+0.036
`[+0.004, +0.068]`** at 50 days -- and it is not a network: it reads all four
bands plus seven NDVI statistics. **On this benchmark, at this sample size, a
frozen EO foundation model does not add anything measurable over 21 percentiles
of the same three bands.** The 2026-08-12 table could only say "within 0.03";
the paired test says the sign.

### rgb vs cir, PAIRED PER TWIN -- the question this run existed to answer

Same weights, same frame selection, same read-out; only the three bands the
network is fed differ, so a twin difference is BAND ACCESS and nothing else.
`cube_mean`, ridge, penalty tuned, shared base:

```
                             cube                 loco                spatial_block
                      diff    verdict       diff    verdict       diff    verdict
imagenet    D=5     +0.056  SEPARABLE     +0.063  SEPARABLE     +0.083  SEPARABLE
            D=25    +0.049  not sep.      +0.054  SEPARABLE     +0.049  not sep.
            D=50    -0.018  not sep.      -0.020  not sep.      -0.057  not sep.
            D=100   +0.031  not sep.      +0.028  not sep.      -0.044  not sep.
satlas SI   D=5     +0.082  SEPARABLE     +0.048  SEPARABLE     +0.126  SEPARABLE
            D=25    +0.027  not sep.      +0.063  SEPARABLE     +0.137  SEPARABLE
            D=50    -0.045  not sep.      -0.056  not sep.      -0.146  not sep.
            D=100   -0.130  not sep.      -0.078  not sep.      -0.076  SEPARABLE
dinov2      D=5     -0.036  not sep.      -0.055  SEPARABLE     -0.053  not sep.
            D=25    -0.026  not sep.      -0.022  not sep.      -0.078  not sep.
            D=50    +0.005  not sep.      +0.024  not sep.      +0.024  not sep.
            D=100   +0.036  not sep.      -0.001  not sep.      +0.095  SEPARABLE
satlas MI   D=5     +0.017  not sep.      +0.016  not sep.      +0.033  not sep.
            D=25    -0.086  not sep.      -0.094  SEPARABLE     -0.125  not sep.
            D=50    -0.084  not sep.      -0.091  SEPARABLE     -0.094  SEPARABLE
            D=100   -0.186  SEPARABLE     -0.100  not sep.      -0.091  not sep.
                                          [satlas MI is si_comparable=False]
```

**16 of 48 twin comparisons are separable; 22 of 48 have the colour-infrared
view ahead at all.** The pattern is a HORIZON pattern, not an encoder one: the
two single-image networks that gain from NIR gain at **5 and 25 days** (+0.05 to
+0.14, separable in 6 of 12 cells) and lose it by 50 days. DINOv2 is the
exception in both directions -- worse with NIR at short horizons, better at 100
days under `spatial_block` (+0.095 `[+0.066, +0.124]`). The multi-image encoder
is made WORSE by NIR at every horizon beyond 5 days.

**So the band-access confound is real but small, and it does not rescue the
foundation-model claim.** Giving the networks the vegetation band moves them by
at most 0.14 and never enough to reach the band-matched baseline: at 5 days
`imagenet_vit_b16_cir` is still -0.141 `[-0.207, -0.076]` below it. The two
readings the confound left open -- "hand-crafted beats learned" vs "NIR beats
RGB" -- resolve toward the first: **NIR helps, and it is not the explanation.**

### What the penalty rule alone did

`nested_cv` minus `fixed_alpha_D`, same rows, same folds, `cube_mean`/cube, at
`feature_base=none` so the two corrections are separated:

```
row                                D      alpha(cv) med    D=5     D=25    D=50    D=100
weather_only                        16           55      -0.007  -0.015  -0.008  -0.113
raw_rgb_only (band-matched)         79            1-10   +0.044  +0.029  +0.008  -0.035
raw_features                       121          100      +0.011  -0.006  +0.013  +0.006
satlas_s2_swinb_mi_rgb            1040          550      -0.014  +0.009  -0.011  +0.000
satlas_s2_swinb_rgb               3088         1000      +0.007  +0.027  +0.038  -0.012
imagenet_vit_b16                  4624         1000      +0.032  +0.041  +0.041  +0.010
dinov2_vitb14                    11536         5500      +0.021  +0.033  +0.003  -0.004
```

Alpha = D spans **79 to 11536** across these rows (80 to 11537 with the shared
base), a 146x range set by the architecture's embedding dimension rather than by
anything about the data. Tuning is worth **+0.03 to +0.04 to the wide network
rows** at 5-50 d and **+0.044 / +0.029** to the narrow band-matched row at 5 and
25 d -- so it helps both sides, and it does not close the gap between them. The
selected penalty is far from alpha = D in both directions: DINOv2 picks 5500
against D = 11536, the band-matched row picks 1-10 against D = 79. **1008 of
12432 tuned folds (8.1%) select at an edge of the grid**, which is on the table
as `n_folds_alpha_at_grid_edge` rather than in a footnote -- P1's lesson, where
the C grid stopped at 1 and selected there in 57% of folds.

**A correction to the preview this phase was specified from, and it matters.**
The expected move for the band-matched row was **+0.483 -> +0.597** at 5 d and
**+0.574 -> +0.693** at 25 d. Measured here, at the matching configuration
(`feature_base=none`, `cube_mean`/cube): **+0.744 -> +0.788** at 5 d and
**+0.677 -> +0.706** at 25 d. The DIRECTION is confirmed -- tuning helps the
band-matched row -- but the size is about a third of the preview (+0.044 and
+0.029 against +0.114 and +0.119), and the levels are 0.26 and 0.10 higher.

The preview was computed on the UNSCREENED row set. The screen moves the same
row from +0.464 to +0.744 on its own, and the two effects are **not additive**:
much of what the tuned penalty was buying on the unscreened table was a better
fit to the three cloud frames, and once those rows are gone there is less for it
to buy. Reported here rather than quietly adopted, because the preview is what
the correction was specified from.

### What the shared base alone did

`+[NDVI(t)]` minus no base, same rows, same folds, same penalty:

```
weather_only            +0.711  +0.492  +0.296  +0.234   <- 16 columns -> 17
satlas_s2_swinb_mi_rgb  +0.113  +0.068  +0.008  -0.000
raw_rgb_only            +0.065  +0.021  +0.002  +0.004
imagenet_vit_b16        +0.013  +0.007  +0.002  -0.000
raw_features            -0.003  +0.002  +0.001  +0.004   <- already had it
dinov2_vitb14           +0.008  +0.005  +0.000  +0.000
```

One column moves `weather_only` by **+0.711** at 5 days and `raw_features` by
**-0.003**, because `raw_features` already held `NDVI_mean(t)`. That gap is
exactly the unearned advantage the base removes: before it, one row in the table
had the strongest single predictor in the problem and no other did. **It is
worth almost nothing to the deep encoders** (+0.000 to +0.013), which is itself
a result -- their embeddings already carry current NDVI, as P2's gate K2 said
they must.

### Skill against persistence reverses at the short horizon

```
skill vs persistence, cube_mean / cube / tuned / shared base
                              D=5      D=25     D=50     D=100
raw_features                +0.583   +0.438   +0.572   +0.585
raw_rgb_only                +0.526   +0.404   +0.531   +0.490
imagenet_vit_b16_cir        +0.071   +0.280   +0.457   +0.370
satlas_s2_swinb_rgb_cir     +0.025   +0.229   +0.445   +0.277
imagenet_vit_b16            -0.110   +0.173   +0.477   +0.335
satlas_s2_swinb_mi_rgb      -0.161   +0.273   +0.572   +0.506
dinov2_vitb14               -0.219   +0.076   +0.443   +0.483
satlas_s2_swinb_rgb         -0.238   +0.169   +0.497   +0.426
dinov2_vitb14_cir           -0.336   +0.020   +0.449   +0.524
```

**Seven of nine encoder views are NEGATIVE against persistence at 5 days.** The
2026-08-12 table reported +0.605 for the best row there; that number was against
a persistence baseline whose error was three cloud frames. With them gone,
"predict today's NDVI" is the better forecast at 5 days for every network, and
only the hand-crafted rows and the two colour-infrared views clear it.

### Nothing else moved

`spatial_block` still does not kill the encoder rows and still destroys the proxy
climatology. The MLP is still unusable at this width. The multi-image encoder is
still the best network at 50 and 100 days and still `si_comparable=False`. The
window boundary still costs 62% of the rows by 100 days, and 21 of 115 cubes
still contribute no 100-day pair.

---

## 2026-08-12: Phase 1.8 (P3) exit test PASSED -- forecasting works, and the hand-crafted baseline is still not beaten

`notebooks/phase1_8_p3_forecast.ipynb`, local CPU, 7 workers, 115 cubes / tile
32UNU / 2018. Archived under `notebooks/runs/`, verbatim stdout at
`docs/runs/2026-08-12_p3_forecast_32UNU_115cubes.txt`. Nothing fine-tuned; no
encoder imported.

```
tests                 496 passed, 5 skipped, 0 failed  (497 after the run: see below)
manifest              REBUILT FRESH: 1580 retained frames, 115 cubes, 2018
                      weather join re-derived from the cubes: 0 rows off their day
horizon axis check    0 of 1653 rows where the two axes agree, 5.0 days per
                      acquisition step, matching the live check on the pairs
run_p3                460 rows x 99 cols, 41.6 min on 7 workers
invariants            all EIGHT PASS
effective n           CUBES: 115 / 114 / 115 / 94 at Delta = 5 / 25 / 50 / 100
```

### Rows retained per horizon -- the window boundary is a result

```
 Delta   rows  cubes  cubes w/ none  % of eligible t  distinct target DOY
     5    518    115              0            38.4%                   29
    25    489    114              1            36.2%                   24
    50    450    115              0            33.3%                   23
   100    196     94             21            14.5%                   15
```

1580 frames, minus 230 that cannot be a `t` for lack of 3 prior retained frames,
leaves 1350 eligible. **At 100 days, 21 cubes contribute no pair at all** and
62% of the 5-day row count is gone: the median cube covers 135 days of retained
frames, so a 100-day horizon has almost nowhere to land. The tolerance is not
doing this -- on a 5-day orbit lattice +/-3 d accepts EXACT matches only, and
+/-2 selects the identical 1653 rows with **0 off-nominal**. Loosening to +/-5
would buy 2936 rows and move 1283 of them off their horizon by 5 days, which at
Delta = 5 is the whole horizon.

Common-masked pixel survival does not collapse at any horizon: 0 of 1653 rows
have zero common pixels, median 0.82-0.88 surviving, minimum 0.23. As in P2 it
is not monotone in the horizon.

### THREE FRAMES SET THIS TABLE'S R-SQUARED, and they are cloud

```
 Delta   rows  <floor  medAE(pers)  RMSE(pers)  SSE from top 1%   top 5%
     5    518       5       0.0221      0.0864            71.2%    85.4%
    25    489       2       0.0392      0.0730            40.0%    55.8%
    50    450       3       0.0565      0.1075            32.5%    49.9%
   100    196       0       0.0530      0.1059             8.4%    33.2%
```

Three frames of 1580 carry a common-masked cube-mean NDVI **below zero in
midsummer**:

```
32UNU_..._441_569_3257_3385_...  DOY 177  clear_frac 0.624  ->  NDVI -0.0404
32UNU_..._441_569_1977_2105_...  DOY 202  clear_frac 0.587  ->  NDVI -0.0336
32UNU_..._441_569_2233_2361_...  DOY 202  clear_frac 0.627  ->  NDVI -0.0279
```

Bare soil is ~0.15 and summer canopy ~0.85. Both filters passed them: the frames
are 59-63% "clear", and the per-pixel mask marked the survivors valid. They
produce five forecast rows with a persistence error of 0.66-0.86 NDVI against a
median of 0.022, and **the worst 1% of rows carry 71% of the persistence sum of
squares at 5 days**. Nothing is dropped -- `sse_share_top1pct` and a MEDIAN
absolute error are on every row instead, because the same three frames are
inside P2's and P4's targets through `p4_ceiling.cube_frame_targets` and a
private filter here would make P3's row set incomparable to theirs.

### HEADLINE: `cube_mean`, ridge, cube folds. Pooled out-of-fold R-squared

```
                              D=5      D=25     D=50     D=100
raw_features + weather      +0.672   +0.704   +0.368   +0.628    <- BASELINE, autoregressive
satlas_s2_swinb_mi_rgb      +0.412   +0.597   +0.454   +0.569    [si_comparable=False]
dinov2_vitb14               +0.486   +0.529   +0.432   +0.554
imagenet_vit_b16            +0.473   +0.566   +0.390   +0.411
satlas_s2_swinb_rgb         +0.442   +0.565   +0.369   +0.512
raw_rgb_only + weather      +0.464   +0.566   +0.329   +0.588    <- BAND-MATCHED baseline
--------------------------------------------------------------
persistence                 +0.169   +0.344   -0.009   +0.129    <- BASELINE
weather only                +0.038   +0.216   +0.190   +0.285    <- BASELINE
proxy climatology           +0.021   +0.130   +0.104   -0.164    <- BASELINE (proxy, NOT Stage B)
observation control         +0.016   +0.056   +0.039   -0.031    <- CONTROL
horizon-alone control       -0.004   -0.007   -0.006   -0.023    <- CONTROL
permutation                 -0.037   -0.035   -0.014   -0.103    <- CONTROL (empirical zero)
                              effective n = 115 / 114 / 115 / 94 CUBES
```

**The three controls are where they should be.** The observation-process control
never exceeds +0.056, the horizon-alone control is negative at every horizon, and
the permutation null sits at -0.014 to -0.103. So the headline is not a calendar
and not cloud retention.

**Every model beats persistence and the proxy climatology at every horizon.**
Best skill against persistence: +0.605 (5 d), +0.549 (25 d), +0.459 (50 d),
+0.573 (100 d). The 5-day case needs care -- persistence's own R-squared is only
+0.169 because those three cloud frames dominate its squared error, and its
median absolute error is 0.0221, the best in the table.

### The result the project has to report: the hand-crafted rows are still not beaten

```
best network minus BAND-MATCHED raw_rgb_only, ridge, cube_mean
                    D=5      D=25     D=50     D=100
cube              +0.022   +0.031   +0.126   -0.018
loco              +0.035   +0.007   +0.112   -0.018
spatial_block     +0.044   -0.009   +0.154   +0.127
```

Seven percentiles of three bands, no NDVI column, no network: within 0.03 of the
best frozen encoder at 5 and 25 days, **ahead of it at 100 days** under the two
primary modes, and behind by 0.11-0.15 only at 50 days. And the full
`raw_features` row -- which legitimately holds `NDVI_mean(t)` here, because the
target is at t+Delta, so this is autoregression and not the K2 leakage case --
**wins outright at 3 of the 4 horizons**.

This is P2's finding again, on a different target: on the delta SIGN probe
`raw_rgb_only` reached +0.695 against DINOv2's +0.606. **"NDVI is forecastable
from a frozen representation plus weather" is established. "Frozen foundation
models are the best way to do it" is not.**

### `spatial_block` does NOT kill P3 -- the second probe in the project to survive it

```
dinov2_vitb14, ridge, cube_mean, pooled R2
                    D=5      D=25     D=50     D=100
cube              +0.486   +0.529   +0.432   +0.554
loco              +0.509   +0.556   +0.445   +0.572
spatial_block     +0.448   +0.491   +0.360   +0.488  [+0.433, +0.543]
```

The penalty is 0.04 to 0.08 and the intervals still exclude zero at 25, 50 and
100 days. P1 and P4 both collapse under the strictest geography holdout; P2's
sign result survived it, and P3's does too. Going in this was UNKNOWN and both
precedents existed -- it was measured, not inherited.

### What `spatial_block` DOES kill is the proxy climatology

```
proxy climatology, cube_mean, pooled R2
                    D=5        D=25      D=50        D=100
cube              +0.021     +0.130    +0.104      -0.164
loco              +0.007     +0.119    +0.098      -0.149
spatial_block     -8.641     -0.011  -1328.396   -2784.151
```

A tile-level day-of-year curve fitted on one geographic cluster does not transfer
to another. `assert_climatology_identifiable` passed -- the thinnest training
fold has 12 distinct target days of year against the curve's 9 parameters -- and
12 for 9 is exactly the regime where a 4-harmonic fit interpolates its training
days and then extrapolates wildly. This bounds the CLIMATOLOGY BASELINE, not the
encoders, and it is a further reason the P4 open item (validate the proxy) is
still open.

### The extreme / dynamic subset reverses the reading, and it is horizon-dependent

Skill against persistence, per severity bin, `cube_mean` / cube / ridge:

```
Delta = 5      extreme_low     low  near_normal    high  extreme_high    (n = 52/104/206/104/52)
raw_features        +0.084  +0.415       +0.868  +0.563        +0.919
dinov2_vitb14       -0.139  -0.185       +0.726  -0.399        +0.809
raw_rgb_only        -0.266  +0.114       +0.832  -0.982        +0.744

Delta = 50     extreme_low     low  near_normal    high  extreme_high    (n = 45/90/180/90/45)
satlas_MI           +0.238  +0.702       +0.728  +0.202        -0.862
dinov2_vitb14       +0.264  +0.611       +0.722  +0.004        -1.220
raw_rgb_only        +0.154  +0.653       +0.600  +0.262        -3.143

Delta = 100    extreme_low     low  near_normal    high  extreme_high    (n = 20/39/78/39/20)
raw_features        +0.739  +0.676       +0.476  -0.225        -0.063
dinov2_vitb14       +0.673  +0.691       +0.422  -0.524        -0.644
```

**The headline lives in `near_normal`.** At 5 and 25 days every network is
NEGATIVE against persistence in `extreme_low` and in `high` -- persistence is
better there -- and only the autoregressive `raw_features` row is positive in
`extreme_low` at 25 days (+0.204). At 50 and 100 days the encoders do become
positive on `extreme_low` (+0.17 to +0.26, then +0.47 to +0.72), but every row
goes negative on `extreme_high`, down to -3.14. **No row in this table beats
persistence on both extreme tails at any horizon.** Reporting only the overall
number would have hidden that completely, which is exactly what the spec
anticipated.

### The MLP is unusable at this width and sample size, and the table says why

```
Delta = 100, cube_mean / cube        D    n_train   D/n     pooled R2
dinov2_vitb14                    11536       158    73.0     -324.24
satlas_s2_swinb_rgb               3088       158    19.5      -25.65
imagenet_vit_b16                  4624       158    29.3      -24.61
satlas_s2_swinb_mi_rgb            1040       158     6.6      -15.58
raw_features + weather             121       158     0.77      -4.01
```

A k=3 DINOv2 context is 11 520 columns against 158 to 416 training rows. The MLP
carries P4's fixed a-priori configuration and an L2 of 1e-3, so it is
essentially unregularised at that width and it diverges. Every row carries
`d_over_n_train`, so the p >> n regime is a measured property of the row rather
than a caveat in prose. **Ridge, whose penalty is alpha = D by rule, is fine at
the same width.** This is P4's "capacity is sample-size dependent" a third time,
in the direction where capacity loses.

### Secondary aggregations

`cell_mean` (16 cells per row, the resolution axis, cube folds) is where the
outlier concentration falls -- `sse_share_top1pct` drops from 65% to 50% at
5 days and to 11-13% at 100 -- and the ordering is the same:

```
cell_mean / cube             D=5      D=25     D=50     D=100
raw_features + weather     +0.778   +0.680   +0.441   +0.513
dinov2_vitb14              +0.593   +0.561   +0.466   +0.572
raw_rgb_only + weather     +0.639   +0.570   +0.366   +0.443
persistence                +0.576   +0.487   +0.107   +0.057
```

At cell level and 5 days **persistence is nearly unbeatable** (+0.576, and
DINOv2's skill against it is only +0.040): the shorter the horizon and the finer
the target, the less there is to add.

### The multi-image encoder is the best NETWORK at 3 of 4 horizons, and that is a lookback result

`satlas_s2_swinb_mi_rgb` leads the single-image encoders at 25, 50 and 100 days
under `cube` and `loco`. It is flagged `si_comparable=False` on every row and it
is not a like-for-like win: its ONE embedding at t already max-pools up to 8
retained frames -- a 0-105 day lookback, median 55 -- while the single-image
encoders get 3 frames spanning perhaps 10-20 days. It is being given more
history, not a better representation of the same history. `context_block`
REFUSES a 3-frame stack for it, because that would double-count the lookback it
already contains.

### Cross-probe: P3's ordering DIVERGES from P2's

P2 (delta sign, 115 cubes, stable under all three fold modes):
`raw_features > dinov2 > satlas_SI > imagenet`.
P3 (`cube_mean`, ridge, single-image encoders only):

```
cube            raw_features > dinov2 > imagenet > satlas_SI
loco            raw_features > dinov2 > satlas_SI > imagenet
spatial_block   raw_features > dinov2 > imagenet > satlas_SI
```

`raw_features` first and `dinov2` second in every mode, as in P2; the two weaker
networks swap. Reported as a cross-probe observation only -- P2 settled that
hypothesis (refuted, `supported=False`) and nothing here re-tests it.

### One assertion was strengthened AFTER the run, and no number moved

Re-checking the eight invariants against the WRITTEN CSV rather than the
in-memory table caught `assert_climatology_rows_labelled` comparing
`climatology_def` to `""` on non-climatology rows. An empty string written to
CSV reads back as NaN, so the assertion passed in memory and failed on the
artefact that table was written to -- the worst direction for an assertion to
fail in, because the CSV is what anyone else reads. Fixed to test the value's
MEANING (NaN or empty = unlabelled), and pinned by
`test_every_table_invariant_survives_a_CSV_ROUND_TRIP`. **497 passed, 5 skipped,
0 failed** afterwards; the archived notebook records 496 because that test did
not exist when it ran. No number in the table changed.

## 2026-08-11: P2 at 115 cubes -- gate K2 becomes a RANKING, sign holds, magnitude was never there, and the MI exclusion is RETRACTED

`scripts/scale_p2.py` against the Phase 1.7 cache (115 cubes x 5 encoders, built
on Colab). Identical probe, identical fold machinery, 5.75x the cubes. Nothing
re-encoded here; nothing fine-tuned anywhere.

```
cache audit           575/575 (cube, encoder) pairs, 115/115 masks, no holes
reproduction check    100 shared pairs vs the 20-cube cache:
                      kept_idx/timestamps/clear_frac BIT-IDENTICAL,
                      worst pooled diff 1.52e-4 (raw_features exactly 0)
gap axis check        1465 pairs, 0 where the two axes agree
run_p2                600 rows x 64 cols, 100.3 min on 7 workers
invariants            all six PASS
manifest              1580 retained frames (6.0x), 1465 pairs (6.0x), n = 115 CUBES
```

The reproduction check is the load-bearing one and its SHAPE is the evidence:
`raw_features` is pure numpy and came back **exactly 0**; the four networks
differ by 5e-5 to 1.5e-4. That is GPU-vs-CPU float32 jitter and nothing else.
The two caches are the same experiment.

### GATE K2: 3 of 4 networks now SEPARABLY beat the hand-crafted baseline

`cube_mean / grid_cell / cube`, cube-clustered, paired per-fold difference
against `raw_features`:

```
                          R2       CI width      paired CI on (enc - raw)   verdict
satlas_s2_swinb_rgb     +0.713   0.126 (0.166)   [+0.094, +0.156]  SEPARABLE  passed
imagenet_vit_b16        +0.687   0.036 (0.394)   [+0.036, +0.162]  SEPARABLE  passed
dinov2_vitb14           +0.649   0.096 (0.582)   [+0.021, +0.101]  SEPARABLE  passed
raw_features            +0.588   0.133 (0.542)   --                           baseline
satlas_s2_swinb_mi_rgb  +0.519   0.145 (0.822)   [-0.192, +0.053]  not sep.   lossy*
                                                          effective n = 115 CUBES
                              (20-cube CI width in brackets)
```

**At 20 cubes K2 was a floor check; at 115 it is a ranking.** Every interval
collapsed -- DINOv2's by 6x, ImageNet's by 11x -- and three encoders moved from
"not separable from a table of percentiles" to separably above it. This is the
claim the 20-cube run could not make and explicitly declined to make.

**The P3 exclusion is RETRACTED.** `satlas_s2_swinb_mi_rgb` went from
-0.363 [-0.592, -0.134] (lossy AND separable, excluded) to -0.069
[-0.192, +0.053] (not separable), and it PASSES the band-matched verdict.
**No encoder is excluded from P3.** The 20-cube exclusion was a small-sample
artefact, exactly as `k2_separable` was built to expose -- it is worth noting
that the column did its job in the other direction too: it stopped DINOv2 being
excluded at 20 cubes on a -0.006 point estimate, and DINOv2 is now +0.061 and
separably ABOVE the baseline.

### SIGN: confirmed, and it does not move

`cube_mean / sign / pooled / linear`, margin over the gap-length control:

```
                          cube     loco   spatial_block      (was, cube)
raw_features             +0.720   +0.696   +0.741             +0.918
raw_features/rgb_only    +0.630   +0.600   +0.653             +0.842
dinov2_vitb14            +0.541   +0.514   +0.547             +0.653
satlas_s2_swinb_rgb      +0.522   +0.505   +0.527             +0.598
imagenet_vit_b16         +0.443   +0.434   +0.462             +0.575
satlas_s2_swinb_mi_rgb   +0.399   +0.377   +0.367             +0.281   si_comparable=False
GAP CONTROL              +0.065   ...      ...                -0.118
```

Margins are within 0.05 of each other across all three fold modes. `spatial_block`
does not kill it -- unlike P1 and P4. This is the strongest result in the phase.

### MAGNITUDE: the 20-cube finding was WRONG, and the truth is worse

The 20-cube run reported "the gap-length control at +0.209 beats every encoder".
**That control was itself a small-sample artefact.** At 115 cubes it falls to
**+0.063 [-0.004, +0.130]**, an interval spanning zero, and to +0.018 under
`spatial_block`.

So 6 of 7 rows now "beat" it -- but the margins are noise:

```
                          cube     loco   spatial_block
dinov2_vitb14            +0.043   +0.065   +0.026     <- only consistently positive
raw_features             +0.033   +0.043   +0.072
raw_features/rgb_only    +0.029   +0.046   +0.018
satlas_s2_swinb_mi_rgb   +0.055   -0.001   +0.037     sign flips
satlas_s2_swinb_rgb      +0.001   +0.035   -0.005     sign flips
imagenet_vit_b16         -0.005   +0.013   +0.052     sign flips
```

Three of four encoders flip sign across fold modes, and every absolute
correlation is between +0.057 and +0.118. **The corrected finding is not "the
calendar wins" but "nobody recovers magnitude at all"** -- the 20-cube version
overstated the control and understated nothing. Direction is in these
representations; rate is not, and now we know that is not a control artefact.

### STRUCTURAL HYPOTHESIS: now determinable, and REFUTED

At 20 cubes DINOv2 and Satlas SI swapped rank between fold modes and
`structural_hypothesis` returned `supported=None`. At 115 the ordering is
**stable across all three modes**:

```
cube           raw_features > dinov2_vitb14 > satlas_s2_swinb_rgb > imagenet_vit_b16
loco           raw_features > dinov2_vitb14 > satlas_s2_swinb_rgb > imagenet_vit_b16
spatial_block  raw_features > dinov2_vitb14 > satlas_s2_swinb_rgb > imagenet_vit_b16
```

`order_stable_across_fold_modes=True`, `supported=False`. research_plan_v3
Section 3/P2 expected augmentation-invariance training (DINOv2) to discard
state-change MORE than reconstruction-style training (Satlas). It does the
opposite, consistently, at every fold mode. Still only 2 EO-relevant SI points,
so this refutes the stated direction rather than establishing a mechanism.

### What did NOT change

`raw_features` still leads every delta table, because it still contains
`NDVI_mean..NDVI_p90`. Read `raw_rgb_only` (+0.630 sign margin) as the fair
comparison; the networks clear it on K2 but not on the delta sign probe.

## 2026-08-11: Phase 1.6 (P2) exit test PASSED -- 600 rows, K2 cleared by all five, and the delta probe splits in two

> **SUPERSEDED IN PART by the 115-cube run above (2026-08-11).** Two claims
> below did not survive the scale-up and must not be quoted from here:
> (1) *"the gap-length control at +0.209 beats every encoder on magnitude"* --
> the control was itself a small-sample artefact and falls to +0.063 with an
> interval spanning zero; the corrected finding is that NOBODY recovers
> magnitude. (2) *"`satlas_s2_swinb_mi_rgb` is excluded from P3"* -- it is not
> separable from the baseline at 115 cubes and the exclusion is retracted.
> The sign result, the common-masking behaviour and the gap-axis finding all
> replicate. Kept in full because the comparison is the evidence that 20 cubes
> could not support these claims.

`probes/p2_deltas.py`. Two questions: (a) gate K2, can a linear head read
CURRENT NDVI out of a frozen embedding at all; (b) do embedding CHANGES track
real NDVI change. Nothing fine-tuned, no weight loaded, no embedding recomputed
-- this phase reads the Phase 1.2 `.npz` cache (embeddings + the per-pixel masks
cached in 1.2b), the cubes for NDVI through `data.ndvi.ndvi`, and the manifest.
CPU only.

```
Step 5   tests                 0 failed, 5 skipped
Step 6   manifest              REBUILT from the cubes; E-OBS join re-verified, max abs diff 0
Step 8   gap axis check        0/244 pairs where the two axes agree (5.0 d per acquisition step)
Step 9   common-masking        244/244 pairs survive; median 88.8% of 16384 px
Step 12  run_p2                600 rows x 61 cols, 8.6 min
Step 13  invariants            all six PASS
Step 16  artefacts             600-row CSV + survival table under data/phase1_6/results/
notebook clean end to end (archived under notebooks/runs/)
```

### THE TWO AXES, MEASURED ON 244 REAL PAIRS

Three quantities could be called "the gap between consecutive retained frames",
and they are not close:

```
gap_days        daily_axis_index      min  5   median 10   max 35
gap_acq_steps   original_axis_index   min  1   median  2   max  6     <- the JOIN KEY
frames between  array position        always 1
```

They disagree on **244 of 244 pairs**, by a factor of **5.0** at the median --
one Sentinel-2 orbit lattice. Every distinct gap is a multiple of 5 days
(5, 10, 15, 20, 25, 30, 35). `assert_gap_axes_disagree` asserts this on the real
data at run time, before anything is fitted.

### COMMON-MASKED PIXEL SURVIVAL BY GAP LENGTH -- no collapse, and NOT monotone

Pixels valid in BOTH frames, from the 1.2b mask cache. 16384 px per frame:

```
gap (d)      5      10      15      20      25      30      35
pairs      106      67      31      15      14      10       1
px median 14668   13747   12821   14749   15612   16142   13269
frac med   0.895   0.839   0.783   0.900   0.953   0.985   0.810
frac min   0.272   0.413   0.370   0.429   0.505   0.868   0.810
cells min     12      13      14      13      13      16      16   (of 16)
zero-common    0       0       0       0       0       0       0
```

**Nothing collapses**: 0 of 244 pairs lose every shared pixel, and the worst
single pair still keeps 27.2%. Survival is *worse* at 15 days than at 30,
because a long gap exists **because** the frames between it were dropped for
cloud, which leaves the two surviving endpoints unusually clear. Do not model
pixel survival as a decreasing function of horizon on this benchmark.

### GATE K2 -- all five encoders, cube-clustered, at cube_mean / grid_cell / cube

```
                          R2      95% CI            vs raw   PAIRED CI on vs-raw   verdict(band)
satlas_s2_swinb_rgb     +0.545  [+0.462, +0.628]    +0.105   [-0.118, +0.327]      passed
imagenet_vit_b16        +0.521  [+0.324, +0.718]    +0.081   [-0.061, +0.223]      passed
raw_features            +0.440  [+0.169, +0.711]     0.000   --                    baseline
dinov2_vitb14           +0.435  [+0.144, +0.726]    -0.006   [-0.142, +0.131]      passed
satlas_s2_swinb_mi_rgb  +0.077  [-0.334, +0.489]    -0.363   [-0.592, -0.134]      audited: lossy
                                                            effective n = 20 CUBES
raw_rgb_only (band-matched floor)  +0.417  [+0.163, +0.670]
retention CONTROL                  -0.129  [-0.356, +0.099]
```

**No single-image encoder is excluded from P3.** The only separably-lossy
verdict is the multi-image control, which was never in the single-image column.

**And the paired interval says something the point estimates hide: at 20 cubes
K2 cannot separate ANY single-image encoder from the hand-crafted baseline.**
Every paired CI on (encoder - raw_features) spans zero except the MI control's.
Satlas SI's nominal +0.105 lead carries `[-0.118, +0.327]`. The gate is a FLOOR
CHECK -- "no encoder is catastrophically lossy" -- and that is all it can be at
this n. It is not a ranking, and the +0.545 vs +0.440 gap must not be reported
as one.

**Two verdicts are recorded, because `raw_features` contains the target.** It
carries `NDVI_mean`..`NDVI_p90` as columns, so at the MATCHED level it IS the
answer: `cell_mean` from grid features **R2 +1.0000**, `cube_p90` from pooled
**+1.0000**, `cube_mean` from pooled **+0.9998**. The primary configuration
above is *not* matched (a cube mean from one cell's statistics is not the
identity) and is a fair gate; the secondary views are not. Read
`k2_verdict_band_matched`.

**A verdict is not a rejection.** DINOv2 sits 0.006 below the baseline with
marginal intervals 0.6 wide. `k2_separable` is the PAIRED per-fold difference,
and it spans zero -- "did not beat the baseline", not "measurably worse".

### THE DELTA PROBE SPLITS IN TWO: SIGN YES, MAGNITUDE NO

Primary cell, `cube_mean` / pooled / linear read-out / cube folds, Spearman
against the common-masked NDVI change:

```
target = SIGN                    rho      95% CI            margin over control
raw_features                   +0.801  [+0.753, +0.849]        +0.918
raw_features/raw_rgb_only      +0.724  [+0.663, +0.785]        +0.842
dinov2_vitb14                  +0.536  [+0.397, +0.674]        +0.653
satlas_s2_swinb_rgb            +0.481  [+0.404, +0.557]        +0.598
imagenet_vit_b16               +0.458  [+0.318, +0.597]        +0.575
satlas_s2_swinb_mi_rgb         +0.163  [+0.070, +0.257]        +0.281   si_comparable=False
GAP-LENGTH CONTROL             -0.118  [-0.370, +0.135]         0.000

target = MAGNITUDE               rho      95% CI            margin over control
GAP-LENGTH CONTROL             +0.209  [+0.066, +0.351]         0.000   <- WINS
dinov2_vitb14                  +0.174  [-0.005, +0.353]        -0.035
satlas_s2_swinb_mi_rgb         +0.081  [-0.082, +0.244]        -0.128   si_comparable=False
raw_features                   +0.078  [-0.052, +0.209]        -0.130
satlas_s2_swinb_rgb            +0.065  [-0.034, +0.165]        -0.143
raw_features/raw_rgb_only      +0.014  [-0.188, +0.217]        -0.194
imagenet_vit_b16               -0.029  [-0.168, +0.110]        -0.237
                                                          effective n = 20 CUBES
```

**Direction is available in these representations; rate is not.** Every encoder
clears the gap-length control on sign by a wide margin. On magnitude **the
control beats every one of them** -- including the raw-pixel baseline. The
`norm` read-out is the one partial exception (`raw_features` ||dE|| reaches
+0.455 on magnitude, margin +0.246), which says the *size* of the pixel-statistic
change carries rate information that a linear read-out of the difference vector
does not extract.

Holds at the cell level too (`cell_mean` / grid_cell / linear / cube, sign):
raw_features +0.766, satlas SI +0.4596, dinov2 +0.4576, imagenet +0.417, MI
+0.250, control -0.027.

### THE GAP-LENGTH CONTROL, ALL FOUR VALUES

```
                         sign                        magnitude
cube            -0.118 [-0.370, +0.135]      +0.209 [+0.066, +0.351]
loco            +0.039 [-0.098, +0.176]      +0.260 [+0.081, +0.438]
spatial_block   +0.009 [-0.226, +0.244]      +0.168 [-0.280, +0.616]
```

### THE STRUCTURAL HUNCH IS NOT DETERMINABLE

research_plan_v3 §3/P2 expected augmentation-invariance training (DINOv2) to
discard state-change more than reconstruction-style training (Satlas). The
ordering **flips with the fold mode**:

```
cube            dinov2 +0.536  >  satlas +0.481     hunch fails
loco            satlas +0.531  >  dinov2 +0.481     hunch holds
spatial_block   satlas +0.489  >  dinov2 +0.473     hunch holds
```

**This is the same pair P1 could not rank** (0.387 vs 0.386, a rank that moved
with a scipy version). `structural_hypothesis` returns `supported=None`.

### ROBUSTNESS: spatial_block does NOT kill the sign result

Unlike P1 and P4, where `spatial_block` collapsed everything, the sign margins
survive it (satlas +0.480, dinov2 +0.464, imagenet +0.358) -- the intervals
widen, the ordering of networks-vs-control does not change. The magnitude
control's interval, by contrast, blows out to [-0.280, +0.616].

## 2026-08-10: Phase 1.5 (P4) exit test PASSED -- 270 rows, Stage B DEFERRED, and the ceiling is not measurable at 20 cubes

`probes/p4_ceiling.py`. What fraction of the post-climatology NDVI anomaly is
explainable from weather alone? Stage A only: the 20 cubes are single-year, so
the leave-target-year-out climatology is not computable and **Stage B is
deferred, not substituted**. Nothing fine-tuned, no embedding opened, no weight
loaded -- this phase reads the cubes and nothing else.

```
Step 5   tests                 382 passed, 5 skipped
Step 6   E-OBS join            VERIFIED against the cubes, max abs diff 0
Step 10  run_p4                270 rows x 95 cols, 3.0 min on 7 workers
Step 11  invariants            COMPLETE; Stage B DEFERRED, explicitly
Step 13  artefacts             270-row CSV under data/phase1_5/results/
notebook clean end to end, fresh runtime, 4.2 min (archived under notebooks/runs/)
```

### THE HEADLINE: the fold-clustered CI includes zero everywhere

Primary cell, `cube_mean / cube / linear / weather_full8`, `r2_vs_climatology`
(= 1 - SSE/SSE_zero; the climatology predicts anomaly zero, so this is skill
against the climatology itself):

```
                          r2_vs_clim   fold-clustered 95% CI   margin   CRPS skill
weather (8 vars, D=64)      +0.066     [-0.159, +0.291]        +0.038     +0.049
weather + observation       +0.095     [-0.187, +0.376]        +0.067     +0.064
OBSERVATION CONTROL         +0.028     [-0.073, +0.129]         0.000     +0.015
day-of-year sanity          +0.007     [-0.005, +0.020]        -0.020     +0.008
PERMUTATION (empirical 0)   -0.120     [-0.382, +0.142]        -0.147     -0.044
                                                          effective n = 20 CUBES
```

**Weather explains about 6.6% of the within-season proxy anomaly, of which 2.8
points were already available from cloud retention alone.** The margin over the
observation-process control is **+0.038**. And the interval spans zero -- as it
does for every one of the 54 weather rows in the table. At 20 cubes, one tile
and one year, the effective sample size is **20 independent weather
realisations**, and a ceiling of this size is not separable from zero at that n.

That is the result. It is not a null about weather attributability; it is a
statement that this subset cannot measure it, and it is the sharpest argument
yet for the seasonal download.

**33 of 54 weather rows sit AT OR BELOW the observation control.** Where they
sit is the finding:

```
margin over the observation control, weather_full8
                        linear     hgb      mlp
cube_mean  cube         +0.038   +0.162    -2.46
cube_mean  loco         -0.032   -0.191    -5.90
cube_mean  spatial_bloc +0.053   -0.178    -7.61
cube_p90   cube         +0.039   +0.497    -5.70
cell_mean  cube         +0.046   +0.040    -0.93
cell_mean  loco         +0.021   +0.025    -0.88
cell_mean  spatial_bloc -0.002   -0.150    -2.27
```

### CAPACITY HURTS. The MLP loses to its own permutation null

The three estimators are not close. `linear` (ridge, alpha = D) is the only one
that is positive on the primary target; `hgb` is around zero except on
`cube_p90`; the small MLP is catastrophic (-2.6 to -17.9) and is **worse than
the permutation control** in most cells. With 211 training rows against 64
features and 20 independent realisations, capacity buys overfitting. Reported
in full because the spec asks for all three, and because "the flexible models do
worse than the linear one" is itself the sample-size result.

Permutation control, mean `r2_vs_climatology` by estimator:

```
             cell_mean   cube_mean   cube_p90
linear         -0.071      -0.082     -0.067
hgb            -0.139      -0.625     -0.314
mlp            -1.751      -5.468    -14.754
```

**Negative, not zero.** A flexible estimator on shuffled features is PENALISED
rather than neutral: it fits noise on train and pays for it on test. The
empirical zero of this pipeline is therefore a ceiling, not a centre, and what
the control establishes is that the pipeline never manufactures positive skill
from an association that has been destroyed.

### THE DAY-OF-YEAR SANITY CONTROL, and what it revealed

```
r2_vs_climatology of day-of-year ALONE
                        linear     hgb      mlp
cell_mean  cube          0.001    0.069    0.019
cell_mean  loco          0.002    0.043    0.030
cell_mean  spatial_bloc  0.019    0.046   -0.117
cube_mean  cube          0.007    0.168    0.111
cube_mean  loco         -0.008   -0.121   -0.076
cube_mean  spatial_bloc  0.002   -0.028   -0.392
cube_p90   cube         -0.007    0.421    0.215
cube_p90   loco         -0.013    0.387    0.184
cube_p90   spatial_bloc -0.007    0.265   -0.859
```

**Linear: worst |r2| = 0.019.** The detrend removed the smooth seasonal cycle,
which is what the control is for, and Stage A's anomaly is a real anomaly.

**Flexible: up to +0.421.** Not a failed detrend -- see the collinearity
measurement below. A boosted tree given only day-of-year fits a per-date mean
over 36 dates, which on this subset is most of a weather model in a different
basis.

### The control CHOSE the climatology's smoothness, and the choice is conservative

Harmonic order was not picked by taste. Measured (cube k=5, linear, full8):

```
H    cube_mean            cube_p90             cell_mean
     weather   DOY        weather   DOY        weather   DOY
2     +0.092  +0.040       +0.132  +0.106       +0.077  +0.010
3     +0.069  +0.009       +0.117  +0.055       +0.067  +0.002
4     +0.066  +0.007       +0.066  -0.007       +0.067  +0.001
5     +0.054  +0.000       +0.068  +0.001       +0.067  -0.001
```

At H=2 the "anomaly" still carried enough seasonal cycle for day-of-year alone
to explain 4.0% of it, and **10.6% on the 90th percentile** -- and the weather
number was inflated to match. Most of what looked like weather attributability
at low order was leftover phenology. **H=4 is the lowest order at which the
control lands at zero for all three targets at once**, and raising the order
LOWERS the reported weather number everywhere, so selecting against this control
cannot talk the headline up. `CLIMATOLOGY_HARMONICS = 4` (resolves a ~91-day
period; absorbs 15-34% of raw variance), `DOY_CONTROL_HARMONICS = 6`.

At 8 control harmonics (17 columns against 264 rows) the control turns positive
again on the cube-level targets (+0.045, +0.048). Recorded, not used: a control
that overfits stops being a floor.

### 36 DAYS OF YEAR, ONE ORBIT LATTICE -- why the flexible DOY control is not a sanity check

Measured before anything was fitted:

```
264 rows land on 36 DISTINCT days of year over a 235-day span
every row satisfies doy % 5 == 2   -- ONE Sentinel-2 orbit lattice
261/264 rows share their date with another cube; one date carries all 20 cubes
```

So day-of-year is close to a **36-level categorical variable**. Within a date,
the across-cube spread as a fraction of total spread:

```
eobs_pp 0.093   eobs_qq 0.091   eobs_tg 0.142   eobs_tx 0.161
eobs_tn 0.189   eobs_hu 0.274   eobs_fg 0.441   eobs_rr 0.766
windowed features (D=64): min 0.076  median 0.375  max 0.920
```

A 150 km tile is one air mass, so temperature, pressure and radiation are nearly
determined by the date; **precipitation is convective and local, and carries 77%
of its variance across cubes.** Roughly 63% of a typical windowed feature is
recoverable from the date alone -- which is why a flexible day-of-year model
recovers so much, and why only the LINEAR row is the sanity check the spec
asks for. Precipitation is doing most of the cross-cube work a weather model can
do here.

### Weather windows are on the DAILY axis, and they are truncated

Available history per frame: **min 4, median 64 days**; 47% of rows have fewer
than 60. `WINDOW_SPECS = ((7,0), (14,0), (30,0), (30,30))` was chosen against
that measurement rather than by convention -- a 90-day window would be mostly
clipping. Truncation is clipping at the start of the cube, never filling;
aggregates are per-day RATES so it costs precision and not scale; and
days-available is deliberately NOT a feature, because it correlates with
position-in-cube and therefore with time of year.

16 aggregations x 4 windows = **64 features** for the 8-variable set, 44 for the
5-variable EO-WM subset.

### The 5-variable EO-WM subset is WORSE than the full 8

```
margin over the observation control, linear
                        weather_eowm5   weather_full8
cube_mean  cube             -0.044          +0.038
cell_mean  cube             -0.032          +0.046
cell_mean  loco             -0.009          +0.021
```

EO-WM's meso channels are precipitation, pressure and mean/min/max temperature
(personal communication, 2026-08-07). Dropping wind, humidity and radiation
costs the whole margin on the primary cells -- and note from the collinearity
table that `fg` and `hu` are the second and third most cross-cube-variable
inputs after `rr`. Reported for commensurability, not as our number.

### Severity bins and strata

Bin edges, printed before any fit, from the reference anomaly (a REPORTING axis
only -- the modelled anomalies are re-derived inside every fold):

```
cube_mean  -0.1006 / -0.0329 / +0.0398 / +0.1013   counts 27/52/106/52/27
cell_mean  -0.1618 / -0.0547 / +0.0719 / +0.1456   counts 420/840/1675/840/420
```

Pooled out-of-fold R-squared, `cell_mean / weather / linear / full8`:

```
                extreme_low   low   near_normal   high   extreme_high
cube               +0.225    +0.090    -1.620    +0.036     +0.117
loco               +0.208    +0.115    -1.442    +0.047     +0.151
spatial_block      +0.058    +0.069    -0.466    -0.097     +0.007
```

**Weather does better at the extremes than in the middle**, which is the
expected shape -- a near-normal anomaly is mostly noise, and dividing by a small
SSE_zero makes that bin's R-squared explode negative. Per stratum, over
`REPLICATION_STRATA` only:

```
                cropland   tree_cover   grassland      (n = 1765 / 1261 / 1087 rows)
cube             +0.129      +0.101      -0.157
loco             +0.130      +0.115      -0.082
spatial_block    -0.032      -0.028      +0.026
```

Cropland and tree cover replicate under cube and LOCO; **grassland does not**,
and goes the other way. Grassland NDVI in the Alpine foreland is dominated by
MOWING, which is a management decision and not a weather response -- the one
stratum where a weather-only ceiling should be expected to fail is the one that
fails. Under `spatial_block` nothing replicates.

### Cost

3.0 min on 7 CPU workers for 270 rows = 2700 outer folds. No GPU, and none would
help. Peak memory under 2 GB.

## 2026-08-11: P4 at 115 cubes -- the first measured ceiling, and capacity stops hurting

`scripts/scale_p4.py --tile 32UNU --n 115`. 5.75x the cubes (192 listed, 115
after the 64 px non-overlap rule), 1580 manifest rows, 270 rows in 37 min. Cubes
live under `data/scaled_32UNU/` and NOT `data/raw`, which stays the 20-cube set
Phase 1.2's embedding cache is keyed to.

**17 of 54 weather rows now exclude zero, against 0 of 54 at 20 cubes.** Median
fold-clustered CI width, weather rows: linear 0.236 -> 0.166, hgb 0.616 ->
0.419, mlp 7.35 -> 1.61. The proxy ceiling is cube-limited and cubes are cheap,
exactly as the 30TVN scaling predicted.

### THE HEADLINE: `cell_mean` under HGB, and it survives both controls

```
target/mode        weather   [CI]              vs OBSERVATION   vs DOY control
cell_mean cube     +0.130  [+0.063, +0.197]        +0.137           +0.094
cell_mean loco     +0.085  [+0.007, +0.162]        +0.109           +0.086
```

This is **the first cell in this project to clear zero and both controls at
once**. The day-of-year control on those rows is +0.036 and -0.002, so the
signal is not residual phenology, and the margin over the observation-process
control is +0.137 / +0.109, so it is not cloud retention either.

### The estimator ordering REVERSED, and that is a sample-size result

At 20 cubes: "capacity hurts, only the linear model is positive, HGB is around
zero". At 115 cubes HGB is the strongest estimator everywhere and linear is the
weak one (+0.078 on the primary cell, interval still spanning zero). Boosted
trees needed the data; 264 rows was not enough for them and 1580 is. The earlier
conclusion was true of the earlier sample size and is now superseded rather than
contradicted.

### A caveat that only the DOY control could have caught

HGB's largest raw number is `cube_p90 / cube` at **+0.320**, which looks like the
headline until it is read against its own day-of-year control at **+0.256**:

```
HGB weather MINUS HGB day-of-year control, weather_full8
cube_mean cube          +0.222 - 0.092 = +0.130
cube_mean loco          -0.068 - 0.038 = -0.106
cube_p90  cube          +0.320 - 0.256 = +0.063   <- most of it is the calendar
cube_p90  loco          +0.106 - 0.012 = +0.094
cube_p90  spatial_block -0.279 - 0.091 = -0.370
cell_mean cube          +0.130 - 0.036 = +0.094
cell_mean loco          +0.085 + 0.002 = +0.086
```

**Four fifths of HGB's best-looking number is day-of-year.** The 36-date orbit
lattice measured at 20 cubes is still there at 115 (more cubes add rows per
date, not new dates), so a flexible learner can still fit a per-date mean.
`cell_mean` is the target where it cannot: 16 cells per frame share a date, so
per-date memorisation buys much less. **Cite `cell_mean`, not `cube_p90`.**

The LINEAR day-of-year control is clean everywhere, worst |r2| **0.011** across
all nine target x mode cells (0.019 at 20 cubes). The H=4 harmonic order chosen
against that control at 20 cubes holds at 5.75x the data, unchanged.

### What did NOT change

* **`spatial_block` still kills it.** Every estimator goes negative or to zero
  under the strictest geography holdout, as in P1. Nothing here survives it.
* **The MLP is still catastrophic** (-0.98 to -5.35) and now excludes zero on the
  WRONG side in 8 rows -- confidently wrong rather than merely noisy.
* **The observation control did not systematically weaken.** It fell in the
  primary cell (+0.028 -> +0.002) but across all cells the mean is flat
  (-0.087 -> -0.092, max +0.031 -> +0.035). The retention confound is not a
  small-sample artefact; it is real and it stays.
* **The permutation control stays negative everywhere** (max -0.007), and is
  much tighter than at 20 cubes ([-0.042, -0.014] on the primary cell). The
  pipeline still never invents skill from a destroyed association.
* **Stage B is still deferred**, correctly: 115 cubes, all 2018.

### The claim this supports

"On tile 32UNU, weather explains **0.09 to 0.13** of the within-season
post-climatology NDVI anomaly at the grid-cell level, above both a
cloud-retention control and a day-of-year control, under cube- and
leave-one-cube-out holdout -- and nothing survives a strict spatial-block
holdout." That is a bounded, controlled, replicated number where four days ago
there was an interval spanning zero.

It remains a PROXY-climatology ceiling, not H1. See the year-limit entry below
for why H1 is not obtainable on this benchmark at all.

## 2026-08-10: H1's precision is bounded by the number of YEARS, not cubes -- and GreenEarthNet has four

`scripts/validate_proxy_climatology.py --tile 30TVN --n 87`. Stage A (proxy)
and Stage B (real leave-target-year-out) over the SAME 87 cubes, 7574 rows,
17 min. This was run to ask "is the proxy any good"; it answered a different and
more important question first.

```
                      folds  eff n   weather r2_vs_clim        CI width  per-fold sd   MDE
PROXY   (cube folds)      5   87 cu  +0.192 [+0.139, +0.245]     0.105      0.042      0.053
REAL    (crossed)         4   87 cu  -0.136 [-0.852, +0.581]     1.433      0.450      0.631
```

**Stage B's interval did not tighten with 3.5x the cubes.** The pilot at 25
cubes gave [-1.024, +0.587], width 1.611; 87 cubes gives width 1.433 -- 11%
narrower for 3.5x the data. The reason is structural and is one line of
`probes/cv.py`:

```python
k = _clamp_k(int(k), min(uniq_years.size, np.unique(cubes).size), ...)
```

**In `crossed` mode the fold count is clamped by the number of YEARS.**
GreenEarthNet's seasonal cubes span 2017-2020, so k = 4 and no cube count
changes that. Each fold holds out one year, and the per-fold scores ARE the
interannual spread:

```
Stage B per-fold r2_vs_clim:  -0.385  -0.642  +0.192  +0.292
```

Those are four draws of "how well does weather explain the anomaly in a year the
model has never seen", and they disagree in SIGN. A t interval on 4 replicates
(3 df, t = 3.18) around that spread is irreducibly wide. The power calculation
makes it concrete: detecting the observed -0.136 at 80% power would need **~87
folds, and the dataset can supply 4.** Off by more than an order of magnitude,
structurally, on the axis that cannot be bought with more downloads.

**So H1 -- the leave-target-year-out ceiling, evaluated under the only fold mode
that agrees with its own climatology -- is not precisely estimable on
GreenEarthNet at all.** Not on 32UNU (no seasonal coverage), and not on a
seasonal tile either (4 years). That is a property of the benchmark, not of this
probe, and it is worth stating plainly rather than discovering at review.

**The proxy, by contrast, scales.** Stage A at 87 cubes has a CI width of 0.105
and a per-fold sd of 0.042 -- it is well-powered, and its MDE (0.053) is a
quarter of the effect it measured. Compare 32UNU's Stage A at 20 cubes: width
0.450. The proxy's uncertainty is cube-limited, and cubes are cheap.

**What this does NOT establish.** The proxy and the real climatology give
different point estimates (+0.192 vs -0.136, gap 0.328) and their intervals
overlap only because Stage B's is enormous. **That is the absence of evidence,
not evidence of agreement**, and the script says so in its own output rather
than letting the overlap be read as validation. The proxy remains unvalidated;
what changed is that we now know validating it against this benchmark is not
achievable by scaling.

Two secondary observations, both confounded and neither load-bearing. On 30TVN
the observation control sits at -0.000, so the full +0.192 is weather rather
than retention -- unlike 32UNU, where +0.028 of +0.066 was retention. And the
semi-arid tile's weather-attributability is roughly 3x the Alpine foreland's,
which is the direction water-limitation predicts. Both compare across different
tiles, cube counts AND feature sets (6 variables vs 8), so they are hypotheses
for a later controlled run, not results.

## 2026-08-10: `year` was the filename's window-start, and it had blocked Stage B on every tile

Found the moment a real seasonal cube was first run through the manifest.

`manifest_rows` parsed `year` from the cube FILENAME's window-start field and
stamped it on every row of that cube. On tile 32UNU that is indistinguishable
from the truth -- one 2018 window per cube, so the id-derived year and each
frame's calendar year always agree -- and it survived four phases. On a real
seasonal cube (one file spanning 2017-2020) they disagree on **1666 of 2092
rows**.

`probes.cv._row_years` refused it instantly and named the fix in its own error
message. `tests/test_cv_folds.py::test_year_mode_rejects_a_lying_year_column`
had tested for exactly this since Phase 1.3, with a docstring reading
"build_manifest CURRENTLY derives year from the cube id". The guard, the test
and the written diagnosis were all correct and none had ever executed, because
nothing in the repo had handed them a multi-year cube.

**The consequence was larger than the bug.** `crossed` is the only fold mode
that agrees with a leave-target-year-out climatology, so **P4's Stage B could
not have run on any tile until this was fixed** -- it was written, gated on a
detector that correctly reported `multi_year=True`, and had simply never met
real seasonal data. It ran for the first time on 2026-08-10.

Fixed: `year` now comes from each frame's own timestamp.
`tests/test_encoders.py::test_manifest_year_is_per_row_not_the_cube_id_window_start`
builds a synthetic cube whose frames straddle 31 December while its filename
says 2018, and was verified to FAIL against the previous code before being kept.

## 2026-08-10: tile 32UNU has no seasonal coverage; H1 is not computable there

Read from the bucket rather than assumed:

```
seasonal split, 15 tiles:  29SQC 29TPF 29UMV 30TVK 30TVN 31TCF 31TFK 31UCS
                           31UEQ 31UGQ 31UGU 32VNM 32VPN 33TWN 33VXK
32UNU: NO      32TPT: NO      33TUN: NO    (no Bavaria-area tile at all)
extreme split, 4 tiles:    32UMC 32UNC 32UPC 32UQC   (32UNU NOT among them)
```

Two corrections follow. **H1 as originally scoped is unavailable on 32UNU** --
not pending a download, not obtainable. And the 20 working cubes are **not "the
extreme split"** as called since 2026-08-02: they came from `--split train`,
where 32UNU holds 192 cubes. Nothing measured changes (every property actually
used was verified from the files), but `docs/specs/phase1_3_cv.md` reserved
"the 20 extreme cubes" for P3 on a false premise, and the real extreme split is
untouched -- so growing the 32UNU pool is unblocked.

Seasonal cubes are schema-identical where they do exist (same 4 bands, same 8
E-OBS variables, 1-day axis, `esawc_lc`, `cop_dem`), but E-OBS completeness is
not guaranteed off 32UNU: on 30TVN `eobs_fg` is **absent from 20 of 25 cubes**
and `eobs_qq` misses the same trailing 13 days in every one. Nothing is filled;
`weather_finite6` is the registered fully-finite intersection, and numbers from
it are not directly comparable with 32UNU's 8-variable ones.

## 2026-08-10: the manifest's E-OBS columns were joined to the wrong day

Found while building P4, whose entire input is the weather. **0 of 264 manifest
rows carried the weather of the day their frame was acquired.**

A GreenEarthNet minicube is stored on a ~150-step DAILY grid of which about 29
steps carry an acquisition. `data.loader.load_cube` drops the empty ones, so
`sel.kept_idx` -- and therefore `original_axis_index` -- counts ACQUISITIONS.
The in-cube E-OBS series are on the DAILY grid. `manifest_rows` indexed the
latter with the former.

```
daily_index - original_axis_index:  min 4  median 53  max 122   (0/264 equal)
eobs_tg  manifest - true:  mean -3.46  MAE  6.26 K   max |.|  22.0 K
eobs_rr                    mean +1.82  MAE  2.52 mm  max |.|  35.0 mm
eobs_qq                    mean -36.5  MAE 89.07     max |.| 232.0
```

A frame acquired 2018-04-22 carried the mean temperature of 2018-03-17: 0.0 C
instead of 17.0 C.

Nothing caught it, and nothing could have. The values were finite, in range, the
right dtype, the right shape, and internally consistent -- they were simply
another day's weather. No assertion on the manifest alone can detect that; the
only test is to go back to the file and look the day up by TIMESTAMP.

**Fixed.** `encoders/manifest.py` gains `cube_daily_axis`,
`frame_daily_positions` and a `daily_axis_index` column; the E-OBS join is by
timestamp with an exact-match assertion (never a nearest neighbour); and
`assert_weather_join` re-derives the whole join from the cubes and refuses any
disagreement above 1e-9. It passes at max abs difference **0** over 264 rows x 8
variables x 20 cubes, and `tests/test_p4_ceiling.py` reconstructs the defect and
proves the guard fires.

`original_axis_index` is unchanged -- it is the embedding join key asserted in
`probes.cv.join_embeddings`, and P1's numbers do not touch weather, so nothing
in Phase 1.2-1.4 is affected.

## 2026-08-09: Phase 1.4 (P1) exit test PASSED -- 432 rows, and the band-matched baseline reverses the headline

`probes/p1_appearance.py`. Month and season decoded from ONE frame's frozen
embedding, five encoders, with the not-a-network baseline, a BAND-MATCHED
baseline, and a degenerate retention control in the same table. Nothing
fine-tuned, nothing re-encoded.

Supersedes the 2026-08-08 entry (192 rows), which is retained below for the
record. Four things changed and all four were forced by that run's own
diagnostics: the band-matched baseline, both class weightings, wider grids, and
the `FeatureBlock` refactor.

```
                              local (canonical)     Colab (reproduction)
Step 5   tests                323 passed, 5 skip    322 passed, 6 skip
Step 11  run_p1               432 rows, 85.4 min    432 rows, 83.8 min
Step 12  invariants           COMPLETE              COMPLETE
Step 13  weakest Spearman     +0.400                +0.400
Step 15  artefacts            432 x 48 CSV + PNG    same
```

The two environments are Python 3.9.6 / 3.12 with different sklearn, scipy and
BLAS. **Agreement to +/-0.003** on every score; Figure 1's explained variance is
identical to the decimal (PCA is an exact SVD). The one Colab skip is
`test_real_cache_joins_...`, which globs `data/raw` relative to cwd and
self-skips inside a phase checkout; Step 8 asserts the equivalent against the
resolved path.

### The realised class distribution, and the chance level derived from it

Unchanged from 2026-08-08 and repeated because everything below is read against
it: **8 realised months (April-November), not 12**, chance 1/8 = 0.1250;
**3 realised seasons (no DJF), not 4**, chance 1/3 = 0.3333. Counts
04:29 05:40 06:43 07:63 08:36 09:29 10:19 11:5, and JJA 142 / MAM 69 / SON 53.
The tile has **15** distinct time windows, not the 16 recorded since
2026-08-02: five window strings appear on two cubes each.

### THE CORRECTION: given the same bands, the networks WIN

2026-08-08 reported "raw_features wins every single-image comparison" and read
it as an input effect that could not be separated. It can be separated, for
free: `raw_features` stores its per-band statistics in a known column order, so
the band-matched baseline is a COLUMN SLICE of an array already on disk.

month / cube k=5 / logreg / grid_cell, balanced accuracy:

```
                                bal-acc      vs CONTROL   vs BAND-MATCHED
raw_features (all four bands)  0.430 +/-0.064   +0.148        +0.102
dinov2_vitb14                  0.387 +/-0.092   +0.105        +0.059
satlas_s2_swinb_rgb            0.386 +/-0.046   +0.104        +0.058
imagenet_vit_b16               0.350 +/-0.054   +0.068        +0.021
satlas_s2_swinb_mi_rgb         0.342 +/-0.058   +0.060        +0.014   NOT-SI
raw_nir_ndvi   (B8A + NDVI)    0.339 +/-0.034   +0.057        +0.011
raw_rgb_only   BAND-MATCHED    0.328 +/-0.065   +0.046         0.000
DEGENERATE CONTROL             0.282 +/-0.118    0.000        -0.046
```

**Every network beats hand-crafted percentiles of the same three bands.**
`raw_features` led only because it also sees B8A and NDVI, and neither half
alone (0.328 RGB, 0.339 NIR+NDVI) approaches the combination (0.430) -- the win
is in having both, not in either.

Margin over the band-matched baseline, all cells (logreg, grid_cell):

```
                      raw_features  dinov2  satlas  imagenet     MI
month  cube              +0.102     +0.059  +0.058   +0.021    +0.014
month  loco              +0.092     +0.038  +0.046   +0.003    -0.041
month  spatial_block     +0.059     -0.017  +0.049   -0.020    -0.041
season cube              +0.145     +0.095  +0.062   +0.025    +0.200
season loco              +0.137     +0.067  +0.047   +0.006    +0.173
season spatial_block     +0.071     -0.016  +0.015   -0.060    +0.154
```

The representation advantage is real but **not robust to the strictest split**:
under `spatial_block`, `dinov2_vitb14` and `imagenet_vit_b16` fall BELOW
hand-crafted features on the same bands.

### The retention control, by cell

Of 120 grid_cell rows, **42 sit at or below** `[clear_frac, window_span_days]`
-- two numbers, no image. Where they sit is the finding:

```
month  cube            0/20 at or below the control
month  loco            0/20
month  spatial_block   6/20
season cube           10/20
season loco            9/20
season spatial_block  17/20
```

**Month under `cube` and `loco` is clean** -- every row clears the control, by
+0.05 to +0.15. **Season is not** -- and under `spatial_block` almost nothing
clears it. Cloud retention on this subset is strongly seasonal, three classes
make it an easy proxy, and holding out geography removes the appearance signal
faster than it removes the retention signal. Where P1 is cited, cite **month
under cube or LOCO**, and cite the margin over the control.

### The positive control behaves like a positive control

`satlas_s2_swinb_mi_rgb` is the only encoder that can represent change, and it
is the only one that beats the retention control on SEASON: +0.092 (cube),
+0.041 (loco), and +0.154 to +0.209 over the band-matched baseline in every
mode. It is simultaneously the WORST single-image-comparable performer on
month (-0.041 vs band-matched under loco and spatial_block).

That is exactly the predicted signature and it is why the row is flagged
`si_comparable=False`: MI aggregates 8 retained frames over 0-105 days, so it
is answering "which season is this stretch" while the others answer "which
month is this frame". Conditioned on its own lookback tercile it is best where
the lookback is SHORTEST, i.e. where it is closest to single-image.

### Regularisation, and the grid edge closed out

Widening `C` to 100 and `alpha` to 1e5 cut edge selection from **34.7% to
21.3%** of 4320 outer folds. Plain `logreg` still selects the top edge in 27.5%
(297/1080), which raised the obvious question of whether the grid is STILL
truncating. Measured, on satlas grid_cell, cube fold 1:

```
C=100    test bal-acc 0.4492
C=1000   test bal-acc 0.4527   98.9% of predictions identical to C=100
C=10000  test bal-acc 0.4533   99.9% identical to C=1000
```

**The fit has saturated.** Selecting the top edge here means "the data wants
essentially no regularisation", which is the honest answer for a 4224-row by
768-column design matrix, and extending further moves the score by ~0.004
against a fold spread of +/-0.044. The grid is adequate; `n_at_grid_edge`
stays in the CSV.

`ridge` is now bracketed on both sides (25/1080 at 1e4, none at 1e5).
`logreg_balanced` uses the whole grid (85 folds at 1e-4, 189 at 100), which is
the imbalance correction doing visible work.

### Class weighting helps the BASELINE most

Both weightings run. Under `logreg_balanced`, month/cube: raw_features
0.430 -> 0.450, while `imagenet_vit_b16` and MI drop BELOW the band-matched
baseline. November is 5 frames of 264 and an unweighted loss forfeits up to 1/8
of balanced accuracy by never predicting it -- hand-crafted percentiles have
the signal to recover it and the weakest networks do not. Reported as a pair;
neither is "the" answer, and the control is weighted identically in each, so
`margin_over_control` is like-for-like under both.

### Fold modes agree in ORDERING -- and the ordering is not stable to a version bump

Weakest pairwise Spearman rho across all comparisons: **+0.400**, in both
environments. But the individual cells differ BETWEEN environments: month /
logreg / cube-vs-spatial_block is +0.800 locally and +0.400 on Colab, because
`dinov2_vitb14` and `satlas_s2_swinb_rgb` land at 0.387/0.386 on one machine
and 0.389/0.389 on the other.

**Two encoders separated by less than floating-point noise cannot be ranked.**
That is not a defect in the protocol; it is the sample size, and it is the
sharpest available argument that 20 cubes / 1 tile / 1 year cannot support a
comparative claim. Recorded as a limit, not worked around.

### Figure 1, the latent clock

264 frames pooled across 20 cubes / 15 time windows, 8 realised months, five
panels on shared axes (99th-percentile radius; 5 of 1320 points outside, counted
per panel). PC1+PC2 explained variance: raw_features 69.6%, MI 43.0%,
imagenet 36.7%, satlas_rgb 28.7%, dinov2 25.9%. Every panel shows a month
gradient; `imagenet_vit_b16` traces the clearest arc despite being the weakest
decoder, which is a reminder that two leading PCs and a linear probe over the
full space are different questions. Descriptive only.

### Cost

85.4 min on 7 CPU workers: 432 rows = 4320 outer folds, each with a 3-fold
inner loop over a 7-point grid, ~95,000 fits. No GPU, and none would help --
these are sklearn solvers. Peak memory under 4 GB.

## 2026-08-08: Phase 1.4 (P1) first run, 192 rows -- SUPERSEDED by 2026-08-09

> **Superseded.** Kept for the record. Its headline ("raw_features wins")
> was an artefact of comparing RGB-only networks against a baseline that
> also sees B8A; the 2026-08-09 entry above measures that and reverses it.
> Its estimator set (2, unweighted) and C grid (capped at 1) are also
> superseded. Numbers below are correct for what they measured.

`probes/p1_appearance.py`. Month and season decoded from ONE frame's frozen
embedding, every encoder, with the not-a-network baseline and a degenerate
retention control in the same table. Nothing fine-tuned, nothing re-encoded.

```
Step 5   302 passed, 5 skipped   (307 collected)         12.8 s
Step 6   manifest (264, 21), 20 cubes, tile ['32UNU'], year [2018]
Step 7   realised distribution printed, chance DERIVED  (see below)
Step 8   100 .npz -> 100 usable (20 x 5) at v3; join re-asserted per encoder
Step 9   30 folds re-derived from the manifest, importing nothing from cv
Step 10  selected C identical on all 5 folds under a poisoned test set
Step 11  run_p1: 192 rows in 33.9 min on 7 workers
Step 13  weakest pairwise Spearman rho across fold modes: +0.400
Step 14  Figure 1, 5 panels, shared axes
Step 15  1 CSV (164 kB) + 1 PNG (246 kB) under data/phase1_4/
```

### The realised class distribution, and the chance level derived from it

Not 12 months and not 4 seasons. The cube windows plus `clear_frac > 0.5`
decide what exists:

```
month    264 rows, 8 REALISED classes -- April..November
   04  29 (11.0%)   05  40 (15.2%)   06  43 (16.3%)   07  63 (23.9%)
   08  36 (13.6%)   09  29 (11.0%)   10  19 ( 7.2%)   11   5 ( 1.9%)
   NOT realised: [1, 2, 3, 12]
   chance  balanced accuracy 0.1250 = 1/8   macro-F1 0.0484
   majority 07 at 23.9%,  most-frequent dummy 0.132 (measured, cube k=5)

season   264 rows, 3 REALISED classes
   JJA 142 (53.8%)   MAM 69 (26.1%)   SON 53 (20.1%)
   NOT realised: ['DJF']
   chance  balanced accuracy 0.3333 = 1/3   macro-F1 0.2332
```

**Three recorded figures are corrected by this.** The P1 spec assumed "roughly
March-December, so month is ~10 classes": it is **8**, April-November. March
and December have cube windows but no frame survives the clear-fraction filter.
Season is realised **3-way**, not 4. And the tile has **15** distinct time
windows, not the 16 recorded since 2026-08-02 -- five window strings appear
twice across the 20 cubes (`2018-03-19_2018-08-15`, `2018-03-29_2018-08-25`,
`2018-04-23_2018-09-19`, `2018-04-28_2018-09-24`, `2018-05-03_2018-09-29`).
The Figure 1 argument is unaffected and slightly strengthened.

### THE HEADLINE: the degenerate control is not a floor, it is a competitor

`[clear_frac, window_span_days]` alone, D=2, **no embedding anywhere in it**:

```
target  fold mode      est      level   bal-acc         chance   best SI encoder
season  spatial_block  logreg   cell    0.681 +/-0.119  0.333    0.584 (raw_features)   <- CONTROL WINS
season  loco           logreg   cell    0.674 +/-0.233  0.333    0.673 (raw_features)   <- CONTROL TIES
season  cube           logreg   cell    0.627 +/-0.123  0.333    0.654 (raw_features)
month   spatial_block  logreg   cell    0.328 +/-0.052  0.125    0.351 (raw_features)
month   loco           logreg   cell    0.315 +/-0.209  0.125    0.460 (raw_features)
month   cube           logreg   cell    0.282 +/-0.118  0.125    0.425 (raw_features)
```

Every one of those is far above chance. **Cloud retention on this subset is
strongly seasonal**, which is not surprising once stated -- Alpine-foreland
cloud climatology is itself a season -- but the size is: two numbers that
contain no image at all reach 0.68 balanced accuracy on a 3-class season task,
beating all five frozen encoders under `spatial_block`.

Encoder margin OVER the control, balanced accuracy, `grid_cell` + logreg
(negative means the two-number control WINS):

```
                       raw_features   dinov2   satlas_rgb   imagenet
month  cube               +0.143      +0.116     +0.111      +0.070
month  loco               +0.145      +0.099     +0.109      +0.063
month  spatial_block      +0.023      -0.053     +0.011      -0.060
season cube               +0.027      -0.012     -0.043      -0.083
season loco               -0.000      -0.056     -0.071      -0.129
season spatial_block      -0.097      -0.161     -0.126      -0.224
```

Consequences, and they bind on how the rest of this table may be read:

* **Only month, and only under `cube` and `loco`, separates any encoder from
  the control** -- there all four single-image encoders lead it by +0.06 to
  +0.15. That is the one cell of this table where P1 measures a representation
  rather than a cloud climatology.
* **A P1 SEASON score is not evidence of representation quality on this
  subset.** Every encoder clears chance comfortably and NONE of them clears the
  control anywhere: on season the margins are at best +0.027 and at worst
  -0.224. P1's season column cannot separate "the embedding encodes season"
  from "the embedding encodes how cloudy it was".
* **`spatial_block` is where it bites hardest**, which is the wrong direction
  for comfort: it is the strictest split available here, the closest this
  subset comes to held-out geography, and under it only `raw_features`
  (+0.023) and `satlas_s2_swinb_rgb` (+0.011) stay above the control on month
  at all, while `dinov2_vitb14` and `imagenet_vit_b16` fall below it.
* Eight classes make retention a weaker proxy than three do, which is why month
  survives where season does not.
* This is exactly what the control existed to catch. It was specified as "not
  optional" on the expectation that it would sit near the floor; it does not.

### Per-encoder scores, PRIMARY feature set (grid_cell, 4224 rows, logreg)

Mean +/- sd across folds, `[min, max]`. The spread is the honest uncertainty at
20 cubes and is routinely larger than the gap between two encoders.

```
month  (chance 0.125, dummy 0.132/0.199/0.147 for cube/loco/spatial_block)
                          cube                   loco                   spatial_block
raw_features           0.425 +/-0.068         0.460 +/-0.096         0.351 +/-0.050
                       [0.326, 0.491]         [0.239, 0.608]         [0.302, 0.427]
dinov2_vitb14          0.397 +/-0.086         0.414 +/-0.142         0.275 +/-0.089
satlas_s2_swinb_rgb    0.393 +/-0.045         0.424 +/-0.145         0.339 +/-0.141
imagenet_vit_b16       0.352 +/-0.057         0.377 +/-0.142         0.268 +/-0.048
DEGENERATE CONTROL     0.282 +/-0.118         0.315 +/-0.209         0.328 +/-0.052
satlas_s2_swinb_mi_rgb 0.320 +/-0.070         0.331 +/-0.131         0.249 +/-0.085   NOT SI-COMPARABLE

season (chance 0.333, dummy 0.333/0.433/0.367)
raw_features           0.654 +/-0.061         0.673 +/-0.108         0.584 +/-0.072
dinov2_vitb14          0.616 +/-0.069         0.618 +/-0.167         0.519 +/-0.100
satlas_s2_swinb_rgb    0.584 +/-0.072         0.603 +/-0.175         0.555 +/-0.100
imagenet_vit_b16       0.544 +/-0.081         0.544 +/-0.155         0.457 +/-0.105
DEGENERATE CONTROL     0.627 +/-0.123         0.674 +/-0.233         0.681 +/-0.119
satlas_s2_swinb_mi_rgb 0.677 +/-0.114         0.704 +/-0.112         0.624 +/-0.109   NOT SI-COMPARABLE
```

**Read the DUMMY column, not the chance column, under LOCO.** Balanced accuracy
is scored over the classes present in each test fold, and a fold holding one
cube holds about 5 of the 8 months, not all 8. Its implicit floor is therefore
~1/5, which is exactly what the measured most-frequent dummy says: 0.132 under
`cube` (8 classes, close to the global 1/8 = 0.125), **0.199 under LOCO**,
0.147 under `spatial_block`. Comparing a LOCO score against the global 0.125
would credit it with a margin the split never offered. The dummy is refit per
fold for this reason and is carried in the CSV beside every score.

**No encoder FAILS P1.** Every one clears chance by a wide margin on both
targets and under all three fold modes, so the surprise this probe was watching
for -- an EO model trained to appearance-invariance -- did not occur. P1 passes
as calibration and P2/P3 are licensed.

**`raw_features` wins every single-image comparison.** Read with the caveat
recorded in `docs/DECISIONS.md`: the baseline reduces all four bands including
**B8A** and adds NDVI statistics, while every network encoder here is RGB-only.
That is a statement about the input as much as about the representation, and
the clean comparisons are network-vs-network and network-vs-control.

**Among the RGB-only networks, the EO-native and the generalist beat the
ImageNet floor and are indistinguishable from each other**: dinov2 and
satlas_rgb swap places between fold modes, always inside one standard
deviation. On this subset P1 does not rank them.

### pooled vs grid_cell: the p >> n correction is worth 0.06-0.13

month / cube / logreg, balanced accuracy:

```
                        pooled (264 rows)     grid_cell (4224 rows)   difference
raw_features            0.550 +/-0.135        0.425 +/-0.068          -0.125
dinov2_vitb14  D=3840   0.454 +/-0.113        0.397 +/-0.086          -0.057
satlas_s2_swinb_rgb     0.391 +/-0.067        0.393 +/-0.045          +0.002
imagenet_vit_b16        0.348 +/-0.101        0.352 +/-0.057          +0.004
satlas_s2_swinb_mi_rgb  0.335 +/-0.073        0.320 +/-0.070          -0.015
```

The pooled score is HIGHER exactly where p >> n bites hardest -- DINOv2 at
D=3840 against 264 rows, and raw_features whose 35 hand-built features are
frame-level summaries that the per-cell version cannot see. The fold spread
roughly halves under grid_cell for every encoder. Both are reported; the
grid_cell column is the one to read.

### The multi-image control, conditioned on its own lookback

Terciles of the REALISED `window_span_days`, cut at 35 and 70 days (the nominal
35-day span for 8 frames at a 5-day revisit is wrong by a factor of three).
Chance is recomputed on each tercile's own class distribution, because
subsetting by lookback changes which months are present:

```
target  tercile  days     rows  classes  chance  bal-acc (cube, logreg)
month   short     0-30      84      5     0.200  0.402 +/-0.221
month   medium   35-65      90      7     0.143  0.185 +/-0.043
month   long     70-105     90      6     0.167  0.296 +/-0.099
season  short     0-30      84      2     0.500  0.753 +/-0.180
season  medium   35-65      90      3     0.333  0.531 +/-0.138
season  long     70-105     90      2     0.500  0.622 +/-0.134
```

MI is best where its lookback is SHORTEST, i.e. where it is closest to being a
single-image encoder, and worst in the middle tercile where the label is most
smeared. That is the caveat behaving exactly as predicted, measured rather than
argued, and it is why no MI number is placed in a ranked column with the
single-image encoders.

### Selected regularisation strengths, and one thing to fix next time

Printed per outer fold in `selected_params`. Over all 1920 outer folds:

```
logreg C      1e-4:5   1e-3:32   1e-2:161   1e-1:212   1:550
ridge  alpha  0.1:86   1:195     10:131     100:247    1e3:276   1e4:25
666/1920 folds (34.7%) selected at a GRID EDGE -- 555 of them logreg
```

**The logreg C grid is too narrow at the weak end.** C=1 is its maximum and is
chosen in 57% of folds, so for those folds the regularisation was pinned by the
grid rather than by the data and the reported score is a lower bound. The ridge
grid is fine (interior modes at 100-1000, only 25/960 at the top edge). Extend
`LOGREG_C_GRID` upward (to 10, 100) before P2 reuses this machinery; the
direction of the bias is known -- scores can only go up -- so no conclusion
above is reversed by it, but "logreg beats ridge" is partly an artefact of
where the two grids end.

### Fold modes agree in ORDERING, as required

Spearman rho over the four single-image encoders, `grid_cell`:

```
                        cube-loco   cube-spatial_block   loco-spatial_block
month  logreg             +0.800          +0.800               +1.000
month  ridge              +0.800          +0.400               +0.800
season logreg             +1.000          +0.800               +0.800
season ridge              +1.000          +0.800               +0.800
```

Levels differ as expected (LOCO trains on 19 cubes and tests on 1;
`spatial_block` holds out whole geographic clusters and is uniformly the
hardest). Ordering holds, with one weak cell: month/ridge cube-vs-spatial_block
at +0.400, where dinov2 and raw_features change places -- both moves are well
inside one fold-spread, so it is a tie being resolved differently, not a
contradiction.

### Figure 1, the latent clock

264 frames pooled across 20 cubes / 15 time windows, coloured by the 8 realised
months, five panels on shared axes (99th-percentile radius; 5 of 1320 points
fall outside and the count is printed per panel). PC1+PC2 explained variance:

```
raw_features            46.4% + 23.2% = 69.6%
imagenet_vit_b16        26.9% +  9.8% = 36.7%
dinov2_vitb14           15.6% + 10.2% = 25.9%
satlas_s2_swinb_rgb     15.3% + 13.4% = 28.7%
satlas_s2_swinb_mi_rgb  30.0% + 13.0% = 43.0%
```

Every panel shows a visible month gradient; `imagenet_vit_b16` traces the
clearest arc (April/May at one end, October/November at the other) despite
being the weakest decoder in the table -- a reminder that two leading PCs and a
linear probe over the full space are different questions. **Descriptive only:
the PCA is fitted on all 264 rows because it estimates nothing, and no score is
read off it.**

### Cost

`run_p1` 33.9 min on 7 CPU workers for 192 rows = 1920 outer folds, each with a
3-fold inner tuning loop over a 5- or 6-point grid (~40k model fits). Whole
notebook, including the 20-cube manifest build and the test suite, ~36 min.
No GPU, no weights, peak memory well under 4 GB.

## 2026-08-06: Phase 1.3 exit test PASSED on Colab, and four silent skips closed

The full five-encoder Phase 1.3 run, on the corrected nested Drive layout.
Every gate green.

```
Step 3   100 .npz -> 100 usable (20 cubes x 5 encoders) at v3
Step 5   255 passed, 9 skipped   (264 collected)      <- see below
Step 6   manifest (264, 21), 20 cubes, tile ['32UNU'], year [2018]
Step 7   cube k=5 / LOCO / spatial_block [9,3,4,3,1] / temporal 17-3, 65 dropped
         31 folds re-checked independently: no cube on both sides
Step 8   year / tile / crossed all RAISE
Step 9   same-cube gate fires; duplicate rows refused
Step 10  COMPLETE: 20 x 5 = 100 pairs; JOIN CONTRACT on all 100, 1320 rows
Step 11  5 files, 0.20 MB under data/phase1_3/folds/
```

`dinov2_vitb14` at **D=3840** ran for the first time in a Phase 1.3 context,
so the join contract is now asserted on the FULL roster rather than the four
encoders a Python 3.9 dev box can build. 264 rows x 5 encoders = 1320.

MI `window_span_days` min 0, median 55, max 105 days -- unchanged through the
re-stamp of all 100 files, which is the point of that migration being
array-preserving.

### The 100-file re-stamp

All 100 embeddings were COMPLETE but UNSTAMPED (written between a1a6a12 and
f4ed234). `scripts/restamp_cache.py` reported `INCOMPLETE 0`, then stamped all
100 and re-verified each through the real `load_encoded`. Verified beforehand
on a local reproduction: array contents bit-identical across the migration in
80 of 80 files, only `schema_version` added.

### Four tests were silently skipping on Colab

`255 passed, 9 skipped` against a documented `259 passed, 5 skipped` -- same
264 collected, so the bundle was current. The gap was four real-cube tests
(`test_grid_landcover_aligns_with_the_embedding_grid`,
`test_grid_landcover_is_finer_than_the_per_cube_label`,
`test_in_cube_eobs_is_present_and_finite`, `test_elevation_is_cached_per_cell`)
skipping because `_real_cube()` globbed `data/raw/*.nc` relative to the WORKING
DIRECTORY. Locally that is the repo root and the cubes are there; on Drive it
is the phase checkout, and the cubes are SHARED one level up at the project
root. The per-phase layout introduced this and nothing caught it: the gate was
quietly weaker in the only environment the phases actually run in.

**Changed.** The lookup is anchored on the test FILE, not the cwd, and checks
both layouts -- `<repo>/data/raw` then `<repo>/../data/raw`. The skip message
now names both paths it searched, so a future skip explains itself. Verified
by running the real bundle from a simulated phase checkout with the cubes one
level up: **261 passed, 5 skipped**, identical to local; and with no cubes
anywhere, an honest 4-test skip naming both directories.

Test count 259 -> **261 passed, 5 skipped (266 collected)**: two tests pinning
that the lookup covers both layouts and does not depend on the cwd.

## 2026-08-06: the flat legacy layout, found by the whole-Drive scan

Running `organise_drive.ipynb` Step 5 (the whole-Drive scan added earlier
today) against a real Drive surfaced a fifth, separate defect from the last
incident's four -- not another stale duplicate, but the project's actual
CURRENT, CORRECT 100-file five-encoder embedding set sitting somewhere no
resolver looks:

```
NeurIPS-CCAI-2026/phase1_2/data/embeddings/        100 x .npz   <- the real one
NeurIPS-CCAI-2026/phase1_2/Back-Up/data/embeddings/ 80 x .npz   <- an older backup, same shape
NeurIPS-CCAI-2026-phase1_3/data/phase1_2/embeddings/ 50 x .npz   <- the ALREADY-diagnosed
                                                                    'Copy of' duplicates
```

Every resolver in the project looks for `data/phase1_2/embeddings` (the
nested, phase-scoped path from `f4ed234`). The real data sits at the FLAT
`data/embeddings` -- one folder short -- because this checkout's embeddings
predate that refactor even though its code does not.

### Verified against a reconstruction of the exact tree

```
before  LEGACY  phase1_2/Back-Up/data/embeddings   (2 .npz)
        LEGACY  phase1_2/data/embeddings            (5 .npz)
                rename to: phase1_2/data/phase1_2/embeddings
after renaming the real one:
        LEGACY  phase1_2/Back-Up/data/embeddings   (2 .npz)   <- backup, left alone
        (the real one no longer flagged)
```

`organise_drive.ipynb` Step 5 now detects this class of directory directly:
any `data/embeddings` or `data/masks` with no phase segment, holding at least
one `.npz`, reported with the exact rename computed from the nearest
`phase1_N`-named ancestor. Detection only -- nothing is moved automatically,
same as every other tool here.

Test count 251 -> **259 passed, 5 skipped (264 collected)**: 8 new tests in
`tests/test_legacy_layout.py`, extracting the sentinel-fenced detector the same
way `test_notebook_resolver.py` already does for the resolver. Confirmed the
guard fires when the detection condition is deliberately broken.

## 2026-08-06: the Drive cache audit, and four faults in one traceback

A real Phase 1.3 run died at Step 10. The traceback named a schema version, but
the schema was the LAST of four faults in the chain, and the first three were
the ones that made it inevitable. Recorded here because each has a different
fix and only one was already known.

```
encoders in .../NeurIPS-CCAI-2026-phase1_3/data/phase1_2/embeddings: [...5 encoders]
AssertionError: Copy of 32UNU_2018-05-03_..._dinov2_vitb14.npz was written
               with cache schema v0, but this code expects v3
```

| # | fault | why it was invisible |
|---|---|---|
| 1 | `EMB_IN` was a stale copy INSIDE the phase1_3 checkout | "this checkout first" is not a selection |
| 2 | Drive had renamed a duplicate to `Copy of ....npz` | `C` sorts before a digit, so `sorted(glob)[0]` picked it every time |
| 3 | that copy predated the schema stamp | one bad file halts the run at an arbitrary point |
| 4 | only 1 pair per encoder was ever joined | a cache with holes would have passed |

Fault 2 is the one worth remembering: the selection was **deterministic and
deterministically wrong**. It would have picked the copy on every future run,
on every machine.

### Measured on the real cache after the fix

```
[audit] 80 .npz on disk -> 80 usable (20 cubes x 4 encoders)
[audit] COMPLETE: all 20 x 4 = 80 (cube, encoder) pairs present at v3
JOIN CONTRACT asserted on 80 (cube, encoder) pairs, 1056 rows total
  imagenet_vit_b16        D=1536   264 rows   window_span_days 0/0/0
  raw_features            D=35     264 rows   window_span_days 0/0/0
  satlas_s2_swinb_mi_rgb  D=1024   264 rows   window_span_days min 0 med 55 max 105
  satlas_s2_swinb_rgb     D=1024   264 rows   window_span_days 0/0/0
```

**The MI figures now cover all 20 cubes rather than one**, and reproduce the
2026-08-03 measurement exactly: min 0, median 55, max 105 days. The earlier
0/38/85 in the Phase 1.3 entry below was cube 1 alone, which is why it differed.

Step 10 previously joined 4 pairs (one per encoder). It now joins **80**, which
is the check that actually protects P1-P4: a cube missing one encoder turns a
per-encoder comparison into a comparison over different cubes.

### The resolver, pinned against the tree that failed

`tests/test_notebook_resolver.py` extracts the sentinel-fenced resolver block
out of the notebook and runs it against simulated Drives -- the nested layout,
the older sibling layout that actually failed, and a plain dev clone. The
precedence rule is asserted directly: a `data/phase1_2` inside the phase1_3
checkout loses even when it holds **500 files against the real folder's 3**.
Verified the guard fails when the rule is regressed to first-hit-wins.

One false positive found and fixed while testing: in a plain development clone
the repo root IS where `data/phase1_2` belongs, so the "inside this checkout"
demotion is gated on the checkout actually being a phase folder
(`PHASE in basename(REPO)`, which covers both `phase1_3` and the older
`NeurIPS-CCAI-2026-phase1_3`). A warning that fires when nothing is wrong is a
warning nobody reads the second time.

### organise_drive Step 5 now scans the whole Drive

Previously it walked only the project folder -- which is exactly why the stale
sibling was invisible. It now starts at `My Drive`, expands project-like
folders in full, lists everything else by name, and warns when more than one
project folder sits at the top level. On a simulated Drive holding both
folders it flags the sibling and the single `Copy of ...npz`, and no longer
mis-flags `.nc` cubes (which have no `__` at all).

### A fourth thing, found while verifying the fix

`sh(f"{PY} -m pytest tests -q")` combined with `addopts = -q` in `pytest.ini`
is **-qq**, which SUPPRESSES pytest's final `N passed, N skipped` line. The
number the runbook tells you to compare against -- and the project's own
stale-bundle signal -- was never printed at Step 5 in either notebook. Both now
let the ini file own the verbosity, and the archived run shows
`251 passed, 5 skipped` where there was previously nothing.

Test count 223 -> **251 passed, 5 skipped (256 collected)**: 19 for the audit,
9 for the resolver.

## 2026-08-05: the Drive cache is unstamped, not incomplete -- a verified re-stamp

**Observed.** Phase 1.3 Step 10 stopped on Colab with the schema guard firing:

```
AssertionError: Copy of 32UNU_2018-05-03_..._dinov2_vitb14.npz was written with
cache schema v0, but this code expects v3.
```

Two separate faults in one line, and the guard was right about both.

*The filename.* `Copy of ` is Google Drive's own prefix for a duplicated file.
Step 10 took `sorted(glob(...))[0]` per encoder, so a stray duplicate could be
the file that loads -- selection by alphabetical accident.

*The version.* `window_span_days` landed in **a1a6a12** and the schema stamp in
**f4ed234**, a LATER commit. Artefacts written between the two carry every
field and no stamp. So "v0" here means UNSTAMPED, which is not the same as
incomplete, and the two have very different costs: a re-stamp takes seconds,
a re-encode takes a GPU run.

**Changed.** Three things, none of which weakens the guard.

1. `encoders.pipeline.inspect_encoded` reads a file WITHOUT enforcing the
   version, so the two cases can be told apart. `migrate_to_current` re-stamps
   a file only if every key in the new `REQUIRED_KEYS` is present AND the full
   `assert_encoded` set passes; it writes atomically through a temp file and
   then re-opens the result through the real `load_encoded`. `assert_encoded`
   is the invariant block extracted from `load_encoded`, so a migrated file is
   held to exactly the load-time standard rather than a subset.

   **Nothing is invented.** A missing `window_span_days` is NOT recomputed from
   the timestamps sitting beside it, and a test asserts that specifically. The
   hazard the stamp exists to prevent was silence -- a missing key read as
   merely absent; requiring every key and re-running every assertion is the
   opposite of that.

2. `scripts/restamp_cache.py`, dry-run by default: reports current /
   unstamped-but-complete / genuinely-incomplete / Drive-duplicate counts, then
   migrates only the second class.

3. Phase 1.3 Step 10 now **diagnoses the whole cache before joining anything**.
   It recognises a file as ours only when the cube half of its name is a cube in
   the manifest -- exact, and needs no locale-specific matching for `Copy of` /
   `Kopie von` / `Copie de`. One stale file used to halt the run at an arbitrary
   point with the rest undiagnosed; now every file is classified first and the
   remedy is printed once.

   Step 10 does not migrate anything itself: Phase 1.3 must not write into
   Phase 1.2's folder.

**Verified** on a copy of the real cache, both paths executed:

```
dry run    7 files -> current 2, unstamped-complete 3, incomplete 1, duplicate 1
--apply    3 stamped v3 and re-verified; the incomplete one left alone
after      5 load through the real guard, 2 still refused (correct)
           the MI file keeps window_span_days max 85 d through the migration
```

and Step 10's stale path fired on a scratch cache holding 3 unstamped, 1
incomplete and 1 `Copy of` file, reporting all four before raising.

Test count 198 -> **223 passed, 5 skipped (228 collected)**; 25 new tests,
including one parametrized over every `REQUIRED_KEYS` field to prove each
absence is refused individually.

## 2026-08-05: a full-tree listing in both notebooks

`organise_drive.ipynb` Step 5 and `phase1_3_cv.ipynb` Step 12: every folder and
every file from the absolute project root, no depth limit, no summarising. The
existing cells could not answer "did it land where I think it did" -- they
print movable units and rollups, both collapsed.

The Phase 1.3 copy is wired to that notebook's variables rather than pasted:
it derives the PROJECT root from `REPO` (the parent when the checkout sits in
a `phase1_*` subfolder, `REPO` itself in the flat layout), then adds two
Phase 1.3-specific sections.

`PHASE 1.3 OUTPUTS` checks the five files Step 11 wrote are present.
`READ-ONLY INPUTS` prints where the cubes and the Phase 1.2 embeddings were
resolved from, tagging any that sit `[inside this checkout]`. In the nested
layout NEITHER should carry that tag, and its absence is the visible evidence
that this phase read another phase's artefacts without writing into its folder
-- the property the per-phase layout exists for, checked rather than asserted.

Both layout branches exercised, not reasoned about:

```
nested   OK data/raw 2 cubes / OK phase folders / OK nothing loose
         -> "Layout matches the target."
         read-only inputs carry NO [inside this checkout] tag
flat     NOTE data/ also holds the checkout modules -- expected, not stray
         -> points at organise_drive.ipynb rather than reporting a fault
```

The flat branch matters: a naive root check would have reported 17 "stray"
entries on a working flat checkout and sent someone chasing a fault that does
not exist.

No new classification logic, so the drift guard is untouched and the count
stays 198 passed, 5 skipped.

## 2026-08-05: the organiser had a bootstrap bug; a self-contained Colab notebook

**Observed.** `python -m scripts.inventory` does not work on Colab, and the
reason is structural rather than a path problem: the scripts live INSIDE
`phase1_3_repo.zip`, so running them requires a checkout already extracted into
the very folder you are trying to reorganise. A tool that tidies the folder a
checkout lives in cannot depend on that checkout existing. Shipping the
reorganiser inside the thing it reorganises was the mistake.

**Changed.** `notebooks/organise_drive.ipynb`: four cells, imports nothing from
the project, works on a Drive folder with no checkout in it. Mount, inventory
(with SHA-256), plan, verify. `APPLY = False` until you flip it.

That requires the classification rules to exist in two places, which the repo
otherwise forbids. The exception is narrow and the drift is guarded rather than
trusted: the notebook's rules sit between sentinel comments,
`tests/test_scripts_organise.py` extracts and execs that block, and compares
`classify` and `destination` against `scripts.inventory` over a table of 18
paths and 7 (path, checkout_phase) pairs, plus the `SKIP` and `SHARED`
constants. Verified the guard fails when the copies disagree, by breaking the
notebook on purpose and watching two tests go red.

**Verified end to end** on a simulated Drive tree, notebook executed rather
than reasoned about:

```
before  11 units at the root, data/ holding three kinds at once
after    3 units: data/raw (shared), phase1_2/, phase1_3/
         47 files before and after, SHA-256 multiset IDENTICAL
         a second plan is empty -- idempotent
```

Test count 171 -> **198 passed, 5 skipped (203 collected)**; the 27 new tests
are the drift guard. Every documented expectation updated in step.

## 2026-08-05: project-tree tooling, and the test count moves to 171

Two scripts, `scripts/inventory.py` and `scripts/organise_phases.py`, added to
reorganise a Drive checkout into the per-phase layout. Not science: nothing
here produces a number.

The pair is split so that the thing which LOOKS and the thing which MOVES are
different programs, and the mover imports its classification from the lister
rather than re-deriving it -- so the plan you approve in the listing is the
plan that runs.

Verified on a simulated Drive tree (131 files, 20.3 MB, 3 cubes + 120
artefacts + a loose checkout + two bundles):

```
before   11 units at the root, data/ holding three different kinds at once
after     3 units: data/raw (shared), phase1_2/, phase1_3/
          131 files before and after, sha256 multiset identical
          a second run plans 0 moves -- idempotent
```

Four refusals are pinned by tests rather than documented: `data/raw` is never
filed under a phase (mirroring `data.paths.reset_phase`), no destination may
escape the root, an existing destination is never overwritten, and a checkout
moves whole or not at all with its owning phase stated explicitly. The last one
matters because a checkout is **not** classifiable from the filesystem -- it
looks identical whichever bundle produced it -- so the script asks instead of
guessing from mtimes.

`apply_moves` re-checks every refusal immediately before touching disk rather
than trusting the plan it was handed; a hand-forged move of `data/raw` raises
there too.

### The test count moved to 171 passed, 5 skipped (176 collected)

Was 146/5 (151 collected). The 25 new tests are `tests/test_scripts_organise.py`.
It rose again to 198/5 the same day when the drift guard landed; see the entry
above.
Every documented expectation has been updated in step -- RUNBOOK (both phases),
README, and the Phase 1.3 notebook's Step 5 -- because a collection count that
disagrees with the docs is the project's stale-bundle signal, and a signal
nobody trusts is worse than no signal.

The 2026-08-05 exit-test run recorded below measured 146/5 and is left as
measured. It is dated evidence, not a live expectation.

## 2026-08-05: Phase 1.3 exit test PASSED, local CPU, 20 cubes

`notebooks/runs/phase1_3_cv_2026-08-05_localCPU.ipynb`. CPU only, no weights,
nothing re-encoded. These are split-structure numbers; the probes that turn
folds into results are P1-P4.

### Gates

```
Step 5   146 passed, 5 skipped      (118 pre-existing + 28 new fold tests)
Step 6   manifest (264, 21), 20 cubes, tile ['32UNU'], year [2018]
Step 7   cube k=5 / LOCO / spatial_block k=5 / temporal all yield folds
Step 8   year, tile, crossed all RAISE -- the designed behaviour
Step 9   same-cube gate fires on a seasonal manifest; duplicates refused
Step 10  join contract asserted on one real pair per encoder
Step 11  5 files, 0.20 MB under data/phase1_3/folds/
```

### Fold structure on the 20-cube subset

```
cube, k=5        test 52-53 rows / 4 cubes per fold; every row tested once
LOCO             20 folds, test sizes 10..16 (= T_kept per cube)
spatial_block    5 blocks by pixel_bbox centroid, sizes [9, 3, 4, 3, 1]
temporal         cutoff 2018-08-15: 17 cubes train / 3 test, 65 frames DROPPED
```

**31 folds re-checked independently** in the notebook (not by the code under
test): no `cube_id` on both sides of any fold, in any mode.

The spatial_block sizes are uneven because the 20 cubes are not uniformly
spread over the tile: 9 of them fall in one contiguous region (tile-pixel rows
1145-2937, columns 3705-5369, against a full extent of 889-5241 by 377-5369),
and one cube sits alone far enough from every other to be its own block. That
is a property of the round-robin selection over 16 time windows, not of the
clustering; complete linkage on the same centroids is deterministic and
reproduces the same blocks on every run (pinned by a test).

The temporal claim above was checked exhaustively rather than by sampling:
over **every** day in the retained-frame span, the number of cutoffs that
separate complete cubes with nothing straddling is **0**.

### What temporal mode costs, measured

The 16 time windows overlap so heavily that **no cutoff separates complete
cubes**: for every candidate date, at least one cube has retained frames on
both sides. Cube-atomic assignment (whole cube to its majority side,
wrong-side frames dropped) at 2018-08-15 keeps 199 of 264 frames, i.e.
**65 dropped, 24.6%**, and leaves 3 test cubes. That is the starvation the
spec predicted, quantified: it is why temporal is a P3 robustness variant and
never a default.

### The three refusals, verbatim

```
year     SingleYearError  -> "Use 'cube' mode ... or 'crossed' once the
                             manifest spans years. Do not fall back to a
                             random split"
tile     SingleTileError  -> "use 'spatial_block' as the prototype-scale
                             substitute ... DEFERRED TO SCALE-UP"
crossed  SingleYearError  -> "cube and year are perfectly confounded, so
                             grouping by cube already holds the year out"
```

### Join contract, real (cube, encoder) pairs

`(cube_id, original_axis_index) == (cube, kept_idx)`, asserted on kept_idx,
timestamps and clear_frac for every encoder present:

```
imagenet_vit_b16         (14, 1536)   window_span_days 0 exactly
raw_features             (14,   35)   window_span_days 0 exactly
satlas_s2_swinb_rgb      (14, 1024)   window_span_days 0 exactly
satlas_s2_swinb_mi_rgb   (14, 1024)   window_span_days min 0 median 38 max 85
```

The MI figures reproduce the 2026-08-03 measurement on cube 1 exactly. The
join REFUSES an embedding without `window_span_days` rather than returning one
silently missing the covariate.

**Four encoders, not five, in this run**: `dinov2_vitb14` cannot be encoded
locally (its hub code needs Python >= 3.10; the dev venv is 3.9.6). The Colab
run sees all five. This also forced a local `reset_phase("phase1_2")` and
re-encode, because the local `.npz` cache predated `SCHEMA_VERSION` and
`load_encoded` correctly refused it -- the v3 guard from `f4ed234` doing
exactly what it was added for.


## 2026-08-04: Phase 1.2 exit test PASSED on the full five-encoder roster

Colab T4, 20 cubes, `notebooks/runs/phase1_2_encoders_2026-08-04_T4_5enc.ipynb`.
The headline: **`dinov2_vitb14` ran against real weights for the first time.**
Its extraction was rewritten in `b84c484` to follow DINOv2's published
linear-probe protocol and had only ever been exercised through the synthetic
dummy encoder, because the real-weight tests need Python >= 3.10 and the local
venv is 3.9.6. It was the most structurally complex wrapper in the roster and
the only one never watched running. It works.

### D per encoder, all verified against real weights

| encoder | pooled D | grid | variants |
|---|---|---|---|
| `raw_features` | 35 | 4x4 x 35 | - |
| `imagenet_vit_b16` | 1536 | 4x4 x 768 | cls_last 768, patch_mean_last 768 |
| `dinov2_vitb14` | **3840** | 4x4 x 768 | cls_last 768, cls_last4_concat 3072, patch_mean_last 768 |
| `satlas_s2_swinb_rgb` | 1024 | 4x4 x 1024 | - (no CLS: Swin is hierarchical) |
| `satlas_s2_swinb_mi_rgb` | 1024 | 4x4 x 1024 | - (positive control) |

### Exit test

```
Step 4   107 passed, 9 skipped   (116 collected -- see the caveat below)
Step 6   five encoders built frozen; DINOv2's 4-block CLS concat live
Step 7   all five return [14, D] on cube 1, identical kept timestamps
Step 8   20/20 cubes pass, worst prevalence 1.89e-05 vs 1e-04
Step 10  peak GPU 1.56 GB at T=290, batch_size=16   (budget 12 GB)
Step 11  100 .npz = 20 cubes x 5 encoders, 264 frames per encoder, 0 cached
```

`T_kept` min 10, median 13, max 16, summing to 264 per encoder -- identical to
the local CPU run and to Phase 1.1's clear-frame counts. Three independent
routes to the same 264.

MI `window_span_days` on cube 1: min 0, median 38, max 85 days.

### Caveat: the run collected 116 tests, local collects 123

The 7-test gap is exactly `tests/test_paths.py`, which was UNTRACKED when the
bundle was built. `make_zip.sh` lists files with `git ls-files`, so an
untracked file is silently absent from every Colab bundle. It is committed now.
The rule this re-proves: **commit before `make_zip.sh`**, and treat a changed
collection count as a stale-bundle signal, not noise.

Two breaks in the notebook were found afterwards and fixed in `31c481f`: a
Python string literal split across raw newlines in Step 11 (SyntaxError before
any encoding), and a Step 2 import of the since-removed `migrate_legacy`.
Neither could have been caught by the 2026-08-04 run, because that run used
the older bundle in which neither line existed yet.

## 2026-08-04: Phase 1.2c exit test PASSED on Colab T4, all five encoders

The re-run that closed the one piece of never-executed code. Numbers are shape
and integrity checks, not results: no quality comparison happens before
`probes/cv.py`.

Environment: Tesla T4, torch 2.11.0+cu128, Python 3.12.

### D per encoder, all five verified against real weights

| encoder | D (pooled) | grid | variants |
|---|---|---|---|
| `raw_features` | 35 | 4x4 x 35 | - |
| `imagenet_vit_b16` | 1536 | 4x4 x 768 | cls_last 768, patch_mean_last 768 |
| `dinov2_vitb14` | **3840** | 4x4 x 768 | cls_last 768, cls_last4_concat 3072, patch_mean_last 768 |
| `satlas_s2_swinb_rgb` | 1024 | 4x4 x 1024 | - |
| `satlas_s2_swinb_mi_rgb` | 1024 | 4x4 x 1024 | - |

**`dinov2_vitb14` ran against real weights for the first time here.** Its
extraction was rewritten in `b84c484` to follow the published linear-probe
protocol and could not be executed locally (its hub code needs Python >= 3.10;
the dev venv is 3.9.6), so until this run the most structurally complex wrapper
in the roster was also the only one never watched running. It produced exactly
the declared shapes: `cls_last4_concat` 3072, `pooled` 3840, `grid` (14, 16, 768)
off a 16x16 patch lattice.

### Gates

```
Step 4   111 passed, 5 skipped        (was 95/3 before 1.2b/1.2c)
Step 7   T_kept = 14/29 on cube 1, identical across all five encoders
Step 8   20/20 cubes, worst prevalence 1.89e-05 vs 1e-04 tolerance
Step 10  peak GPU 1.56 GB at T=290, batch_size=16   (budget 12 GB)
Step 11  100 .npz, 264 frames per encoder, T_kept min 10 median 13
```

Peak GPU rose 1.20 -> 1.56 GB because the ViT probe default is now 1536-dim
rather than 768. `window_span_days` on cube 1: min 0, median 38, max 85 days.

### One thing this run did NOT do, and the guard added because of it

Step 11 reported **100 cached, 0 encoded**: the artefacts already existed from a
prior execution, so this pass re-asserted them rather than recomputing them.
Re-assertion is real verification -- every one of the 100 files passed
`load_encoded`, including the fp16 grid check and the
`grid_clear_frac.mean == clear_frac` identity -- but it left one question open.

Encoder dimensionality does **not** prove a cache is current. The multi-image
encoder landed in `d58e98e` and `window_span_days` in `a1a6a12`, a later
commit, so a cache can hold MI files at exactly the right D and still predate
the covariate. `np.load` reports a missing key as simply absent, and
`load_encoded` returned `None` for it and continued. A probe would then read
`window_span_days`, find nothing, and quietly drop the confound control.

Fixed by stamping `SCHEMA_VERSION` (now 3) into every `.npz` and refusing to
load anything older, with the reset command in the message. Version 1 was
Phase 1.2, 2 added the grid, 3 added `window_span_days`.

## 2026-08-03: Phase 1.2c re-encode, and the MI lookback measured

Local CPU run, all 20 cubes. Four of the five encoders: `dinov2_vitb14` runs on
Colab only, because the `facebookresearch/dinov2` hub code evaluates PEP 604
unions at import and needs Python >= 3.10 (this venv is 3.9.6).

```
build              52 s      3 network wrappers, weights already cached
re-encode         150 s      CPU, 4 encoders x 20 cubes
embeddings       27.3 MB     80 .npz  (pooled fp32 + grid fp16 + variants
                             + grid_clear_frac + window_span_days)
per-cube masks    0.074 MB   20 .npz
manifest          264 rows   strata green
```

### window_span_days: the MI lookback is NOT a fixed window

264 MI embeddings, calendar days spanned by each 8-retained-frame window:

```
min 0   p25 25   median 55   p75 80   max 105
```

If every frame sat at the 5-day Sentinel-2 revisit, 8 frames would span **35
days**. The median is **55** and the maximum is **105**, three times nominal.
The spread between quartiles alone is 25 to 80 days.

This is the measurement that justifies caching it. The lookback is not a
nuisance of a few days around a nominal value; it is a covariate varying over
a 105-day range, and it varies **with cloud**, because a cloudier stretch drops
more frames and the same 8 retained frames therefore reach further back. Any
P2/P3 comparison involving the MI encoder that does not condition on
`window_span_days` is comparing embeddings built from different amounts of
history, with the difference correlated with weather. `min = 0` is the first
frame of each cube, which has no history to look back on.

Single-image encoders store `window_span_days = 0` exactly, an honest constant
rather than a NaN.

### Projection to seasonal scale, five encoders (~290 frames per cube)

```
  200 cubes    7.5 GB    ~689 min CPU
  500 cubes   18.8 GB   ~1722 min CPU
 1000 cubes   37.6 GB   ~3443 min CPU
```

At 1000 cubes the store is ~38 GB, comfortably past a default Drive allowance.
The grid is already fp16, so the lever remains cube count, not precision. GPU
cuts the time roughly an order of magnitude; the disk figure is
device-independent.

## 2026-08-03: Phase 1.2b dimensionalities, sizes and timings

Local CPU run over all 20 cubes of tile 32UNU. Clay deferred (see DECISIONS),
so three encoders here, not five.

### Feature variants and dimensionality

| encoder | variant | dim |
|---|---|---|
| `raw_features` | pooled (35 whole-frame stats) | 35 |
| | grid, per cell | 35 (x16 = 560) |
| `imagenet_vit_b16` | cls_last | 768 |
| | patch_mean_last | 768 |
| | **pooled** = concat of the two | **1536** |
| | grid, per cell (14x14 -> 4x4) | 768 |
| `dinov2_vitb14` | cls_last | 768 |
| | cls_last4_concat | 3072 |
| | patch_mean_last | 768 |
| | **pooled** = concat(cls_last4_concat, patch_mean_last) | **3840** |
| | grid, per cell (16x16 -> 4x4) | 768 |
| `satlas_s2_swinb_rgb` | **pooled** = GAP of final stage | **1024** |
| | grid, per cell (4x4 map, pooling is identity) | 1024 |

Satlas has NO CLS token and therefore no `cls_*` variant; Swin is hierarchical.

### Grid round-trip (the classic patch-token reshape bug)

`grid.mean(cells)` versus the whole-frame patch mean:

```
16x16 -> 4x4  divisible      max abs diff 5.96e-08   (DINOv2)
 4x4  -> 4x4  identity       max abs diff 5.96e-08   (Satlas, exact)
14x14 -> 4x4  NOT divisible  max abs diff 8.50e-03   (ViT-B/16)
```

The ViT gap is geometry, not a bug: 14 does not divide by 4, so
`adaptive_avg_pool2d` uses uneven bins and the cell mean is a *weighted* patch
mean. Pinned by `test_uneven_lattice_explains_its_own_mismatch`.

### Sizes and timings, 20 cubes / 264 retained frames

```
re-encode          52 s        CPU, 3 encoders, 20 cubes
embeddings         18.5 MB     60 .npz (pooled fp32 + grid fp16 + variants)
per-cube masks      0.07 MB    20 .npz, bool (T_kept, 128, 128)
mask compression   ~146x       vs packed bits; clear-fraction is bimodal
manifest           264 rows    strata: cropland 8, grassland 6, tree_cover 6 cubes
```

### Projection to seasonal scale (~290 frames per cube)

```
  200 cubes    4.1 GB     ~192 min CPU
  500 cubes   10.1 GB     ~479 min CPU
 1000 cubes   20.3 GB     ~958 min CPU
```

**Size the seasonal prototype against Drive before running it.** At 1000 cubes
the grid store alone is ~20 GB, which exceeds a default Drive allowance; the
grid is already fp16, so the lever is cube count, not precision. GPU will cut
the time roughly an order of magnitude; the disk figure is device-independent.

## 2026-08-03: Phase 1.2 exit test PASSED, Colab T4, 20 cubes

`notebooks/phase1_2_encoders.ipynb` run clean end to end in a fresh runtime.
These are shape and integrity numbers, not results: **no quality comparison
happens in Phase 1.2, and any number produced outside `probes/cv.py` does not
exist.** The embeddings are inputs to Phase 1.3, nothing more.

Environment: Tesla T4 15.6 GB, torch 2.11.0+cu128, torchvision 0.26.0+cu128,
Python 3.12.

### Embedding dimension per encoder

| encoder | what | params | D |
|---|---|---|---|
| `raw_features` | not a network | 0 | **35** |
| `imagenet_vit_b16` | torchvision ViT-B/16 | 85.8M | **768** |
| `dinov2_vitb14` | torch.hub DINOv2 ViT-B/14 | 86.6M | **768** |
| `satlas_s2_swinb_rgb` | SatlasPretrain Swin-B | 87.9M | **1024** |

All three networks reported `eval()=True, requires_grad=False` on every
parameter at construction, re-asserted on every `encode` call.

### Frame retention, 20 cubes

```
T_kept       min 10  median 13  max 16      (of T = 28-30)
frames total 264 per encoder, 80 cube x encoder pairs
distribution 10:1  11:1  12:4  13:6  14:4  15:3  16:1
```

`T_kept` reproduces Phase 1.1's `frames >50% clear` counts (min 10, median 13,
max 16) exactly, so neither the finite-mask correction nor anything else in
Phase 1.2 changed a frame's keep/drop decision. All four encoders returned
identical kept timestamps on every cube.

### Valid-pixel reflectance, 20 cubes

All 20 pass. Worst prevalence **1.89e-05** against the 1e-4 tolerance; 12 of 20
cubes have zero implausible valid pixels. The all-finite maximum reaches
**1.9817**, confirming Phase 1.1's 1.98 figure and confirming it stays behind
the mask: the valid maximum on that same cube is 1.7839, from 5 isolated
pixels.

### Other gates

- Unit tests **95 passed, 3 skipped** (the 3 download weights inside pytest).
- Malformed input refused loudly by all four wrappers: rank-3, channels-last,
  empty batch, plus the baseline refusing to run without the mask.
- Memory does not scale with T: **peak GPU 1.20 GB at T=290, batch_size=16**,
  against the 12 GB T4 budget. The seasonal split (~290 frames) fits with an
  order of magnitude to spare.

### Cross-machine reproducibility

The per-cube valid maxima and implausible-pixel counts from the T4 run match a
local CPU run of the same code exactly (0.8235/0, 1.7839/5, 1.4655/4,
0.7446/0, 1.4637/1, ...), as do `T_kept` and the 264-frame total. DINOv2 ran on
Colab; it is skipped locally only because `facebookresearch/dinov2` hub code
evaluates PEP 604 unions at import and needs Python >= 3.10.

## 2026-08-03: two encoder-input data properties, tile 32UNU, 20 cubes

Both found by Phase 1.2's own assertions firing on real cubes. Measured over
all 20 cubes: 17,340,401 valid pixels, 264 retained frames.

### Implausible reflectance among VALID pixels

| quantity | value |
|---|---|
| valid pixels with reflectance > 1.2 | **44** (2.537e-06) |
| worst single cube | 15 px, 1.89e-05 |
| cubes affected | 8 of 20 |
| max valid-pixel reflectance | 1.7839 |
| per band (B02, B03, B04, B8A) | 7, 7, 12, 18 |

They are **isolated singletons and 2-4 px clusters in 99.7-100% clear frames**,
not contiguous regions. At the worst pixel `s2_dlmask=0`, `s2_SCL=5` (bare
soil), while the legacy `s2_mask=1` (cloud). Bright across all visible bands
(B02 1.78, B03 1.63, B04 1.55): the signature of a specular target such as a
greenhouse or metal roof, of which the Allgau has many, or a sub-pixel bright
fragment.

Downstream impact, measured rather than assumed:

- NDVI at those pixels stays in **[-0.19, 0.72]**; NDVI over all 20 cubes stays
  inside [-1, 1]. A pixel bright in both B04 and B8A gives NDVI near 0, not a
  wild value.
- Recomputing the raw-feature baseline with those pixels masked moves no
  feature by more than **3.9e-04 relative** (worst: `NDVI_p10`).

**Adopted.** The check asserts on **prevalence, not the maximum**. A maximum
over 17M pixels is the most outlier-sensitive statistic available; the failure
worth halting for is cloud passing as clear over real area, which for even one
fully-clouded frame of a 13-frame cube would be ~7.7e-02. Tolerance set at
**1e-4**, about 5x the worst observed cube and ~770x below the smallest
systemic leak worth the name. The threshold of 1.2 for "physically implausible"
is unchanged.

### Pixels that are both mask-valid and no-data

| quantity | value |
|---|---|
| pixels mask-valid AND non-finite | **113** |
| non-finite pixels in retained frames | 117 (6.762e-06) |
| retained frames affected | 6 of 264 (2.3%) |
| worst frame's non-finite fraction | 0.0014 |

GreenEarthNet's clear-sky conjunction is computed from `s2_dlmask` and `s2_SCL`
and never consults the reflectance bands, so it can mark a no-data pixel clear.
`data.ndvi.ndvi` already guarded against this internally (`usable = mask &
isfinite(...)`); the encoder path did not, and a single NaN returns an all-NaN
ViT embedding.

**Adopted.** `encoders.frames.finite_valid_mask` ANDs the mask with
"every band finite", so the encoder path and the target path agree about which
pixels exist. Pixels only ever leave the valid set; nothing is filled. Frame
selection is unchanged in effect: `T_kept` is min 10, median 13, max 16, which
reproduces Phase 1.1's `frames >50% clear` counts exactly, so no frame changed
its keep/drop decision.

For the three network wrappers, which have no concept of a mask and cannot
consume NaN, non-finite pixels are replaced by a printed sentinel
(`NONFINITE_FILL = 0.0`) inside each wrapper. 6.8e-06 of band-pixels, at most
0.14% of any one frame. The raw-feature baseline needs no sentinel: its
statistics are NaN-aware.

## 2026-08-02: mask switch, s2_mask to s2_dlmask

Tile 32UNU, 20 cubes, 9,486,336 finite pixels.

Clear-sky fraction over all finite pixels, three definitions:

| definition | clear fraction |
|---|---|
| `s2_mask == 0` (legacy sen2flux/Sen2Cor lineage) | 0.4263 |
| `s2_dlmask == 0` (variable switch alone) | 0.4815 |
| `s2_dlmask == 0` AND `s2_SCL in (1,2,4,5,6,7)` (**adopted**) | 0.4570 |

The SCL conjunction removes a further 5.1% of dlmask-clear pixels.

**Correction to the figures quoted when this switch was requested.** The
0.366 -> 0.509 shift was measured on a single cube, the first one listed. Over
all 20 the shift is **0.4263 -> 0.4570**, a relative gain of +7.2% in usable
pixels, not +39%. Effective sample size per probe therefore rises modestly.

Per-cube under the adopted definition:

```
valid fraction    min 0.372  median 0.455  max 0.555
frames >50% clear min 10     median 13     max 16     (of T ~= 29)
NDVI valid frac   median 0.455
NDVI median       min 0.592  median 0.747  max 0.822
```

Snow: `s2_dlmask` has no snow class (0 clear, 1 thick cloud, 2 thin cloud,
3 cloud shadow). Snow is masked through the SCL half instead, since 11 is not
in the allow-list. Under the legacy mask snow was only 0.00054 of finite pixels
in this tile and window, so the practical effect here is negligible, but it is
the reason the conjunction is not optional.

Evidence for the lineage claim is in the cubes themselves, not only the paper:

- `s2_dlmask.attrs["description"]`: "Deep Learning Cloud Mask, trained by Vitus
  Benson on cloudSEN12, leveraging code from Cesar Aybar."
- `s2_mask.attrs["description"]`: "sen2flux Cloud Mask", with class 4 being
  "masked by SCL".

## 2026-08-02: adopted climatology definition

Per-pixel NDVI climatology, adopted verbatim in behaviour from
`model_pixelwise/climatology.py` in vitusbenson/greenearthnet:

> For a target year, pool NDVI from **all other years**, keep pixels where
> `s2_dlmask == 0` and `s2_SCL` is in `(1,2,4,5,6,7)`, linearly interpolate
> remaining gaps along time, group by day-of-year and average across years,
> then smooth with a 30-day centred rolling mean over a wrap-padded year and
> drop the 30 padding days at each end.

Two notes on our implementation in `data/climatology.py`:

- NDVI comes from `data.ndvi.ndvi`, which returns NaN where
  `|B8A + B04| < 1e-12`. GreenEarthNet instead adds `1e-8` to the denominator.
  The two agree everywhere the denominator is not near zero; ours declines to
  guess where it is.
- **Not computable on the current subset.** Every cube in 32UNU is from 2018,
  and the definition requires at least one year other than the target year.
  `ndvi_climatology` raises `SingleYearError` rather than inventing a
  within-year substitute. It becomes available at scale-up.

## 2026-08-02: year-grouped CV deferred

`probes/cv.py` keeps a year-grouped mode, but it raises `SingleYearError` when
the manifest contains one year, which is the case for the whole 32UNU subset.
The year-leakage check is therefore **deferred to scale-up**. Spatial grouping
is unaffected and remains the operative split for Phase 1.3.

The error message explicitly warns against falling back to a random split,
which would put the same season on both sides and produce exactly the inflated
number the year grouping exists to prevent.

## 2026-08-02: probe scope under the single-year subset

Recorded in the `probes` package docstring so it is read where it binds:

- **P2 (5-day deltas): unaffected.** The step is the S2 revisit, well inside a
  150-day cube.
- **P3 (horizons to ~100 days): unaffected.** A 150-day window holds a 100-day
  horizon with context to spare.
- **Ceiling claim: narrowed to "within-season".** No interannual or multi-year
  seasonality on this subset. Conventional for the benchmark, but it has to be
  written down rather than assumed.

**PCA latent clock.** Not computable from one cube: ~29 frames over ~150 days is
a fragment of an annual cycle, so the "clock" would be a line rather than a
loop. Adopted instead: pool embeddings across cubes spanning the tile's 16
distinct time windows (starts 2018-03-09 through 2018-07-07, aggregate coverage
2018-03 to 2018-12) and colour by month or season. The spread across windows is
the only seasonal axis available.

## 2026-08-07: EO-WM authors replied, K1 fired

Full correspondence in
[docs/correspondence/2026-08-07-eowm-authors.md](docs/correspondence/2026-08-07-eowm-authors.md).
Persistence is not a main-table baseline in EO-WM (Appendix A.1 only); their
Earthformer row is a self-trained 200-epoch baseline, not an official
checkpoint; they use EarthNet2021-era masks, not the `s2_dlmask` + `s2_SCL`
allow-list this repo uses; `era5_climatology_all.pt` is a 5-channel weather
climatology, not an NDVI climatology, so it does not unblock P4. K1's
pre-registered condition fired: see
[docs/DECISIONS.md](docs/DECISIONS.md#2026-08-07-k1-fired----eo-wms-published-rows-are-not-a-validation-surface).
