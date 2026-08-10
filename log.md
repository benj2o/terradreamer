# Working log

Running record of measurements and adopted definitions. Reverse chronological.
Decisions and their rationale live in [docs/DECISIONS.md](docs/DECISIONS.md).

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
