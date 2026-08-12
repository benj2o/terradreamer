# Convergence — what the four probes were for, and what they found

State of the repo as of **2026-08-12**, after Phase 1.8 (P3). One page, the
format research_plan_v3 §5.4 asks for. Every number is pulled from the probe
that owns it — `docs/HANDOFF_P2.md`, `docs/HANDOFF_P3.md` and `log.md` — and
nothing is re-derived here.

**One sourcing caveat, stated up front.** `research_plan_v3` is not in this
checkout. The "prior" column below quotes each probe's pre-registered
expectation from where the repo records it — the probe's own module docstring,
`docs/DECISIONS.md`, or `log.md` — and every row names that source. Where the
plan's §5.4 wording is not recoverable from the repo, this page says so instead
of paraphrasing it.

---

## 1. The four probes

| probe | prior (research_plan_v3 §3, as recorded in this repo) | measured result | verdict |
|---|---|---|---|
| **P1** appearance | An EO model trained to appearance-invariance would FAIL to decode month/season from one frame, and that failure would be the finding. Success is a floor being met, not a result. *(source: `probes/p1_appearance.py` docstring)* | **No encoder fails.** All five clear chance widely on month and season under all three fold modes. Given the same three bands, the networks beat the hand-crafted baseline by **+0.02 to +0.06** on month/cube. The degenerate control `[clear_frac, window_span_days]` — two numbers, no image — sits at or above **42 of 120** grid-cell rows, and under `spatial_block` it beats season on **17 of 20**. *(log.md 2026-08-09)* | **Floor met, surprise did not occur.** P2/P3 licensed. Cite month under `cube` or `loco`, and cite the margin over the control, never the distance from chance. |
| **P2** dynamics in deltas | Augmentation-invariance training (DINOv2) discards state-change MORE than reconstruction-style training. *(source: `docs/DECISIONS.md` 2026-08-11, quoting research_plan_v3 §3/P2)* | Gate K2 passed, **3 of 4 networks separably beat** the hand-crafted baseline; **no encoder excluded**. **SIGN is recoverable and robust** — DINOv2 ρ = **+0.606 [+0.546, +0.665]**, margin **+0.54** over the gap-length control, moving < 0.05 across all three fold modes. **MAGNITUDE is not** — every ρ ≤ **+0.12**, margins ≤ +0.07, and 3 of 4 encoders flip margin sign across fold modes. The band-matched hand-crafted row `raw_rgb_only` reaches **+0.695**, above every network. *(HANDOFF_P3 §6, log.md 2026-08-11)* | **Structural hypothesis REFUTED**, stably: the ordering is `raw_features > dinov2 > satlas_SI > imagenet` under all three fold modes, the opposite of the stated direction. Direction is available in these representations; **rate is not**. |
| **P4** weather-attributability ceiling (H1) | H1 is the fraction of post-climatology NDVI anomaly explainable from weather alone, and it is P3's denominator. *(source: `probes/p4_ceiling.py` docstring)* | Weather explains **0.130 [+0.063, +0.197]** (cube folds) and **0.085 [+0.007, +0.162]** (LOCO) of the within-season post-**proxy**-climatology anomaly at `cell_mean` under HGB — the first cell in the project to clear zero and both controls at once (margin over the observation control +0.137 / +0.109; over the DOY control +0.094 / +0.086). `spatial_block` kills it. Four fifths of the best-*looking* number (`cube_p90` HGB +0.320) is day-of-year. *(log.md 2026-08-11)* | **Measured, but as a PROXY.** H1 **as originally scoped is not obtainable on this benchmark**: tile 32UNU has no seasonal-split coverage, and on a tile that has it the fold count clamps to the number of years (4). Stage B remains deferred and its second-order leak path is still unquantified. |
| **P3** forecastability | Does a cheap read-out over frozen embeddings plus weather forecast future NDVI, and does it beat persistence and climatology? *(source: this phase's spec; `probes/p3_forecast.py` docstring)* | **Yes, and it clears every control.** Best pooled out-of-fold R² **+0.672 / +0.704 / +0.454 / +0.628** at Δ = 5 / 25 / 50 / 100 d (`cube_mean`, ridge, cube folds), against an observation control ≤ **+0.056**, a horizon-alone control negative at every horizon, and a permutation null at −0.014 to −0.103. Skill against persistence **+0.605 / +0.549 / +0.459 / +0.573**. Survives `spatial_block` (DINOv2 +0.554 → **+0.488 [+0.433, +0.543]** at 100 d). **But the band-matched hand-crafted row is within 0.03 at 5 and 25 d and AHEAD at 100 d**, and full `raw_features` + weather wins outright at 3 of 4 horizons. *(log.md 2026-08-12)* | **The capability is established; the attribution to foundation models is not.** And on the extreme tails no row beats persistence at any horizon. |

---

## 2. P3's row, in full

**What was asked:** does a cheap read-out over frozen EO embeddings plus the
weather over the horizon forecast future NDVI, and does it beat
persistence/climatology — overall *and* on the extreme/dynamic subset?

**What was measured**, 115 cubes, tile 32UNU, 2018, `cube_mean` target, ridge
read-out, cube folds, pooled out-of-fold R² with a delete-one-fold jackknife
interval, effective n in CUBES:

```
                              D=5      D=25     D=50     D=100     effective n
raw_features + weather      +0.672   +0.704   +0.368   +0.628      115/114/115/94
best frozen network         +0.486   +0.597   +0.454   +0.569
raw_rgb_only + weather      +0.464   +0.566   +0.329   +0.588   <- BAND-MATCHED
persistence                 +0.169   +0.344   -0.009   +0.129
proxy climatology           +0.021   +0.130   +0.104   -0.164   <- proxy, NOT Stage B
observation control         +0.016   +0.056   +0.039   -0.031
horizon-alone control       -0.004   -0.007   -0.006   -0.023
permutation (empirical 0)   -0.037   -0.035   -0.014   -0.103
```

**Four things this row establishes, and one it does not.**

1. **Forecast skill is real and it is not an artefact of the observation
   process or the calendar.** Every model clears all three controls at every
   horizon, and the permutation null is where a null should be.
2. **It survives the strictest geography holdout.** `spatial_block` costs 0.04
   to 0.08 and the intervals still exclude zero at 25, 50 and 100 days. P1 and
   P4 both collapse there; P2 survived; P3 survives. This was UNKNOWN going in
   and both precedents existed.
3. **The window boundary is a benchmark limit, not a nuisance.** 518 → 489 →
   450 → **196** rows as Δ goes 5 → 100, and **21 of 115 cubes contribute no
   100-day pair at all**.
4. **The extreme subset does not follow the headline.** At 5 and 25 days every
   network is *negative* against persistence in `extreme_low` and `high`; at 50
   and 100 days they turn positive on `extreme_low` and negative on
   `extreme_high`, to −3.14. **No row beats persistence on both tails at any
   horizon.**

**What it does not establish: that frozen foundation models are the way to do
it.** Seven percentiles of the same three bands, with no NDVI column and no
network, are within 0.03 of the best encoder at 5 and 25 days and ahead of it at
100. Full `raw_features` + weather — legitimately autoregressive here, since the
target is at t+Δ — wins outright at 3 of 4 horizons. This is P2's band-matched
result reproduced on a second target.

**Two caveats that travel with every number above.** Three cloud-contaminated
frames of 1580 carry **71%** of the persistence sum of squares at 5 days
(`sse_share_top1pct` is on every row); and the proxy climatology **collapses
under `spatial_block`** (−8.6, −1328, −2784), so the climatology *baseline* is
not usable under that mode even though the encoder rows are.

---

## 3. Recommendation among H1 / H2 / H3

**A sourcing statement first, because it changes what this section can honestly
say.** `research_plan_v3` is not in this checkout. **H1** is defined in the repo
(`probes/p4_ceiling.py`: the fraction of post-climatology NDVI anomaly
explainable from weather alone). **H2 and H3 are not** — the only trace of H3 is
one line in `docs/DECISIONS.md` (2026-08-07) contemplating "their
reconstruction-diagnostic conventions if H3 reaches EarthNetScore", which
suggests a full forecasting/reconstruction-quality hypothesis but does not
define one. So §5.4's pre-registered *rules* cannot be applied by name from this
repo. What follows is the recommendation the **evidence** supports, stated in
plain language, with the decision each rule would turn on made explicit so it
can be checked against the plan in one reading.

**Recommendation: report H1, in its proxy form, as the paper's quantitative
spine — and demote the foundation-model framing from a claim to a comparison.**

1. **H1 is the only one of the three this benchmark can carry, and even it is a
   proxy.** P4 measured 0.130 `[+0.063, +0.197]` of the within-season
   post-proxy-climatology anomaly at `cell_mean`, clearing both its controls,
   and P3 now sits *above* that denominator on its own target — so the two
   numbers are coherent and mutually interpretable. But **H1 as originally
   scoped is not obtainable here**: tile 32UNU has no seasonal-split coverage,
   and on a tile that has it the fold count clamps to the number of years (4).
   **Every H1 number in the paper must carry the proxy label**, and Stage B's
   second-order leak path must be quantified before any leave-year-out number is
   quoted. *That open item is unchanged by this phase.*

2. **Do not build the paper on "frozen EO foundation models forecast
   vegetation."** Two independent probes, on two different targets, now say the
   same thing: on P2's delta-sign probe the band-matched hand-crafted row beat
   every network (+0.695 vs +0.606), and on P3's forecast target it is within
   0.03 at two horizons and ahead at a third. The defensible claim is the
   weaker and more useful one: **a cheap read-out over a frozen representation
   plus weather forecasts NDVI well above persistence and climatology, and
   hand-crafted band statistics do so about equally well** — which is a result
   about what is *available* in Sentinel-2 RGB plus E-OBS, not about
   representation learning.

3. **Whatever H2 turns out to be in the plan, the two axes that would decide it
   are already measured and both are negative for capacity.** The MLP is
   unusable at this width (D/n up to 73, pooled R² −324) while ridge at α = D is
   fine; and P2 settled that no encoder recovers change *magnitude* (ρ ≤ 0.12,
   sign-flipping across fold modes). **If H2 depends on more model capacity or
   on rate rather than direction, the evidence is against it at this sample
   size**, and more cubes will not fix the magnitude half — P2 scaled 5.75× and
   it got worse, not better.

4. **H3, if it is the EarthNetScore-level forecasting hypothesis the
   correspondence note implies, is not reachable from this checkout.** It would
   need pixel-level prediction, and every target in P1/P2/P3/P4 is spatially
   aggregated by design — never pixel-wise. That is a scope decision made four
   phases ago and this phase gives no reason to revisit it.

**The one sentence to put in the abstract, if only one fits.** On 115
GreenEarthNet minicubes of tile 32UNU, a ridge read-out over frozen EO
embeddings plus the weather over the horizon predicts cube-mean NDVI 5 to 100
days ahead at pooled out-of-fold R² 0.41–0.60, clearing persistence, a proxy
climatology, an observation-process control and a permutation null, and
surviving a geographic block holdout — **but a 21-column summary of the same
three bands does as well, and no configuration beats persistence on the extreme
tails.**

**Verify against research_plan_v3 §5.4 before publishing.** Points 1 and 2 rest
only on measurements in this repo. Points 3 and 4 name H2 and H3, whose
definitions are not in this checkout; if the plan defines them differently, the
*measurements* cited there still stand and only the mapping changes.

---

## 4. What the next phase inherits

| | state |
|---|---|
| `data/scaled_32UNU/{raw,embeddings,masks}` | 115 cubes, 575 `.npz`, 115 mask files. **The set P2, P4 and P3 were all measured on.** Do not add cubes to `data/raw`. |
| `probes/p3_forecast.py` | New. The forecast probe, its five baselines, its three controls, and the horizon index P4/P2 do not have. |
| `probes/cv.py` | Unchanged. `cube`, `loco`, `spatial_block` run; `year`, `tile`, `crossed` correctly RAISE on this single-tile, single-year subset. |
| tests | **0 failed.** The count grows every phase; the invariant is 0 failed. |

**The three open items that outlive this phase**, in the order they would change
a published number:

1. **Three cloud-contaminated frames are inside every probe's target.** Frames
   with a common-masked cube-mean NDVI below zero in midsummer, at clear
   fractions of 0.59–0.63, passed both the clear-fraction filter and the
   per-pixel mask. They carry most of P3's short-horizon squared error, and they
   are in `p4_ceiling.cube_frame_targets`, which P2 and P4 read too. A physical
   plausibility screen belongs in that shared function, not in one phase.
2. **Stage B's climatology has an unquantified second-order leak path**, and no
   H1-style number may be quoted from it until that is measured
   (`probes/p4_ceiling.run_stage_b`, DECISIONS 2026-08-11).
3. **P4's proxy climatology is still unvalidated**
   (`scripts/validate_proxy_climatology.py`).
