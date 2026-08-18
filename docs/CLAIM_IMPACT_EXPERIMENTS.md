# Claim / Impact Experiments Strategy Memo

Date: 2026-08-13

This memo is about **what to do next to maximize claim strength, climate impact, and CCAI acceptance odds before the 2026-08-29 AoE deadline**. It is intentionally not a paper draft.

## Executive take

The current repo already supports a publishable **ground-up evaluation** story, but not the originally implied **foundation-model advantage** story:

- P3 establishes real forecast skill, but the attribution to frozen EO foundation models is refuted on the current evidence.
- P2 says direction of change is recoverable, magnitude is not.
- P4 gives a useful within-season proxy ceiling, but true H1 is not obtainable on `32UNU`.

For CCAI 2026, that is not fatal. In fact, it can fit the workshop **better** than a standard NeurIPS-style model-comparison paper, because the call explicitly welcomes **smaller/hybrid models, open tools, evaluation, practical constraints, scalability, and impact**. The problem is not that the current result is negative; the problem is that **without one more climate-facing move, it still reads too much like a one-tile FM comparison that happened to end negatively**.

My recommendation is:

1. **Do not spend the remaining 16 days on a full 20x many-tile Tier-1 sweep.** The compute is more feasible than previously assumed, but the real risk is integration, new failure modes, and writing time.
2. **Do spend the remaining time on one integrity fix and one climate-facing extension.**
3. The best pre-deadline climate-facing extension is **an extreme-tile result**, ideally with **trigger- or event-oriented metrics**, because it raises both climate relevance and workshop fit without requiring unreleased EO-WM training code.

The highest-value paper framing is no longer:

> frozen foundation models forecast vegetation well

It is closer to:

> a resource-light, open, auditable benchmark-and-baseline stack can forecast vegetation stress and diagnose what weather-driven signal is actually available; in this setting, simple tabular baselines match or beat frozen EO foundation models, and climate-facing evaluation matters more than raw FM branding.

## A. Tile / Geography Strategy

### What the repo and bucket actually offer

From the repo and direct bucket inspection:

- `train` split covers **85** tiles.
- `extreme` split covers **4** tiles: `32UMC`, `32UNC`, `32UPC`, `32UQC`.
- `seasonal` split covers **15** tiles:
  `29SQC`, `29TPF`, `29UMV`, `30TVK`, `30TVN`, `31TCF`, `31TFK`,
  `31UCS`, `31UEQ`, `31UGQ`, `31UGU`, `32VNM`, `32VPN`, `33TWN`, `33VXK`.

Non-overlapping cube capacity under the repo's 64 px rule:

- `32UNU/train`: **192 listed -> 115 non-overlapping**
- `32TPT/train`: **41 -> 23**
- `33TUN/train`: **38 -> 29**

Extreme tiles:

- `32UMC`: **670 -> 246**
- `32UNC`: **1014 -> 315**
- `32UPC`: **1076 -> 312**
- `32UQC`: **1212 -> 348**
- Extreme total: **1221** non-overlapping cubes

Seasonal tiles:

- total: **1439** non-overlapping cubes across 15 tiles
- largest audited examples:
  - `30TVK`: **296 -> 114**
  - `30TVN`: **299 -> 87**
  - `31TCF`: **289 -> 108**
  - `31UGU`: **293 -> 117**
  - `29SQC`: **282 -> 117**

Combined `extreme + seasonal` non-overlapping capacity is **2660 cubes**, which is already **more than 20x** the current `115`-cube scaled set (`20 x 115 = 2300`).

### Temporal structure matters more than raw count

- `32UNU/train` is **single-year 2018**, with **16** within-year windows.
- `extreme` tiles appear to be **single-year 2018 event windows**. In the sampled listings, each tile used one long 2018 window.
- `seasonal` tiles are **multi-year**. Example: `30TVN` spans **2017-2020** inside one cube family.

That means:

- `extreme` is excellent for **climate relevance** and event stress, but **does not rescue H1**.
- `seasonal` is necessary for **cross-year response fidelity**, but **still does not make Stage B precise**, because `crossed` mode is bounded by **4 years**, not by cube count.

### Cost model from repo-measured runtimes

Reference runtimes already in the repo:

- Download `32UNU` 115 cubes: **65 s**, **379 MB**
- Encode 115 cubes / 5 encoders on T4 (`phase1_7` runbook): **20-40 min**
- P4 on 115 cubes: **37.2 min** on **7 CPU workers**
- P2 on 115 cubes: **100.3 min** on **7 CPU workers**
- P3 first scaled run on 115 cubes: **41.6 min** on **7 CPU workers**
- P3 Tier-1 rerun on 115 cubes: **173.7 min** on **7 CPU workers**

Storage projections already recorded in `log.md`:

- 5-encoder cache at 1000 cubes: **~37.6 GB**
- therefore 2300 cubes is roughly **~86.5 GB** for RGB embeddings alone
- raw cubes scale from `379 MB / 115`, so 2300 cubes is roughly **~7.6 GB raw**
- adding CIR re-encodes roughly adds another **~80% of the RGB-network cache**

Linear wall-clock projections from the current scripts are crude but useful:

- P4 at 2300 cubes: roughly **12.4 CPU-hours** on one 7-worker node
- P2 at 2300 cubes: roughly **33.4 CPU-hours**
- P3 Tier-1 at 2300 cubes: roughly **57.9 CPU-hours**
- RGB encoding at 2300 cubes: roughly **6.7-13.3 GPU-hours** on one T4

This changes the old conclusion:

- **Compute is not the main blocker anymore if a university cluster is available.**
- **Integration and narrative focus are now the blockers.**

The scripts are already close to tile-portable:

- `scripts/scale_p4.py` takes `--tile` and `--split`
- `scripts/scale_p2.py` and `scripts/rerun_p3_tier1.py` take `--tile` and cache-root assumptions
- the main missing piece for many-tile work is orchestration and consistent cross-tile reporting, not core probe logic

### Geography options: predictability, climate relevance, verdict

#### 1. Stay on `32UNU`

Pros:

- lowest engineering risk
- existing cache, runs, and story are already coherent
- strongest basis for a clean 4-page submission on schedule

Cons:

- weak climate salience compared with extreme/seasonal alternatives
- one tile, one year, Alpine-foreland mowing and cloud confounds
- easy for reviewers to read as "careful but narrow"

Verdict:

- **Keep as the paper's clean baseline surface.**
- **Do not make it the only empirical geography if a climate-facing extension is achievable.**

#### 2. Move to the EO-WM `extreme` tiles

Pros:

- strongest immediate lift in climate relevance
- directly tied to **2018 European heat / drought stress**
- large cube budget: **1221** non-overlapping across 4 tiles
- one-tile pilots are already substantial (`246-348` cubes each)
- best fit to workshop language around **extreme weather**, **adaptation**, and **decision relevance**

Cons:

- still single-year, so no Stage B / real H1
- probably requires a small orchestration layer and careful restatement of what is and is not comparable with EO-WM
- could yield harder forecasting rather than better scores

Verdict:

- **Best THIS-CYCLE geography move for Claim/Impact.**
- Even if the score is lower than on `32UNU`, the paper likely gets stronger because the geography is more climate-relevant.

#### 3. Move to a `seasonal` tile

Pros:

- multi-year, so cross-year response questions become possible
- closer to EO-WM's seasonal matched-pair logic
- several tiles have around `90-117` non-overlapping cubes
- the existing `30TVN` validation already hinted that a more water-limited tile
  can produce a much larger proxy weather signal than `32UNU` (about `+0.192`
  vs `+0.066`, though confounded by fold structure, cube count, and
  `weather_finite6`)

Cons:

- Stage B remains structurally underpowered at **4 years**
- off-`32UNU` weather completeness is messy (`30TVN` already forces `weather_finite6`)
- higher methodological complexity for a short-paper deadline
- easier to burn time on benchmark pathology instead of a claim-moving result

Verdict:

- **Good post-deadline science; weaker pre-deadline trade.**
- Use only if the goal is specifically to build a matched-pair / response-fidelity benchmark, not to rescue H1.

#### 4. Full 20x many-tile expansion

Pros:

- now clearly **feasible** on a cluster
- would answer the obvious "does this generalize beyond one tile?" question in a strong way
- very on-theme if framed as a **resource-light benchmark release**

Cons:

- likely blows up storage, orchestration, auditing, and writing complexity
- multi-tile comparability needs deliberate design: feature sets, fold modes, tile leakage, per-tile versus pooled reporting
- a 4-page workshop paper may not have room to explain the result well

Verdict:

- **Feasible in compute, but not the best use of the final 16 days unless the question is aggressively narrowed.**
- Good after acceptance or for a journal / conference extension.

## B. Experiment Shortlist Ranked by Delta Claim / Impact per Day

Below is the shortlist I would actually consider before 2026-08-29.

### c

**Hypothesis.** The current `32UNU` result becomes much more CCAI-relevant if the paper reports not just pooled `R^2`, but whether a forecast would correctly identify **stress-relevant anomalies or trigger crossings** at useful lead times.

**What to run.** Re-slice existing P3 predictions into operationally legible metrics:

- anomaly below threshold
- bottom-quantile / `extreme_low` hit rate
- lead-time-specific hit / miss / false-alarm rates
- sign-of-anomaly-change or DHR-style directional metrics

Do this first on the existing `115`-cube `32UNU` outputs; if an extreme-tile P3 pilot is run, reuse the same metrics there.

**Data.** Existing `data/scaled_32UNU/p3_tier1_results.csv` plus any stored per-row predictions needed for thresholding.

**Compute.** Low. Mostly analysis code and one light rerun if per-row predictions need to be re-emitted.

**Kill date.** **Aug 15.** If the thresholds look arbitrary or unstable, drop this rather than forcing it.

**How it changes the 4-page paper.** Moves the paper from "generic EO forecasting numbers" toward "would this support an early-warning trigger or not?" That is much closer to the workshop's requested pathway-to-impact framing.

**Risk.** If the threshold is not anchored to something operational, reviewers may read it as a post hoc metric hack. Use explicit language: this is a **decision-oriented diagnostic**, not a claim of operational deployment.

### 2. Re-run P2 and P4 with the `frame_plausible` screen

**Hypothesis.** The same three cloud-contaminated frames that moved P3 strongly may also distort P2 magnitude and P4 ceiling enough that a fully aligned row set materially improves credibility.

**What to run.**

- rerun scaled P2 on `32UNU` with the `frame_plausible` screen applied
- rerun scaled P4 on `32UNU` with the same screen
- diff both against the current published tables

**Data.** Existing `data/scaled_32UNU/{raw,embeddings,masks}`.

**Compute.** Roughly **~2.3 CPU-hours** on the current local baseline:

- P4 ~37 min
- P2 ~100 min

**Kill date.** **Aug 16.**

**How it changes the 4-page paper.** Closes the cleanest reviewer attack: "P3 was corrected for cloud-pathology but P2/P4 were not." It also lets the paper say all core claims are measured on the same screened row logic.

**Risk.** It may not change the headline numerically very much. This is more about **trust** than novelty. I still think it is worth doing.

### 3. Extreme-tile P4 pilot

**Hypothesis.** The weather-attributability ceiling will be more climate-relevant, and possibly larger, on a heat/drought tile than on `32UNU`, making the paper's adaptation story stronger even if H1 remains proxy-only.

**What to run.**

- start with one extreme tile, preferably `32UQC` or `32UNC` because they are the largest non-overlapping sets (`348` and `315`)
- run a P4 Stage-A proxy ceiling first
- if the controls behave sensibly, either:
  - keep the one-tile result as the climate-facing extension, or
  - expand to all 4 extreme tiles as separate runs and meta-report them

**Data.** `extreme` split, one-year 2018 windows, `246-348` non-overlapping cubes per tile.

**Compute.**

- one tile: roughly **1.3-1.9 CPU-hours** by linear scaling from the `115`-cube run
- all 4 extreme tiles: roughly **6.6 CPU-hours** on one 7-worker node
- tile-parallel cluster execution makes this realistically a **same-day** job

**Kill date.** **Aug 18.**

**How it changes the 4-page paper.** Adds a climate-relevant geography without requiring new model training. This is probably the cleanest way to avoid the "one calm-ish tile" criticism.

**Risk.** Still only a proxy ceiling; still one-year. If observation-process or DOY controls dominate again, it adds complexity without improving the claim.

### 4. Extreme-tile slim P3

**Hypothesis.** The current conclusion on `32UNU` may or may not generalize to true heat/drought windows. Measuring that on a climate-relevant split is more important for CCAI than squeezing another decimal out of `32UNU`.

**What to run.** Do **not** start with the full 9-view Tier-1 stack. Start with a slim, decision-oriented table:

- persistence
- `weather_only`
- `[NDVI(t), weather]`
- `raw_rgb_only + weather`
- 1 or 2 frozen encoder views, ideally one strong SI model plus, optionally, MI

Use the same horizons, then add the trigger/event metrics from item 1.

**Data.** One extreme tile first; only expand if the pilot is clean.

**Compute.**

- one extreme tile (`~300` cubes): roughly **1-2 GPU-hours** for RGB cache build on one T4-equivalent, plus a CPU overnight for P3 if run naïvely
- pooled 4-tile extreme split (`1221` cubes): roughly **3.5-7 GPU-hours** RGB encoding and **~31 CPU-hours** for a full Tier-1-style rerun on one 7-worker node

On a small cluster, the 4-tile version is feasible because tiles can be run independently and summarized together.

**Kill date.**

- one-tile pilot: **Aug 21**
- expand to 4 tiles only if the pilot is clearly claim-moving by **Aug 24**

**How it changes the 4-page paper.** If it works, this is the single strongest empirical addition: it turns the paper from a one-tile cautionary FM result into a resource-light climate-stress evaluation paper.

**Risk.** Moderate engineering tax. Also, the result could stay negative. That is acceptable only if the paper is explicitly framed around **resource-light baselines and evaluation**, not "we expected FMs to win."

### 5. Event / onset metrics on top of extreme-tile P3

**Hypothesis.** On heat/drought windows, event-oriented metrics such as onset detection, trough error, or drop-amplitude error may be more decision-relevant than pooled `R^2`, and may reveal behavior hidden by persistence.

**What to run.**

- after an extreme-tile P3 run, derive aggregate analogues of:
  - trough NDVI error
  - drop amplitude error
  - event detection rate by severity bin
- keep them explicitly as **aggregate/cube-level analogues**, not as claimed reproductions of EO-WM's pixel-level metrics

**Data.** Same predictions as item 4.

**Compute.** Low once the predictions exist.

**Kill date.** **Aug 22.**

**How it changes the 4-page paper.** Gives one figure that obviously speaks to drought / heat response rather than to generic forecasting.

**Risk.** If the definitions feel too bespoke, it may look like metric shopping. Keep only if the construction is simple and well motivated.

## Candidates I would kill early

These are real options, but I would **not** spend the pre-deadline window on them unless the shortlist above goes unusually smoothly.

### Kill for this cycle

**Full 20x many-tile Tier-1 rerun.**  
Compute is feasible; schedule risk is not. Storage, orchestration, auditing, and paper compression become the dominant problem.

**Trying to rescue true H1 on a seasonal tile.**  
This benchmark gives you 4 years, not 40. Fixing the second-order leak is scientifically right, but it will not change the workshop paper enough before Aug 29.

**EO-WM / Earthformer reproduction.**  
K1 already fired. EO-WM core code is unreleased and Earthformer is a self-trained 200-epoch baseline. This is not a deadline bet.

**New pixel-level / H3-style forecasting.**  
Would create a different paper. Scope explosion.

**More `32UNU`-only ablations.**  
Unless they are integrity fixes like the plausibility screen, they mostly add detail to the wrong story.

**Immediate CIR re-encode on new geographies.**  
Useful only after a climate-relevant RGB result exists. Right now it is a second-order question.

## C. Revised P(accept)

### My calibrated view

I do **not** think an honest **80-90%** acceptance probability is currently supportable.

The `75 / 122 ~= 61%` ICLR 2024 figure is useful context, but it is not a promise:

- it was across workshop tracks, not necessarily only papers
- acceptance depends heavily on fit to the year's theme
- 4-page papers are especially vulnerable to framing failures

### Scenario estimates

**Scenario 0: submit essentially the current `32UNU` story, but written well.**  
Estimated `P(accept)`: **0.55-0.65**

Why:

- technically careful
- clearly negative but interesting FM result
- already on-theme if framed as resource-light evaluation

What still hurts:

- one tile
- weak direct impact pathway unless explicitly spelled out
- risk of reading like "NeurIPS leftover, climate wrapper added later"

**Scenario 1: current story + explicit impact framing + public code posture + P2/P4 screened reruns.**  
Estimated `P(accept)`: **0.65-0.75**

Why this is materially better:

- removes an avoidable consistency objection
- matches the CFP's emphasis on open tools, smaller/hybrid models, evaluation, and practical constraints
- tells reviewers who uses the result and why

**Scenario 2: Scenario 1 + one clean climate-facing extra result**  
Examples:

- extreme-tile P4
- extreme-tile slim P3
- trigger/event-oriented evaluation that is clearly decision-linked

Estimated `P(accept)`: **0.72-0.82**

This is the best realistic pre-deadline zone.

### What 80-90% would require

To get even close to **0.80**, I think all of the following would need to be true:

1. **The paper must stop reading as an FM dunk.**  
   It must read as a **ground-up climate evaluation / resource-light baseline** paper.

2. **There must be an explicit pathway to action.**  
   Who uses this? For what decision? At what horizon? Using what trigger?

3. **The result surface must extend beyond "one calm tile, one year".**  
   The cleanest way is one extreme-tile result.

4. **The open-science story must be strong.**  
   Public code, scripts, CSVs, benchmark logic, and careful documentation all help at this workshop.

5. **The paper must be humble about H1 and benchmark limits.**  
   Overclaiming on proxy climatology or hiding the Stage-B limitation would damage trust fast.

I do **not** think **0.90** is credible for a workshop paper in this setting. Too much reviewer variance remains, and 4 pages leaves little room to recover if one reviewer dislikes the framing.

### What kills papers at this workshop

From the CFP and past workshop norms, I would expect the main failure modes to be:

- **No impact paragraph, or a vague one.**
- **No clear stakeholder or decision pathway.**
- **Metrics that sound ML-native but not climate-native.**
- **A one-tile result presented as more general than it is.**
- **A paper that looks like a standard FM benchmark writeup with "climate" swapped in afterward.**
- **No public code / weak reproducibility posture when the work is explicitly about evaluation and resources.**
- **Claims that outrun the benchmark, especially around H1 / proxy climatology.**

One-tile is not automatically fatal. One-tile **plus** no impact pathway **plus** a paper that reads like a leftover comparison paper is what becomes fatal.

## D. Recommended Sequence: Next 16 Days vs Post-Deadline

### Recommended 16-day plan

#### Aug 13-15: lock trust and impact framing

1. Re-run P2 and P4 with the plausibility screen.
2. Add trigger-oriented or anomaly-threshold evaluation on existing P3 outputs if it can be done cleanly.
3. Rewrite the paper's core claim around:
   - resource-light baselines
   - open evaluation
   - what signal is actually available for vegetation-stress forecasting
   - who could use it

If items 1-2 do not sharpen the story by **Aug 15**, stop expanding them.

#### Aug 15-18: run one climate-relevant geography pilot

Run **P4 first** on one extreme tile (`32UQC` or `32UNC`).

Why first:

- cheapest climate-facing experiment
- no re-encoding needed
- faster feedback on whether a heat/drought geography really changes the ceiling

If the ceiling is still confounded or boring, do **not** escalate blindly.

#### Aug 18-21: run one slim extreme-tile P3 if the pilot is promising

Use a **small model set**, not the full Tier-1 stack:

- persistence
- `weather_only`
- `[NDVI(t), weather]`
- `raw_rgb_only + weather`
- 1-2 frozen encoder rows

Add event/trigger metrics only if they clarify the result.

#### Aug 21-24: decide whether to pool all 4 extreme tiles

Only expand to all 4 extreme tiles if at least one of these is true:

- the one-tile result materially changes the climate story
- the one-tile result is clearly cleaner than `32UNU`
- the pooled extreme result can still fit into 4 pages without becoming incomprehensible

Otherwise stop and write.

#### Aug 24-29: freeze experiments and maximize workshop fit

- final paper framing
- impact paragraph
- appendix with benchmark limits and controls
- public-code cleanup
- figure/table pruning for a 4-page submission

### Post-29 Aug / journal-extension sequence

After the deadline, the best longer-path agenda is:

1. **Seasonal matched-pair analogue** on audited seasonal tiles
2. **Cross-tile benchmark release** with tile-aware holdouts
3. **Full many-tile sweep** of P3/P4 with consistent feature-set rules
4. **Event-level benchmark package** for stress-response evaluation
5. **If still desired, H3 / pixel-level forecasting** as a separate paper, not as this one

## Bottom line

If the sole question is "can we scale 20x?", the answer is now **yes, probably**.

If the real question is "what most increases acceptance odds by Aug 29?", my answer is:

- **do not spend the remaining time on full many-tile scale for its own sake**
- **do spend it on one integrity fix plus one climate-facing geography or trigger-based result**
- **frame the submission as a ground-up, open, resource-light climate evaluation paper, not as a failed FM paper**

That path gives the best trade-off between scientific honesty, workshop fit, and deadline risk.

## External anchors consulted

- CCAI NeurIPS 2026 CFP: workshop theme, paper-track expectations, impact guidance, open-source encouragement  
  <https://www.climatechange.ai/events/neurips2026>
- CCAI ICLR 2024 proposal text noting `122` submissions and `75` acceptances  
  <https://openreview.net/pdf?id=xVA6REQFGA>
- EO-WM paper and repo: extreme-summer / seasonal-matched-pair benchmark definitions  
  <https://arxiv.org/html/2606.27277>  
  <https://github.com/Luo-Z13/EO-WM>
- VITO Early Drought Prediction Service: operational 10/30/90-day NDVI-anomaly framing and stakeholder mapping  
  <https://vito.be/en/applications/early-drought-prediction-service>
- OCHA drought anticipatory-action guidance: trigger logic and decision relevance  
  <https://reliefweb.int/attachments/3ac23967-f69e-4b5c-a057-efa7ba7267b4/OCHA_AA%20Guidance_DROUGHT_2025.pdf>
- Copernicus EDO/GDO vegetation-stress indicators: FAPAR anomaly as an operational drought-monitoring benchmark  
  <https://www.copernicus.eu/en/european-drought-observatory>  
  <https://drought.emergency.copernicus.eu/data/factsheets/factsheet_fapar.pdf>
