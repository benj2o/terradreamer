# Prompt: draft the CCAI paper (new file, do not edit the existing one)

You are drafting a workshop paper for the NeurIPS Climate Change AI workshop
(non-archival). A prior draft exists at `paper/main.tex`. **Do not modify it.**
Write to `paper/main_v2.tex` and put any new figure code in
`paper/make_figures_v2.py`, writing images to `paper/figures/v2/`. The old draft
is a reference and a fallback, nothing more.

---

## Phase 0 — before writing a single sentence

Produce `docs/PAPER_PREWORK.md` for the author's own use. Not part of the paper.
It has four sections and stays under two pages:

1. **What we actually claim.** Every claim the evidence supports, one line each,
   each with the artefact and the number behind it. If a claim cannot be traced
   to a row in a CSV, it is not a claim.
2. **The chain of reasoning, probe by probe, in the order it happened.** This
   is the most important section and the longest. Read `log.md` (reverse
   chronological, so read it bottom-up for the real sequence) and
   `docs/DECISIONS.md` (append-only, assumed/observed/changed/commit) end to
   end, then reconstruct the actual path: the toy load, P1, P2, P3, P4, the
   scale-up to 115 cubes, the Tier-1 corrections, the trigger metrics, the
   extreme-tile P4 pilot, the extreme-tile P3, the `loco` top-up.

   For **each** stage, in chronological order, one short block:

   - what question it was supposed to answer
   - what was assumed going in
   - what it measured
   - what that forced to change: a definition, a control, a fold mode, a
     baseline, the scope of a claim
   - what it made possible or ruled out next

   The point is that a reader of this file can follow how the project arrived
   at its conclusion, and see that each step was forced by the previous one
   rather than chosen to reach a destination. Flag specifically: where a
   hypothesis was refuted by its own measurement, where a gate was opened
   against a NO-GO and on what grounds, where a rule was applied that had not
   been pre-authorised, and where an inconvenient result was kept. Those are
   the paper's credibility, not its embarrassments.

   Most of this will not appear in a four-page paper. It exists so the author
   can pick the two or three pivots that belong in the introduction, and so no
   reviewer can ask a question about provenance that has no answer.
3. **What we cannot say.** Limits that a careful reviewer will find anyway.
   Write them down before they can become surprises.
4. **Highest-leverage framing for CCAI specifically.** Given the venue is a
   workshop for climate-relevant ML, decide what the single most useful
   contribution is to a reader who builds these systems. State the one sentence
   you want them to remember. Then check that the paper is organised around it.

Only after this file exists, start drafting.

---

## The evidence base

Start from `docs/RUN_INDEX.md`: it maps every result name to its files and its
DECISIONS anchor. Then `log.md` and `docs/DECISIONS.md`. Results live under
`data/scaled_*/`, run logs under `notebooks/runs/`.

Recompute every number you print, from the CSVs, in `make_figures_v2.py`. Do not
copy a number out of prose, including out of `log.md`. If a figure and a
sentence disagree, the CSV settles it.

---

## How to write

**Register.** Plain declarative sentences. The result carries the weight; the
prose does not need to help it. Sophisticated vocabulary only where a simpler
word would be less precise, never for texture.

**Understatement over emphasis.** Report what was measured and let the reader
judge its size. Never tell the reader a result is important, surprising,
striking, or notable. If it is, they will see it. Adjectives of significance are
a substitute for significance.

**Signal density.** Every sentence carries a fact, a number, a mechanism, or a
limit. Cut any sentence that only sets up the next one. Cut throat-clearing
openers to sections and paragraphs.

**Simple by default.** Do not complicate an idea that is simple. A short method
section for a simple method is a strength.

### Hard constraints

- **No em dashes.** Use a full stop, a comma, or a colon. Rewrite the sentence
  if none fit.
- **No "not X, but Y" constructions**, and no antithetical pairs used for
  rhythm. State Y. If X needs refuting, refute it in its own sentence with
  evidence.
- **No tricolons or escalating triples** used for cadence.
- **No rhetorical questions.** No "Interestingly," "Notably," "Importantly,"
  "It is worth noting," "Crucially."
- **No hedging stacks.** One qualifier per claim at most.
- **No metaphor for technical content.** A metaphor in a results section is a
  gap in the explanation.
- Prefer active voice and the concrete subject. Name the thing that did it.

### Two failure modes to avoid by name

**Pretension.** Formal vocabulary standing in for content, method described in
more generality than was implemented, related work cited for stature rather than
because it bears on the argument.

**Overclaiming.** A null presented as a discovery, a single tile presented as a
population, an effect stated without the interval that governs it, a limitation
moved to an appendix because it is awkward. If the honest reading is weaker than
the exciting reading, write the honest one. It is also the one that survives
review.

---

## Structure

Standard workshop paper, four pages of body plus references, appendix
unlimited. Abstract, introduction, method, results, limitations, conclusion.

Two requirements specific to this work:

- **The baseline is the point.** Whatever the models do, the comparison against
  the trivial baseline is the contribution. Make it impossible to read the
  paper and miss what the baseline scored.
- **The negative result is the finding.** Present it as a measurement with a
  mechanism and a consequence for practice. Do not apologise for it, do not
  dress it up, do not bury it after the positive-sounding parts.

Put the reproduction path in the appendix: exact commands, roster sizes,
exclusions and their reasons, library versions, wall clocks and hardware.

---

## Figures

Three or four. Each earns its place by showing something a table cannot.

**Hold these to the standard of a well-produced thesis from a strong group.**
In a short paper the figures are read before the text and often instead of it.
A reviewer forms a judgement about rigour from how the figures look, before
reading a single number. Treat that as part of the argument.

### What each figure must do

- The reader should be able to answer the paper's central question from the
  first figure alone.
- Show uncertainty wherever a claim rests on it. A point estimate without its
  interval is not reportable here.
- The baseline appears in the same visual frame as the models, at the same
  scale, so the comparison needs no arithmetic from the reader.

### Craft

- **Typography.** One family throughout, matching the body text. Figure text no
  smaller than 7pt at final print size. Never let a figure font differ from the
  paper's, and never rely on default matplotlib sizing after scaling.
- **Layout.** Consistent panel dimensions, aligned axes across panels, uniform
  margins. Panels that share an axis share its range and its ticks. Column
  width fixed to the template's text width so nothing is rescaled in LaTeX.
- **Ink.** Remove top and right spines. Thin axes, no boxes, no gridlines
  unless a value must be read off, and then light grey and behind the data.
  No background fill, no drop shadows, no 3D, no dual axes.
- **Colour.** One restrained palette across every figure, colourblind-safe,
  legible in greyscale, and never the only carrier of meaning: pair it with
  position, shape or direct labelling. Muted over saturated. If two colours
  suffice, do not use five.
- **Labels.** Direct annotation on or beside the series beats a legend. If a
  legend is unavoidable, place it inside the plot area where it costs no space
  and add no frame. Axis labels carry units. No title inside the figure; the
  caption is the title.
- **Numbers.** Consistent decimal places within a panel. Do not print more
  precision than the interval supports.
- **Output.** Vector `.pdf` for the paper, `.png` at 300dpi for preview. No
  raster text. Deterministic: same script, same figure.

### Sanity checks before you finish

Print at actual size and look at it. Then view it at 50%. Then convert to
greyscale. A figure that survives all three is done. Also check that no two
figures in the paper use the same encoding for different variables.

Reuse `paper/make_figures.py` as a starting point for style and data loading if
useful. Define the style once as a shared block at the top of
`make_figures_v2.py` and have every figure inherit it, so the set reads as one
system rather than four separate plots. All new code goes in
`make_figures_v2.py`, all new images in `paper/figures/v2/`.

---

## Before you hand it back

- Every number in the text appears in a CSV, and you regenerated it.
- Search the draft for `—`, "not only", "but rather", "Notably", "Importantly",
  "Interestingly". Zero hits.
- Every claim in the abstract is supported in the results.
- The limitations section contains the objection you would raise if you were
  reviewing it.
- Read the whole thing aloud once. Cut what sounds like performance.

Report: what you wrote, which claims you could not support and dropped, and the
three weakest points a reviewer will attack.
