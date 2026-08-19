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
2. **What we assumed, and where those assumptions moved.** This project pivoted
   several times. Read `log.md` (reverse chronological) and `docs/DECISIONS.md`
   (append-only, assumed/observed/changed/commit) end to end. Record where a
   hypothesis was refuted by its own measurement, where a rule was applied that
   had not been pre-authorised, and where a result was kept despite being
   inconvenient. These are the paper's credibility, not its embarrassments.
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

- The reader should be able to answer the paper's central question from the
  first figure alone.
- Show uncertainty wherever a claim rests on it.
- Label directly on the figure. Avoid legends where a label will do.
- No chartjunk, no 3D, no dual axes, no colour carrying information that
  shape or position could carry. Readable in greyscale and when printed.
- Caption states what the figure shows and what to conclude, in two sentences.
  It does not repeat the axis labels.

Reuse `paper/make_figures.py` as a starting point for style and data loading if
useful. All new code goes in `make_figures_v2.py`, all new images in
`paper/figures/v2/`. Both `.pdf` and `.png`.

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
