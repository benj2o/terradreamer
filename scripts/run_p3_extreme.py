"""Extreme-tile P3 on 32UQC: one invocation, caches in, tables out.

WHAT THIS IS
------------
P3 is the forecastability probe: predict NDVI at horizon d from a frozen
representation of the current frame plus ``[NDVI(t), weather]``, scored against
persistence and climatology. This runs the whole nine-view table on the
``extreme`` split's 32UQC, the same 346 cubes the 2026-08-17 P4 pilot measured
its ceiling on, so the two tables are about the same place.

The four Tier-1 corrections are load-bearing and NONE of them change here:
nested-CV ridge penalty alongside fixed ``alpha=D``; paired per-fold
separability with fold-clustered intervals, never two marginal CIs;
``feature_base = [NDVI(t), weather]`` under every model row; and nine encoder
views with the plausibility screen applied.

WHY THE ENCODING IS NOT HERE
-----------------------------
Unlike P4, P3 READS EMBEDDINGS. Building them needs a GPU and, more absolutely,
python >= 3.10 -- ``dinov2_vitb14`` loads torch.hub code using PEP 604 unions,
so on the 3.9.6 dev venv two of the nine views cannot be built AT ALL. That work
is ``notebooks/phase1_10_extreme_encoding.ipynb`` on Colab. This script imports
no encoder and fine-tunes nothing: it reads the frozen caches under
``data/scaled_32UQC/{embeddings,embeddings_cir,masks}``. If they are absent it
stops and says so rather than encoding anything.

THE FOUR RULES, APPLIED AUTOMATICALLY AND LOGGED EACH TIME
-----------------------------------------------------------
1. ROSTER. 346 cubes, the P4 set. Read from ``cache_roster.csv`` when the
   encoding notebook wrote one, else from the ``cubes_excluded_fill`` column of
   ``p4_extreme_results.csv``. If neither is available, or the count is not 346,
   STOP -- a silently different roster makes the P3/P4 comparison meaningless.
2. CACHES. Both must exist and be COMPLETE, and must cover the SAME cubes.
   A cube in one cache and not the other silently changes which rows a paired
   ``_cir``/``_rgb`` difference is computed over, which is the one comparison
   this run exists to make.
3. RUNTIME VALVE. The 32UNU run was 1540 rows in 173.7 min on 7 workers, and
   naive scaling to this tile says ~13 h. Do not believe it: the P4 pilot
   measured ``loco``'s growth exponent at 1.72, not 1, and its own two-point
   projection still came in 52% under the truth. So this measures s/row PER
   FOLD MODE at two cube counts on this tile's own rows, fits one exponent per
   mode, and projects. Budget 12 CPU-hours.
4. NARROWING. If the projection exceeds the budget: drop ``loco``, then
   ``cell_mean``, then ``fixed_alpha_D``. ALL NINE ENCODER VIEWS SURVIVE EVERY
   NARROWING -- model coverage is what this run was commissioned for, fold modes
   and aggregations are what pay for it. A narrowed table is a SUBSET table:
   it is written under its own name, the full-table completeness assertions are
   NOT run on it, and every printed table says so.

    .venv/bin/python -m scripts.run_p3_extreme --n-jobs 7
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

TILE = "32UQC"
EXPECTED_CUBES = 346
BUDGET_HOURS = 12.0
CUBE_HEARTBEAT_EVERY = 25

#: The two cubes the P4 pilot excluded, as a last-resort fallback when neither
#: cache_roster.csv nor the P4 CSV is on disk. Rule 1 still asserts the count.
EXCLUDED_FILL_FALLBACK = (
    "32UQC_2018-01-28_2018-11-23_1337_1465_441_569_20_100_6_86.nc",
    "32UQC_2018-01-28_2018-11-23_441_569_441_569_6_86_6_86.nc",
)


# --------------------------------------------------------------------------
# visibility: a run measured in hours is polled, not watched
# --------------------------------------------------------------------------
class Tee:
    """stdout to the console AND to the run log, line-buffered, both flushed."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s); self.stream.flush()
        self.fh.write(s); self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush(); self.fh.flush()

    def isatty(self):
        return False


_PRELUDE: list = []
_TEE_OPEN = False


def say(msg: str = "") -> None:
    print(msg, flush=True)
    if not _TEE_OPEN:
        _PRELUDE.append(msg)


def open_tee(path: str):
    global _TEE_OPEN
    fh = open(path, "a", encoding="utf-8", buffering=1)
    for line in _PRELUDE:
        fh.write(line + "\n")
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    _TEE_OPEN = True
    return fh


def banner(msg: str, char: str = "=") -> None:
    say("\n" + char * 92)
    say(msg)
    say(char * 92)


def _fmt(x, w=7, p=3):
    return f"{x:+{w}.{p}f}" if np.isfinite(x) else " " * (w - 3) + "nan"


class EvaluateCounter:
    """Wraps ``p3_forecast.evaluate`` so the fitting loop reports where it is.

    ``evaluate`` is called once per output row, so "row N/total" means the same
    thing in the log as it does in the CSV. Per-mode wall clock is accumulated
    here too -- that is what the runtime valve fits its exponent on, and taking
    it from the same call that produces the row means the calibration and the
    real run are measuring the identical unit of work.
    """

    def __init__(self, module, total: int, label: str = "", quiet: bool = False,
                 heartbeat_every: int = 100):
        self.module, self.total, self.label = module, total, label
        # quiet suppresses the PER-ROW line but never all output: a calibration
        # point that prints nothing for twenty minutes is indistinguishable
        # from a hang, and this run is polled rather than watched.
        self.quiet = quiet
        self.heartbeat_every = heartbeat_every
        self.original = module.evaluate
        self.n = 0
        self.t0 = time.time()
        self.per_mode: dict = {}
        self.rows_per_mode: dict = {}

    def __enter__(self):
        counter = self

        def evaluate(sources, target, rows, manifest, mode, estimator, kind,
                     *a, **kw):
            t = time.time()
            out = counter.original(sources, target, rows, manifest, mode,
                                   estimator, kind, *a, **kw)
            dt = time.time() - t
            counter.n += 1
            counter.per_mode[mode] = counter.per_mode.get(mode, 0.0) + dt
            counter.rows_per_mode[mode] = counter.rows_per_mode.get(mode, 0) + 1
            elapsed = time.time() - counter.t0
            per_row = elapsed / counter.n
            if not counter.quiet:
                if counter.total:
                    eta = per_row * max(0, counter.total - counter.n)
                    where = f"row {counter.n}/{counter.total}"
                    eta_s = f"ETA {eta / 60:6.1f} min"
                else:
                    # no calibration ran, so there is no honest total to divide
                    # into: report position and rate, never a made-up ETA.
                    where = f"row {counter.n}"
                    eta_s = "ETA n/a (no calibration)"
                say(f"[p3-eta]{counter.label} {where}  "
                    f"{mode}/{estimator}/{kind}  this {dt:6.1f}s  "
                    f"elapsed {elapsed / 60:6.1f} min  {per_row:6.1f} s/row  "
                    f"{eta_s}")
            elif counter.n % counter.heartbeat_every == 0:
                say(f"[p3-cal]{counter.label} {counter.n} rows  "
                    f"elapsed {elapsed / 60:6.1f} min  {per_row:6.2f} s/row  "
                    f"(last {mode}/{estimator})")
            return out

        self.module.evaluate = evaluate
        return self

    def __exit__(self, *exc):
        self.module.evaluate = self.original
        return False

    def seconds_per_row(self) -> dict:
        return {m: self.per_mode[m] / self.rows_per_mode[m]
                for m in self.per_mode if self.rows_per_mode.get(m)}


# --------------------------------------------------------------------------
# rule 1: the roster
# --------------------------------------------------------------------------
def resolve_roster(out_root: str, cube_dir: str) -> tuple:
    """The cubes to fit, and the recorded reason for every exclusion.

    TWO DIFFERENT RULES LIVE HERE AND THEY MUST NOT BE CONFLATED.

    Rule 1 governs the P4 roster: the tile's cubes minus the two the P4 pilot
    excluded for the zero-reflectance fill block. That number must be 346, or
    the two tables are not about the same place and nothing below is worth
    running.

    Rule 3 governs encode-time drops: a cube whose nine views could not all be
    built is out of BOTH caches and the masks, never partial. Those drops are
    legitimate and expected -- they are reported with count and reason, and the
    fitted set is correspondingly smaller than 346. Asserting 346 on the FITTED
    set would refuse exactly the outcome rule 3 exists to produce.

    Three sources in preference order, because the run has to work both
    straight after the Colab copy-down and on a fresh checkout:

      1. ``cache_roster.csv`` -- written by the encoding notebook, and the only
         source that knows about a cube DROPPED during encoding.
      2. the ``cubes_excluded_fill`` column of ``p4_extreme_results.csv``.
      3. the hardcoded fallback, which still has to pass the count assertion.
    """
    import pandas as pd

    on_disk = sorted(os.path.basename(p)
                     for p in glob.glob(os.path.join(cube_dir, "*.nc")))
    say(f"[rule 1] {len(on_disk)} cubes on disk at {cube_dir}")

    roster_csv = os.path.join(out_root, "cache_roster.csv")
    p4_csv = os.path.join(out_root, "p4_extreme_results.csv")

    if os.path.exists(roster_csv):
        r = pd.read_csv(roster_csv)
        assert {"cube", "in_cache"} <= set(r.columns), (
            f"{roster_csv} has no cube/in_cache columns")
        keep = sorted(r.loc[r.in_cache.astype(bool), "cube"])
        ex = r.loc[~r.in_cache.astype(bool)]
        reasons = {c: str(w) for c, w in zip(ex.cube, ex.get(
            "reason", pd.Series([""] * len(ex))))}
        source = "cache_roster.csv (written by the encoding notebook)"
    elif os.path.exists(p4_csv):
        col = pd.read_csv(p4_csv, usecols=lambda c: c == "cubes_excluded_fill")
        assert "cubes_excluded_fill" in col.columns, (
            f"{p4_csv} has no cubes_excluded_fill column. RULE 1 STOP: the "
            "roster cannot be reproduced, and a silently different roster "
            "makes the P3/P4 comparison meaningless.")
        vals = col["cubes_excluded_fill"].dropna().unique()
        assert len(vals) == 1, (
            f"cubes_excluded_fill is not constant across the P4 table: {vals}")
        drop = [c for c in str(vals[0]).split(";") if c]
        keep = [c for c in on_disk if c not in set(drop)]
        reasons = {c: "p4_fill_block: zero-reflectance B04/B8A no-data"
                   for c in drop}
        source = "p4_extreme_results.csv:cubes_excluded_fill"
    else:
        drop = list(EXCLUDED_FILL_FALLBACK)
        keep = [c for c in on_disk if c not in set(drop)]
        reasons = {c: "p4_fill_block: zero-reflectance B04/B8A no-data"
                   for c in drop}
        source = "hardcoded EXCLUDED_FILL_FALLBACK (no roster CSV on disk)"

    say(f"[rule 1] roster source: {source}")

    p4_excluded = {c: w for c, w in reasons.items()
                   if w.startswith("p4_fill_block")}
    encode_drops = {c: w for c, w in reasons.items()
                    if not w.startswith("p4_fill_block")}

    say(f"[rule 1] P4 fill-block exclusions: {len(p4_excluded)}")
    for c in sorted(p4_excluded):
        say(f"[rule 1]   {c}")

    p4_roster = len(on_disk) - len(p4_excluded)
    if p4_roster != EXPECTED_CUBES:
        raise SystemExit(
            f"\nRULE 1 STOP: the P4 roster is {p4_roster} cubes, not "
            f"{EXPECTED_CUBES}.\nP3 must start from the same cubes as the P4 "
            "pilot or the two tables are not about the same place.\n"
            f"Source was: {source}\n")
    say(f"[rule 1] P4 ROSTER = {p4_roster} cubes -- matches the pilot")

    # Rule 3: encode-time drops are legitimate and are NOT a rule 1 failure.
    if encode_drops:
        say(f"\n[rule 3] {len(encode_drops)} cube(s) dropped at encode time, "
            "out of BOTH caches and the masks:")
        for c in sorted(encode_drops):
            say(f"[rule 3]   {c}")
            say(f"[rule 3]     {encode_drops[c][:220]}")
        say(f"[rule 3] cubes ENCODED {p4_roster} -> cubes FITTED {len(keep)}")
        say("[rule 3] P4 fitted 346; P3 fits "
            f"{len(keep)}. The two tables cover nearly, but not exactly, the "
            "same cubes -- and that is reported, not smoothed over.")
    else:
        say("[rule 3] no cube was dropped at encode time")

    missing = [c for c in keep if c not in set(on_disk)]
    assert not missing, (
        f"RULE 1 STOP: {len(missing)} rostered cubes are not on disk, e.g. "
        f"{missing[:2]}. The caches and the cubes disagree.")

    say(f"[p3-extreme] FITTING {len(keep)} cubes")
    return keep, reasons, source, encode_drops


# --------------------------------------------------------------------------
# rule 2: the caches are whole, and cover the same cubes
# --------------------------------------------------------------------------
def assert_caches_complete(emb_dir, emb_dir_cir, mask_dir, cube_ids, encoders):
    from encoders.pipeline import assert_embeddings_complete, audit_embeddings
    from probes import p3_forecast as p3

    want_rgb = [e for e in encoders if not p3.is_cir(e)]
    want_cir = [e for e in encoders if p3.is_cir(e)]
    for d, want, what in ((emb_dir, want_rgb, "RGB"),
                          (emb_dir_cir, want_cir, "colour-infrared")):
        if not want:
            continue
        missing = [(c, e) for c in sorted(cube_ids) for e in want
                   if not os.path.exists(
                       os.path.join(d, f"{os.path.splitext(c)[0]}__{e}.npz"))]
        if missing:
            raise SystemExit(
                f"\nRULE 2 STOP: the {what} cache is missing "
                f"{len(missing)} (cube, view) pairs, e.g. {missing[:3]}.\n"
                f"Build it with notebooks/phase1_10_extreme_encoding.ipynb; "
                "no probe produces it, and this script encodes nothing.\n")
        audit = audit_embeddings(d, cube_ids=set(cube_ids), verbose=False)
        assert_embeddings_complete(audit, set(cube_ids), tuple(want))
        say(f"[rule 2] {what} cache complete: {len(cube_ids)} x {len(want)} "
            f"= {len(cube_ids) * len(want)} .npz")

    have_rgb = {f.split("__")[0] for f in os.listdir(emb_dir)
                if f.endswith(".npz")}
    have_cir = {f.split("__")[0] for f in os.listdir(emb_dir_cir)
                if f.endswith(".npz")}
    if have_rgb != have_cir:
        raise SystemExit(
            f"\nRULE 2 STOP: the two caches cover different cubes "
            f"({len(have_rgb - have_cir)} rgb-only, "
            f"{len(have_cir - have_rgb)} cir-only).\nEvery _cir/_rgb paired "
            "difference would be computed over a different set on each side, "
            "which is the one comparison this run exists to make.\n")
    say(f"[rule 2] both caches cover the SAME {len(have_rgb)} cubes")

    n_msk = len(glob.glob(os.path.join(mask_dir, "*.npz")))
    assert n_msk >= len(cube_ids), (
        f"RULE 2 STOP: {n_msk} masks for {len(cube_ids)} cubes; "
        "common-masking needs one per cube.")
    say(f"[rule 2] masks {n_msk} .npz")


# --------------------------------------------------------------------------
# rule 2b: the cached mask must equal the finiteness of the canonical NDVI
# --------------------------------------------------------------------------
def _mask_ndvi_check(args) -> dict:
    """One cube. Module-level and picklable, so it can run in a process pool."""
    cube, cube_dir, mask_dir = args
    try:
        import numpy as np

        from data.loader import CubeSample, cube_ndvi, load_cube
        from encoders.frames import select_clear_frames
        from encoders.pipeline import load_masks
        from probes.p3_forecast import MIN_CLEAR_FRACTION

        stem = os.path.splitext(cube)[0]
        mp = os.path.join(mask_dir, f"{stem}__masks.npz")
        if not os.path.exists(mp):
            return {"cube": cube, "ok": False, "why": "no cached mask"}
        s = load_cube(os.path.join(cube_dir, cube), verbose=False)
        m = np.asarray(load_masks(mp).mask)
        sel = select_clear_frames(s.values, s.timestamps, s.mask,
                                  min_clear=MIN_CLEAR_FRACTION, verbose=False)
        nd = cube_ndvi(CubeSample(values=sel.values, timestamps=sel.timestamps,
                                  mask=sel.mask, path=s.path, bands=s.bands))
        if m.shape != nd.shape:
            return {"cube": cube, "ok": False,
                    "why": f"mask {m.shape} != NDVI {nd.shape}"}
        d = (m != np.isfinite(nd))
        n = int(d.sum())
        if n == 0:
            return {"cube": cube, "ok": True, "why": ""}
        fr = int((d.reshape(d.shape[0], -1).sum(1) > 0).sum())
        return {"cube": cube, "ok": False,
                "why": (f"mask != isfinite(NDVI) on {n} pixels in {fr}/"
                        f"{d.shape[0]} frames ({n / d.size:.2e} of pixels); "
                        "the mask calls them valid where NDVI is 0/0 -- the "
                        "zero-reflectance fill block, same family as the two "
                        "cubes P4 excluded")}
    except Exception as e:                       # noqa: BLE001 -- reported
        return {"cube": cube, "ok": False, "why": f"{type(e).__name__}: {e}"[:200]}


def drop_mask_ndvi_mismatches(roster, cube_dir, mask_dir, n_jobs):
    """Cubes whose cached mask disagrees with the canonical NDVI's finiteness.

    WHY THIS IS A PRE-FLIGHT AND NOT AN EXCEPTION HANDLER. ``build_p3_data``
    asserts this per cube, deep inside target construction and ~10 minutes into
    a run, and its message says "one of them is stale". On this tile that
    diagnosis is wrong: the cached mask reproduces bit-identically from
    ``cube_masks`` locally, so nothing is stale -- ``cube_masks`` and
    ``cube_ndvi`` genuinely disagree, because the mask calls a pixel valid on
    the strength of s2_dlmask/SCL plus finite bands while NDVI is 0/0 there.
    That is the zero-reflectance fill block the P4 pilot documented.

    The properly correct fix is a zero-reflectance rule beside
    ``finite_valid_mask``. That is a SHARED path -- P1, P2 and P3 all read
    through it, and every published 32UNU table with it -- so moving it would
    move numbers this run has no mandate to move. The P4 pilot made the same
    call and recorded it so it can be overruled. This does the same: the
    affected cubes are excluded WHOLE, named, and counted.
    """
    from concurrent.futures import ProcessPoolExecutor

    say(f"[rule 2b] checking {len(roster)} cached masks against the canonical "
        f"NDVI on {n_jobs} workers")
    t0 = time.time()
    args = [(c, cube_dir, mask_dir) for c in roster]
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for i, r in enumerate(ex.map(_mask_ndvi_check, args), 1):
            results.append(r)
            if i % CUBE_HEARTBEAT_EVERY == 0 or i == len(args):
                el = time.time() - t0
                bad = sum(1 for x in results if not x["ok"])
                say(f"[rule 2b] [{i:>3}/{len(args)}] {el / 60:5.1f} min | "
                    f"{bad} mismatched")
    bad = {r["cube"]: r["why"] for r in results if not r["ok"]}
    if not bad:
        say(f"[rule 2b] all {len(roster)} masks agree with the canonical NDVI")
        return list(roster), {}
    say(f"[rule 2b] {len(bad)} cube(s) EXCLUDED -- mask/NDVI disagreement:")
    for c in sorted(bad):
        say(f"[rule 2b]   {c}")
        say(f"[rule 2b]     {bad[c]}")
    keep = [c for c in roster if c not in bad]
    say(f"[rule 2b] {len(roster)} -> {len(keep)} cubes")
    return keep, bad


# --------------------------------------------------------------------------
# rule 3: the two-point runtime valve
# --------------------------------------------------------------------------
def _manifest_for(paths, verbose=False):
    from data.loader import load_cube
    from encoders.manifest import build_manifest
    return build_manifest([load_cube(p, verbose=False) for p in paths],
                          verbose=verbose)


def calibrate(paths, cube_dir, dirs, encoders, horizons, aggregations,
              fold_modes, alpha_rules, k, n_jobs, screen, n_small, n_large):
    """s/row PER FOLD MODE at two cube counts, and one exponent per mode.

    Why per fold mode, and why an exponent at all. ``cube`` folds grow their
    training set with the tile; ``loco`` grows its fold COUNT and its per-fold
    training set together, which is why the P4 pilot measured its exponent at
    1.72 rather than 1. A single blended s/row hides that, and a linear estimate
    on this tile has been measured 4-6x low.

    The subsets are STRIDED, not prefixes: the filenames sort by window-start
    date, so a prefix shares a narrow slice of the season and the proxy
    climatology is not identifiable on it.
    """
    from probes import p3_forecast as p3

    points = {}
    for n_cubes in (n_small, n_large):
        stride = max(1, len(paths) // n_cubes)
        sub = paths[::stride][:n_cubes]
        banner(f"[rule 3] CALIBRATION POINT: {len(sub)} cubes", "-")
        t0 = time.time()
        manifest = _manifest_for(sub)
        say(f"[rule 3] manifest {manifest.shape}, "
            f"{manifest.cube_id.nunique()} cubes, {time.time() - t0:.0f}s")

        with EvaluateCounter(p3, total=0, label=f"[cal{len(sub)}]",
                             quiet=True) as counter:
            t0 = time.time()
            df, _ = p3.run_p3(
                manifest, cube_dir, encoders=encoders, horizons=horizons,
                aggregations=aggregations, fold_modes=fold_modes,
                alpha_rules=alpha_rules, k=k, emb_dir=dirs["emb"],
                emb_dir_cir=dirs["cir"], mask_dir=dirs["mask"],
                plausibility_screen=screen, n_jobs=n_jobs, verbose=False)
            wall = time.time() - t0
        spr = counter.seconds_per_row()
        points[len(sub)] = {"s_per_row": spr,
                            "rows_per_mode": dict(counter.rows_per_mode),
                            "rows": len(df), "wall": wall}
        say(f"[rule 3] {len(sub)} cubes: {len(df)} rows, {wall / 60:.1f} min "
            f"wall, {counter.n} evaluate() calls")
        for m in sorted(spr):
            say(f"[rule 3]     {m:<14} {counter.rows_per_mode[m]:>5} rows  "
                f"{spr[m]:8.3f} s/row")
    return points


def project(points, n_small, n_large, n_full, budget_hours):
    """One growth exponent per fold mode, from the two measured points."""
    a, b = points[n_small], points[n_large]
    ratio_cal = n_large / n_small
    ratio_full = n_full / n_large

    banner("[rule 3] PROJECTION TO THE FULL TILE", "-")
    say(f"  two points: {n_small} and {n_large} cubes; full tile {n_full}")
    say(f"  {'mode':<14} {'s/row@' + str(n_small):>12} "
        f"{'s/row@' + str(n_large):>12} {'exponent':>9} "
        f"{'s/row@' + str(n_full):>12} {'rows':>6} {'hours':>8}")
    total_h = 0.0
    per_mode = {}
    n_invalid = 0
    for m in sorted(set(a["s_per_row"]) | set(b["s_per_row"])):
        sa = a["s_per_row"].get(m)
        sb = b["s_per_row"].get(m)
        if not sa or not sb or sa <= 0 or sb <= 0:
            continue
        exponent = math.log(sb / sa) / math.log(ratio_cal)
        # A NEGATIVE exponent is not a measurement, it is a corrupted one. It
        # says a fold gets CHEAPER per row as the tile grows, which none of
        # these modes can do: cube grows its training set, loco grows fold
        # count and training set together, spatial_block grows both. It shows
        # up when the two points were measured under different machine load --
        # on 2026-08-18 a Spotlight reindex of the freshly copied 2 GB cache
        # made the 20-cube point 5x slower than the 40-cube one and produced
        # loco = -2.65, projecting 0.00 h for what a quiet machine had just
        # measured at 10.12 h. Believing it would have run loco at 342 cubes.
        # So: reject it, fall back to LINEAR growth from the WORSE of the two
        # points, and say so loudly. Conservative on purpose -- the failure
        # this guards against is under-narrowing into a 30-hour run.
        invalid = exponent < 0.0
        if invalid:
            sb = max(sa, sb)
            exponent = 1.0
        # Rows are fixed by the GRID, not by the cube count: the same table is
        # produced at 20 cubes and at 346. What scales is the cost of a row.
        rows = b["rows_per_mode"].get(m, 0)
        s_full = sb * (ratio_full ** exponent)
        hours = rows * s_full / 3600.0
        total_h += hours
        per_mode[m] = {"exponent": exponent, "s_per_row_full": s_full,
                       "rows": rows, "hours": hours}
        flag = "  <- REJECTED (negative); linear from the worse point" if invalid else ""
        say(f"  {m:<14} {sa:12.3f} {sb:12.3f} {exponent:9.2f} "
            f"{s_full:12.2f} {rows:6d} {hours:8.2f}{flag}")
        if invalid:
            n_invalid += 1
    say(f"  {'TOTAL':<14} {'':>12} {'':>12} {'':>9} {'':>12} {'':>6} "
        f"{total_h:8.2f}")
    say(f"\n  budget {budget_hours:.1f} CPU-hours -> "
        f"{'WITHIN' if total_h <= budget_hours else 'OVER'} budget")
    say("  NOTE: this is a two-point fit. The P4 pilot's own two-point "
        "projection came in 52% under the truth, so treat it as a floor.")
    if n_invalid:
        say(f"\n  *** {n_invalid} fold mode(s) produced a NEGATIVE exponent and "
            "were rejected. ***")
        say("  That means the two calibration points were measured under "
            "different machine load, so this projection is NOT a clean read of "
            "the algorithm. Re-run the valve on a quiet machine, or pass "
            "--narrow with the axis a previous clean calibration justified.")
    return total_h, per_mode


def narrow(total_h, per_mode, fold_modes, aggregations, alpha_rules,
           budget_hours, n_cubes=None):
    """Rule 4's ladder. ALL NINE ENCODER VIEWS SURVIVE EVERY STEP.

    Order is fixed and each step is logged: (a) drop ``loco``, 346 folds of two
    to five rows each here, and a contingency table over such a fold is not a
    contingency table; (b) drop ``cell_mean``, the resolution axis rather than
    the decision axis, and 16 rows per forecast row; (c) drop
    ``fixed_alpha_D``, keeping ``nested_cv``, which is the rule the Tier-1
    headline is read on.
    """
    from probes import p3_forecast as p3

    steps = []
    fm = list(fold_modes)
    ag = list(aggregations)
    ar = None if alpha_rules is None else list(alpha_rules)
    est = total_h

    def _out(a):
        return None if a is None else tuple(a)

    if est <= budget_hours:
        return tuple(fm), tuple(ag), _out(ar), steps, est

    if "loco" in fm:
        saved = per_mode.get("loco", {}).get("hours", 0.0)
        fm.remove("loco")
        est -= saved
        steps.append(f"(a) dropped fold_mode=loco (-{saved:.2f} h): "
                     f"{n_cubes if n_cubes else 'many'} folds of two to five "
                     "rows each on this tile")
        say(f"[rule 4] {steps[-1]}; projection now {est:.2f} h")
    if est > budget_hours and "cell_mean" in ag:
        # cell_mean is 16 rows per forecast row: the aggregation multiplies the
        # row COUNT, so removing it scales the estimate rather than subtracting
        # a measured block.
        before = est
        est *= (len(ag) - 1) / len(ag)
        ag.remove("cell_mean")
        steps.append(f"(b) dropped aggregation=cell_mean "
                     f"({before:.2f} -> {est:.2f} h): the resolution axis, not "
                     "the decision axis")
        say(f"[rule 4] {steps[-1]}")
    if est > budget_hours and (ar is None or p3.ALPHA_RULE_FIXED in ar):
        before = est
        est *= 0.5
        # ALPHA_RULE_NA must survive: it is the only rule hgb, the MLP and the
        # unfitted rows have, and dropping it removes those rows entirely
        # rather than narrowing the ridge.
        ar = [p3.ALPHA_RULE_TUNED, p3.ALPHA_RULE_NA]
        steps.append(f"(c) dropped alpha_rule=fixed_alpha_D "
                     f"({before:.2f} -> {est:.2f} h): nested_cv is the rule the "
                     "Tier-1 headline is read on; not_a_ridge kept so hgb and "
                     "the MLP still contribute rows")
        say(f"[rule 4] {steps[-1]}")

    if est > budget_hours:
        say(f"[rule 4] STILL over budget at {est:.2f} h after every permitted "
            "narrowing. Proceeding anyway: the ladder is exhausted and all "
            "nine encoder views survive by construction.")
    return tuple(fm), tuple(ag), _out(ar), steps, est


def _resolve_predictions(path: str) -> str:
    """The path ``write_predictions`` actually used.

    It appends ``.gz`` when the projected size crosses its threshold and
    returns the real path -- but ``run_p3`` does not pass that back through its
    signature (by design: it goes to the run log and stdout). So the caller has
    to look for both spellings, or the trigger stage dies on a missing file
    AFTER the expensive part has already succeeded. That is exactly what
    happened on the 2026-08-19 run: 6.4 h of fits landed, then the triggers
    could not find a 361 MB file sitting right next to where they looked.
    """
    for cand in (path, path + ".gz"):
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        f"\nno predictions at {path} or {path}.gz. run_p3 was asked for "
        "emit_predictions=True; check the '[p3] wrote' line in the run log "
        "for where they actually went.\n")


def _score_triggers(pred_path: str, out_root: str, args) -> None:
    """Threshold-crossing skill, same thresholds and levels as 2026-08-16."""
    banner("TRIGGER METRICS -- same thresholds and levels as 2026-08-16")
    from probes import p3_triggers as tg

    preds = tg.load_predictions(pred_path)
    tg.assert_thresholds_are_train_fitted(preds)
    tr = tg.trigger_metrics(preds)
    tpath = os.path.join(out_root, args.triggers_name)
    tr.to_csv(tpath, index=False)
    say(f"[p3-extreme] wrote {tpath} ({len(tr)} rows)")
    tg.print_trigger_table(tr, level="extreme_low")


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def report(df, args, roster, reasons, narrowed, dirs, gpu_note,
           encode_drops):
    """Every headline this run was commissioned for, beside 32UNU's Tier-1."""
    import pandas as pd

    from probes import p3_forecast as p3

    old = None
    if args.published_csv and os.path.exists(args.published_csv):
        old = pd.read_csv(args.published_csv)
        say(f"\n[report] comparing against {args.published_csv} ({len(old)} rows)")
    else:
        say("\n[report] no 32UNU Tier-1 CSV on disk; the 'was' column is "
            "omitted rather than filled with zeros")

    banner("ROWS RETAINED PER HORIZON")
    if "delta_days" in df.columns and "n_retained" in df.columns:
        piv = (df.groupby("delta_days")["n_retained"].max()
               .rename("rows_32UQC").to_frame())
        if old is not None and {"delta_days", "n_retained"} <= set(old.columns):
            piv["rows_32UNU"] = old.groupby("delta_days")["n_retained"].max()
            piv["ratio"] = piv.rows_32UQC / piv.rows_32UNU
        say(piv.to_string())

    banner("THE HEADLINE -- cube_mean / cube folds / nested_cv / +base, "
           "skill vs persistence")
    if narrowed:
        say("*** NARROWED RUN: this is a SUBSET table. " + "; ".join(narrowed))
    # raw_features is one of the nine views but does NOT carry
    # model_kind == "forecast": it has its own kinds. Filtering on "forecast"
    # alone silently drops a view from the headline, which is the one thing
    # this run was commissioned not to do.
    VIEW_KINDS = ("forecast", "raw_features_weather")
    q = {"aggregation": "cube_mean", "fold_mode": "cube",
         "alpha_rule": p3.ALPHA_RULE_TUNED,
         "feature_base": p3.FEATURE_BASE_SHARED}
    sub = df
    for kk, vv in q.items():
        if kk in sub.columns:
            sub = sub[sub[kk] == vv]
    if "model_kind" in sub.columns:
        sub = sub[sub.model_kind.isin(VIEW_KINDS)]
    n_views = sub.encoder.nunique() if "encoder" in sub.columns else 0
    say(f"  {len(sub)} rows match {q} with model_kind in {VIEW_KINDS}")
    say(f"  {n_views} distinct encoder views present")
    if n_views and n_views != 9:
        say(f"  *** WARNING: {n_views} views, not 9. A view is missing from "
            "the headline. ***")
    cols = [c for c in ("encoder", "delta_days", "skill_vs_persistence",
                        "paired_diff_vs_persistence",
                        "paired_ci_lo_vs_persistence",
                        "paired_ci_hi_vs_persistence",
                        "separable_vs_persistence")
            if c in sub.columns]
    if len(sub) and cols:
        say(sub.sort_values([c for c in ("delta_days", "encoder")
                             if c in sub.columns])[cols].to_string(index=False))
        if "separable_vs_persistence" in sub.columns:
            banner("SEPARABLY BETTER / WORSE THAN PERSISTENCE, per horizon", "-")
            say("  the same accounting the 2026-08-16 entry uses")
            for h in sorted(df.get("delta_days", pd.Series(dtype=float))
                            .dropna().unique()):
                g = df[(df.delta_days == h)
                       & df.separable_vs_persistence.notna()]
                if not len(g):
                    continue
                better = int(((g.separable_vs_persistence.astype(bool))
                              & (g.paired_diff_vs_persistence > 0)).sum())
                worse = int(((g.separable_vs_persistence.astype(bool))
                             & (g.paired_diff_vs_persistence < 0)).sum())
                say(f"    {int(h):>4} d   better {better:>4}   worse {worse:>4}"
                    f"   of {len(g):>4} rows")

    banner("THE _cir VS _rgb PAIRED DIFFERENCE, all four twins")
    say("  same weights, same frames, same read-out -- only the bands differ.")
    tw = df
    for kk, vv in q.items():
        if kk in tw.columns:
            tw = tw[tw[kk] == vv]
    tcols = [c for c in ("encoder", "delta_days", "paired_diff_vs_rgb_twin",
                         "paired_ci_lo_vs_rgb_twin", "paired_ci_hi_vs_rgb_twin",
                         "separable_vs_rgb_twin") if c in tw.columns]
    if "paired_diff_vs_rgb_twin" in tw.columns:
        tw = tw[tw.paired_diff_vs_rgb_twin.notna()]
        say(f"  {len(tw)} twin rows carry a paired difference")
        if len(tw):
            say(tw.sort_values([c for c in ("delta_days", "encoder")
                                if c in tw.columns])[tcols]
                .to_string(index=False))
    else:
        say("  no paired_diff_vs_rgb_twin column in this table -- "
            "add_paired_separability did not attach it.")

    banner("weather_only -- read against the 93% date-recoverability on this tile")
    wo = df
    for kk, vv in q.items():
        if kk in wo.columns:
            wo = wo[wo[kk] == vv]
    if "model_kind" in wo.columns:
        wo = wo[wo.model_kind == "weather_only"]
    wcols = [c for c in ("delta_days", "skill_vs_persistence",
                         "paired_diff_vs_persistence",
                         "paired_ci_lo_vs_persistence",
                         "paired_ci_hi_vs_persistence",
                         "separable_vs_persistence") if c in wo.columns]
    if len(wo) and wcols:
        say(wo.sort_values("delta_days")[wcols].to_string(index=False))
        say("\n  On 32UQC about 93% of a typical windowed weather feature is")
        say("  recoverable from the DATE alone (32UNU: ~61%), because 346 cubes")
        say("  on one orbit lattice and 47 distinct dates read an E-OBS grid too")
        say("  coarse to separate them. Expect this row to be doing more")
        say("  calendar-fitting here than on 32UNU.")
    else:
        say("  no weather_only rows in this table")

    banner("PROVENANCE")
    say(f"  tile              {TILE} (extreme split)")
    say(f"  cubes FITTED      {len(roster)}")
    say(f"  cubes excluded    {len(reasons)}  "
        f"(P4 fill-block + encode-time drops)")
    for c, why in sorted(reasons.items()):
        say(f"                      {c}")
        say(f"                        {str(why)[:200]}")
    if encode_drops:
        say(f"  encoded vs fitted {EXPECTED_CUBES} encoded-roster -> "
            f"{len(roster)} fitted ({len(encode_drops)} dropped by rule 3)")
        say("                      P4 fitted 346; these two tables cover "
            "nearly, not exactly, the same cubes.")
    say(f"  embeddings        {dirs['emb']}")
    say(f"  embeddings_cir    {dirs['cir']}")
    say(f"  masks             {dirs['mask']}")
    say(f"  encoding          {gpu_note}")
    say(f"  narrowed          {'; '.join(narrowed) if narrowed else 'no'}")


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", default=TILE)
    ap.add_argument("--out", default=None, help="cache root; default data/scaled_<tile>")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0)
    ap.add_argument("--budget-hours", type=float, default=BUDGET_HOURS)
    ap.add_argument("--cal-small", type=int, default=20)
    ap.add_argument("--cal-large", type=int, default=40)
    ap.add_argument("--narrow", default="",
                    help="apply a narrowing WITHOUT re-measuring, e.g. 'loco'. "
                         "For use ONLY when a previous CLEAN calibration "
                         "justified it -- the reason travels on every row and "
                         "the table is written under the subset name.")
    ap.add_argument("--narrow-reason", default="",
                    help="provenance for --narrow; goes in the log and the CSV")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="run the full grid without the valve. Only sensible "
                         "when a previous run already measured it.")
    ap.add_argument("--horizons", default="")
    ap.add_argument("--aggregations", default="")
    ap.add_argument("--fold-modes", default="")
    ap.add_argument("--encoders", default="")
    ap.add_argument("--no-screen", action="store_true")
    ap.add_argument("--no-predictions", action="store_true",
                    help="do not emit the per-observation predictions file. "
                         "It changes NO number in the results table -- the "
                         "paired columns come from payloads, not predictions "
                         "-- but it is the bulk of the disk and I/O. Use it "
                         "for a fold-mode top-up whose deliverable is the "
                         "table. Implies --skip-triggers.")
    ap.add_argument("--max-cubes", type=int, default=0,
                    help="SMOKE TEST ONLY: strided subset of the roster, so a "
                         "new machine can be validated in minutes. Any table "
                         "it produces is a shape check, never a result -- the "
                         "effective n is CUBES and this changes it.")
    ap.add_argument("--skip-triggers", action="store_true")
    ap.add_argument("--triggers-only", action="store_true",
                    help="skip the fitting entirely and score triggers + "
                         "report from tables already on disk. For finishing a "
                         "run whose fits landed but whose trigger stage did "
                         "not -- never for producing a table.")
    ap.add_argument("--csv-name", default="p3_extreme_results.csv")
    ap.add_argument("--predictions-name", default="p3_extreme_predictions.csv")
    ap.add_argument("--triggers-name", default="p3_extreme_triggers.csv")
    ap.add_argument("--log-name", default="p3_extreme_run.log")
    ap.add_argument("--published-csv",
                    default="data/scaled_32UNU/p3_tier1_results.csv")
    args = ap.parse_args()

    out_root = args.out or os.path.join("data", f"scaled_{args.tile}")
    cube_dir = os.path.join(out_root, "raw")
    log_path = os.path.join(out_root, args.log_name)
    os.makedirs(out_root, exist_ok=True)

    say(f"[p3-extreme] {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"[p3-extreme] tile {args.tile}, cache root {out_root}")

    from probes import p3_forecast as p3

    dirs = {"emb": os.path.join(out_root, "embeddings"),
            "cir": os.path.join(out_root, p3.CIR_EMB_DIRNAME),
            "mask": os.path.join(out_root, "masks")}
    for d, what in ((cube_dir, "cubes"), (dirs["emb"], "RGB embeddings"),
                    (dirs["cir"], "colour-infrared embeddings"),
                    (dirs["mask"], "masks")):
        if not os.path.isdir(d):
            raise SystemExit(
                f"\nSTOP: no {what} at {d}.\n"
                "The caches are built by notebooks/phase1_10_extreme_encoding."
                "ipynb on Colab (GPU + python >= 3.10); this script encodes\n"
                "nothing and imports no encoder. Copy embeddings/, "
                "embeddings_cir/, masks/ and cache_roster.csv down first.\n")

    open_tee(log_path)
    say(f"[p3-extreme] logging to {log_path}")

    # --- rule 1 -----------------------------------------------------------
    banner("RULE 1: THE CUBE ROSTER")
    roster, reasons, source, encode_drops = resolve_roster(
        out_root, cube_dir)
    paths = [os.path.join(cube_dir, c) for c in roster]

    encoders = (tuple(args.encoders.split(",")) if args.encoders
                else p3.ENCODER_VIEWS_ALL)
    horizons = (tuple(int(x) for x in args.horizons.split(","))
                if args.horizons else p3.HORIZONS)
    aggregations = (tuple(args.aggregations.split(",")) if args.aggregations
                    else p3.AGGREGATIONS)
    fold_modes = (tuple(args.fold_modes.split(",")) if args.fold_modes
                  else p3.FOLD_MODES)
    # None means NO narrowing: every estimator gets its natural rules
    # (ALPHA_RULES for the ridge, ALPHA_RULE_NA for hgb / the MLP / unfitted
    # rows). Passing p3.ALPHA_RULES explicitly is NOT the same thing -- it
    # excludes "not_a_ridge" and leaves hgb contributing no rows at all. The
    # published scripts/rerun_p3_tier1.py leaves it None for exactly this
    # reason; rule 4(c) is the only thing that may narrow it.
    alpha_rules = None
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 2) - 1)
    screen = not args.no_screen

    # Nine views is the commission, and narrowing must never touch it. The
    # ONLY exemption is --max-cubes, which already declares itself a shape
    # check rather than a result.
    assert len(encoders) == 9 or args.max_cubes, (
        f"this run is nine encoder views, got {len(encoders)}: {encoders}. "
        "Model coverage is what it was commissioned for. (Pass --max-cubes "
        "if this is a smoke test.)")

    # --- rule 2 -----------------------------------------------------------
    banner("RULE 2: BOTH CACHES ARE WHOLE AND COVER THE SAME CUBES")
    say("[rule 2] nothing is encoded here and no encoder is imported: this "
        "reads the frozen caches.")
    assert_caches_complete(dirs["emb"], dirs["cir"], dirs["mask"],
                           roster, encoders)
    banner("RULE 2b: CACHED MASKS vs THE CANONICAL NDVI")
    roster, mask_bad = drop_mask_ndvi_mismatches(
        roster, cube_dir, dirs["mask"], n_jobs)
    if mask_bad:
        reasons.update({c: f"mask_ndvi_mismatch: {w}"
                        for c, w in mask_bad.items()})
        paths = [os.path.join(cube_dir, c) for c in roster]

    gpu_note = ("built on Colab by notebooks/phase1_10_extreme_encoding.ipynb; "
                "this host has no CUDA and its dev venv is python 3.9.6, which "
                "cannot build dinov2_vitb14 at all")

    if args.max_cubes:
        stride = max(1, len(roster) // args.max_cubes)
        roster = roster[::stride][:args.max_cubes]
        paths = [os.path.join(cube_dir, c) for c in roster]
        say(f"\n[p3-extreme] *** SMOKE TEST: {len(roster)} cubes, STRIDED. "
            "The effective n is CUBES, so no number below is a result. ***")

    say(f"\n[p3-extreme] {len(encoders)} views, horizons {horizons}, "
        f"aggregations {aggregations}, fold modes {fold_modes}, "
        f"alpha rules {alpha_rules or 'all (unnarrowed)'}")
    say(f"[p3-extreme] n_jobs {n_jobs}, k {args.k}, "
        f"plausibility_screen {screen}")

    # --- rule 3 + 4 -------------------------------------------------------
    narrowed = []
    expected_rows = 0
    if args.skip_calibration:
        say("\n[rule 3] SKIPPED by --skip-calibration; no projection was made.")
    else:
        banner("RULE 3: THE TWO-POINT RUNTIME VALVE")
        points = calibrate(paths, cube_dir, dirs, encoders, horizons,
                           aggregations, fold_modes, alpha_rules, args.k,
                           n_jobs, screen, args.cal_small, args.cal_large)
        total_h, per_mode = project(points, args.cal_small, args.cal_large,
                                    len(paths), args.budget_hours)
        # The row count is fixed by the GRID, not the cube count, so the
        # calibration's own row total is the full run's total -- before any
        # narrowing trims it.
        expected_rows = points[args.cal_large]["rows"]
        banner("RULE 4: NARROWING, IF THE PROJECTION DEMANDS IT")
        fold_modes, aggregations, alpha_rules, narrowed, est = narrow(
            total_h, per_mode, fold_modes, aggregations, alpha_rules,
            args.budget_hours, n_cubes=len(paths))
        if not narrowed:
            say(f"[rule 4] projection {total_h:.2f} h is within the "
                f"{args.budget_hours:.1f} h budget; the FULL table runs.")
        else:
            say(f"[rule 4] narrowed to fold_modes={fold_modes}, "
                f"aggregations={aggregations}, alpha_rules={alpha_rules}; "
                f"projection now {est:.2f} h")
            say("[rule 4] ALL NINE ENCODER VIEWS SURVIVE.")

    if args.narrow:
        banner("RULE 4: NARROWING APPLIED FROM A PREVIOUS CLEAN CALIBRATION")
        if not args.narrow_reason:
            raise SystemExit(
                "\n--narrow requires --narrow-reason: a narrowing with no "
                "recorded provenance is indistinguishable from an arbitrary "
                "one, and it travels on every row of the table.\n")
        for axis in [a.strip() for a in args.narrow.split(",") if a.strip()]:
            if axis in fold_modes:
                fold_modes = tuple(m for m in fold_modes if m != axis)
                what = "fold_mode"
            elif axis in aggregations:
                aggregations = tuple(a for a in aggregations if a != axis)
                what = "aggregation"
            else:
                raise SystemExit(
                    f"\n--narrow {axis!r} matches no fold mode "
                    f"{fold_modes} or aggregation {aggregations}\n")
            narrowed.append(f"{what}={axis} dropped (not re-measured): "
                            f"{args.narrow_reason}")
            say(f"[rule 4] {narrowed[-1]}")
        say("[rule 4] ALL NINE ENCODER VIEWS SURVIVE.")

    if narrowed and args.csv_name == "p3_extreme_results.csv":
        args.csv_name = "p3_extreme_subset_results.csv"
        say(f"[rule 4] a narrowed table is a SUBSET table: writing "
            f"{args.csv_name}, and the full-table completeness assertions are "
            "NOT run on it.")

    # --- triggers-only: the fits already landed ---------------------------
    if args.triggers_only:
        import pandas as pd

        banner("TRIGGERS ONLY -- reading tables already on disk")
        cand = [os.path.join(out_root, n) for n in
                ("p3_extreme_subset_results.csv", args.csv_name)]
        csv_path = next((c for c in cand if os.path.exists(c)), None)
        if csv_path is None:
            raise SystemExit(f"\nno results table at any of {cand}\n")
        df = pd.read_csv(csv_path)
        say(f"[p3-extreme] read {csv_path} ({len(df)} rows)")
        if "narrowed" in df.columns and df["narrowed"].notna().any():
            narrowed = [str(df["narrowed"].dropna().iloc[0])]
            say(f"[p3-extreme] this table is NARROWED: {narrowed[0][:120]}")
        pred_path = _resolve_predictions(
            os.path.join(out_root, args.predictions_name))
        say(f"[p3-extreme] predictions at {pred_path}")
        _score_triggers(pred_path, out_root, args)
        report(df, args, roster, reasons, narrowed, dirs, gpu_note,
               encode_drops)
        banner("DONE")
        return

    # --- the run ----------------------------------------------------------
    banner("THE FULL GRID")
    t0 = time.time()
    manifest = _manifest_for(paths, verbose=True)
    from encoders.manifest import assert_strata_present, assert_weather_join
    assert_strata_present(manifest)
    join = assert_weather_join(manifest, cube_dir, verbose=False)
    assert max(join["max_abs_diff"].values()) == 0.0, join
    say(f"[p3-extreme] manifest {manifest.shape}, "
        f"{manifest.cube_id.nunique()} cubes, {time.time() - t0:.0f}s")

    pred_path = os.path.join(out_root, args.predictions_name)
    csv_path = os.path.join(out_root, args.csv_name)

    with EvaluateCounter(p3, total=expected_rows, label="") as counter:
        t0 = time.time()
        df, data = p3.run_p3(
            manifest, cube_dir, encoders=encoders, horizons=horizons,
            aggregations=aggregations, fold_modes=fold_modes,
            alpha_rules=alpha_rules, k=args.k, emb_dir=dirs["emb"],
            emb_dir_cir=dirs["cir"], mask_dir=dirs["mask"],
            plausibility_screen=screen, n_jobs=n_jobs,
            emit_predictions=not args.no_predictions,
            predictions_path=pred_path,
            log_path=log_path, verbose=True)
        wall = time.time() - t0
    say(f"\n[p3-extreme] {len(df)} rows in {wall / 60:.1f} min "
        f"({wall / 3600:.2f} h) on {n_jobs} workers")

    if narrowed:
        df = df.assign(narrowed="; ".join(narrowed))
    df = df.assign(tile=args.tile, roster_source=source,
                   n_cubes=len(roster),
                   cubes_excluded="; ".join(sorted(reasons)) or "")
    df.to_csv(csv_path, index=False)
    say(f"[p3-extreme] wrote {csv_path}")
    if args.no_predictions:
        say("[p3-extreme] predictions NOT emitted (--no-predictions); the "
            "results table is unaffected -- paired separability comes from "
            "payloads, not from the predictions file")
    else:
        pred_path = _resolve_predictions(pred_path)
        say(f"[p3-extreme] predictions at {pred_path}")

    # --- triggers ---------------------------------------------------------
    if args.skip_triggers or args.no_predictions:
        say("[p3-extreme] trigger metrics skipped (no predictions to score)")
    else:
        _score_triggers(pred_path, out_root, args)

    report(df, args, roster, reasons, narrowed, dirs, gpu_note)
    banner("DONE")


if __name__ == "__main__":
    main()
