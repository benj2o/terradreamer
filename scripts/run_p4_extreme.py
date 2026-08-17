"""Stage-A weather-attributability ceiling on ONE 2018 heat/drought tile.

WHAT THIS IS
------------
The extreme-tile P4 pilot from docs/CLAIM_IMPACT_EXPERIMENTS.md section B.3.
One tile of GreenEarthNet's ``extreme`` split, Stage A only, with the same
plausibility screen, fold modes and estimators as the screened 115-cube 32UNU
run, so the two tables sit side by side without a footnote.

P4 reads NO embeddings (see ``probes.p4_ceiling``'s module docstring), so this
is download plus CPU: no GPU, no encoder cache, no touching ``data/raw``.

Stage B is not attempted and is not expected to be available: the extreme split
is single-year 2018 event windows. That is REPORTED, never gated on -- Stage A
runs under the single-year proxy regime either way, which is the same regime
32UNU's number was computed in.

THE THREE RULES THIS SCRIPT APPLIES WITHOUT ASKING
---------------------------------------------------
Each one has a precedent in this repo, so each is encoded rather than raised.
All three announce themselves in the log and in the artefacts.

1. TILE. ``32UQC`` is preferred because it is the largest non-overlapping set
   (348 against ``32UNC``'s 315, both measured 2026-08-13). If its measured
   count falls below ``32UNC``'s, the reason for preferring it has evaporated,
   so the run takes ``32UNC`` and says so. Below ``MIN_USABLE_CUBES`` on both,
   nothing runs -- the download itself was wrong.

2. WEATHER COMPLETENESS. ``weather_full8`` is tried first. Off 32UNU the E-OBS
   stack has real, non-random gaps: on 30TVN ``eobs_fg`` is absent from 20 of
   25 cubes and ``eobs_qq`` misses the same trailing 13 days in every one. The
   registered fully-finite intersection ``weather_finite6`` is the precedented
   answer (scripts/validate_proxy_climatology.py), and NOTHING IS FILLED. Which
   set actually ran is printed in every table and written to every CSV row,
   because a ceiling computed on a different weather block is not comparable
   with 32UNU's unless that is flagged everywhere the two are printed together.

3. RUNTIME. Stage A's cost is NOT linear in cubes and the memo's "1.3-1.9
   CPU-hours by linear scaling" is the estimate this check exists to replace.
   ``loco`` is leave-one-cube-out, so its fold COUNT grows with the cube count
   while each fold's training set grows too, and HGB and the MLP are
   superlinear in rows on top of that. So the projection is MEASURED here, on
   this tile's own row counts, at two cube counts, and the growth exponent is
   read off the two rather than assumed. Over ``--budget-hours`` the run stops
   and reports the projection instead of burning the budget blind.

    .venv/bin/python -m scripts.run_p4_extreme --n-jobs 7
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np

#: The ``extreme`` split's four tiles, and their measured non-overlapping
#: capacity (docs/CLAIM_IMPACT_EXPERIMENTS.md, read from the bucket 2026-08-13).
EXTREME_TILES = {"32UMC": 246, "32UNC": 315, "32UPC": 312, "32UQC": 348}
PRIMARY_TILE = "32UQC"
FALLBACK_TILE = "32UNC"
SPLIT = "extreme"

#: Below this on both candidates, the download is wrong and no fallback helps.
MIN_USABLE_CUBES = 246

#: 32UNU, 115 cubes, screened (data/scaled_32UNU/p4_screened_results.csv).
#: Carried here only so the log can print the comparison it exists to make.
REF = {
    "tile": "32UNU", "n_cubes": 115, "n_frames": 1580, "frames_per_cube": 13.7,
    "cell_mean_rows": 25013, "implausible_frames": 3, "minutes": 36.2,
    "feature_set": "weather_full8",
    "cube": {"weather": 0.116, "observation": -0.004, "doy": 0.030,
             "margin_over_control": 0.120, "margin_over_doy": 0.086},
    "loco": {"weather": 0.096, "observation": -0.021, "doy": 0.030,
             "margin_over_control": 0.117, "margin_over_doy": 0.066},
}


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

class Tee:
    """stdout to the console AND to the run log, line-buffered, both flushed.

    A run measured in hours is polled, not watched. Anything that buffers puts
    the log minutes behind the process, and a log minutes behind the process
    cannot distinguish a slow fold from a hung one.
    """

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


#: Lines printed before the log path is known. The log lives under
#: ``data/scaled_<tile>/`` and the tile is chosen by rule 1, so rule 1's own
#: reasoning would otherwise be the one thing missing from the file that
#: records it. Buffered here and replayed into the log the moment it opens.
_PRELUDE: list = []
_TEE_OPEN = False


def say(msg: str = "") -> None:
    print(msg, flush=True)
    if not _TEE_OPEN:
        _PRELUDE.append(msg)


def open_tee(path: str):
    """Install the tee and replay everything said before it existed."""
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


class EvaluateCounter:
    """Wraps ``p4_ceiling.evaluate`` so the fitting loop reports where it is.

    ``run_stage_a`` calls the module-global ``evaluate`` once per output row.
    Counting there -- rather than inside the fold loop -- gives a unit of
    progress that maps 1:1 onto a row of the results table, so "row 138/270"
    means the same thing in the log as it does in the CSV.
    """

    def __init__(self, module, total: int, label: str = ""):
        self.module, self.total, self.label = module, total, label
        self.original = module.evaluate
        self.n = 0
        self.t0 = time.time()
        self.per_mode: dict = {}

    def __enter__(self):
        counter = self

        def evaluate(sources, target, data, mode, estimator, *a, **kw):
            t = time.time()
            out = counter.original(sources, target, data, mode, estimator,
                                   *a, **kw)
            dt = time.time() - t
            counter.n += 1
            counter.per_mode[mode] = counter.per_mode.get(mode, 0.0) + dt
            elapsed = time.time() - counter.t0
            per_row = elapsed / counter.n
            eta = per_row * (counter.total - counter.n)
            say(f"[p4-eta]{counter.label} row {counter.n}/{counter.total}  "
                f"{mode}/{estimator}  this {dt:6.1f}s  "
                f"elapsed {elapsed / 60:6.1f} min  {per_row:6.1f} s/row  "
                f"ETA {eta / 60:6.1f} min")
            return out

        self.module.evaluate = evaluate
        return self

    def __exit__(self, *exc):
        self.module.evaluate = self.original
        return False

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0


# ---------------------------------------------------------------------------
# Rule 1: which tile
# ---------------------------------------------------------------------------

def choose_tile(n: int, s3=None) -> tuple:
    """Measure both candidates' non-overlapping capacity, then pick.

    Listing is an ``ls``; downloading is a gigabyte. So the choice is made on
    the listing, before a byte is fetched.
    """
    from data.download_greenearthnet import (list_cubes, s3fs_client,
                                             select_non_overlapping)

    s3 = s3 or s3fs_client()
    counts, chosen = {}, {}
    for tile in (PRIMARY_TILE, FALLBACK_TILE):
        keys = list_cubes(tile=tile, split=SPLIT, s3=s3)
        picked = select_non_overlapping(keys, n=n)
        counts[tile] = (len(keys), len(picked))
        chosen[tile] = picked
        say(f"[tile] {tile}/{SPLIT}: {len(keys)} listed, {len(picked)} "
            f"non-overlapping (documented {EXTREME_TILES[tile]})")

    primary = counts[PRIMARY_TILE][1]
    fallback = counts[FALLBACK_TILE][1]
    if primary >= EXTREME_TILES[FALLBACK_TILE]:
        tile, why = PRIMARY_TILE, (
            f"{PRIMARY_TILE} returned {primary} non-overlapping cubes, at or "
            f"above {FALLBACK_TILE}'s documented {EXTREME_TILES[FALLBACK_TILE]}"
            "; it is still the largest set and the preferred tile stands")
    else:
        tile, why = FALLBACK_TILE, (
            f"RULE 1 FIRED: {PRIMARY_TILE} returned only {primary} "
            f"non-overlapping cubes, below {FALLBACK_TILE}'s documented "
            f"{EXTREME_TILES[FALLBACK_TILE]}. The only reason to prefer "
            f"{PRIMARY_TILE} was that it was the larger set, so the run falls "
            f"back to {FALLBACK_TILE} ({fallback} cubes)")
    say(f"[tile] {why}")
    n_usable = counts[tile][1]
    assert n_usable >= MIN_USABLE_CUBES, (
        f"{tile} yields {n_usable} non-overlapping cubes, below the "
        f"{MIN_USABLE_CUBES} floor (the smallest extreme tile in the memo is "
        f"32UMC at 246). Both candidates are short -- that is a broken listing "
        "or a changed bucket, not a tile choice."
    )
    return tile, chosen[tile], counts, why


def fetch(keys, out_dir: str, s3=None) -> list:
    """Download with a heartbeat. Skips what is already on disk."""
    from data.download_greenearthnet import s3fs_client

    s3 = s3 or s3fs_client()
    os.makedirs(out_dir, exist_ok=True)
    t0, paths, fetched = time.time(), [], 0
    for i, key in enumerate(keys, start=1):
        path = os.path.join(out_dir, os.path.basename(key))
        if not os.path.exists(path):
            tmp = path + ".partial"
            s3.download(key, tmp)
            os.replace(tmp, path)
            fetched += 1
        paths.append(path)
        if i % 25 == 0 or i == len(keys):
            dt = time.time() - t0
            mb = sum(os.path.getsize(p) for p in paths) / 1e6
            rate = i / dt if dt > 0 else float("nan")
            say(f"[dl]   {i}/{len(keys)} cubes ({fetched} new), {mb:.0f} MB, "
                f"{dt:.0f}s, {rate:.2f} cubes/s, "
                f"ETA {(len(keys) - i) / rate:.0f}s")
    return paths


# ---------------------------------------------------------------------------
# Pre-flight: cubes whose "clear" pixels are fill values
# ---------------------------------------------------------------------------

def _cube_targets_ok(path: str) -> dict:
    """Can ``cube_frame_targets`` build targets for this cube at all?

    NOT a quality judgement and NOT a new screen on the science. It re-runs the
    invariant ``cube_frame_targets`` already asserts -- a grid cell has finite
    NDVI exactly when it has clear pixels -- so a cube that would crash the run
    is found in one cheap parallel pass instead of 40 minutes into a fit.
    """
    import warnings

    from data.loader import CubeSample, cube_ndvi, load_cube
    from encoders.frames import (MIN_CLEAR_FRACTION, grid_clear_fraction,
                                 select_clear_frames)

    s = load_cube(path, verbose=False)
    sel = select_clear_frames(s.values, s.timestamps, s.mask,
                              min_clear=MIN_CLEAR_FRACTION, verbose=False)
    kept = CubeSample(values=sel.values, timestamps=sel.timestamps,
                      mask=sel.mask, path=s.path, bands=s.bands)
    nd = cube_ndvi(kept)
    T, H, W = nd.shape
    ch, cw = H // 4, W // 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells = np.nanmean(nd.reshape(T, 4, ch, 4, cw), axis=(2, 4))
    cell_valid = np.isfinite(cells.reshape(T, 16))
    gcf = grid_clear_fraction(sel.mask, grid=4)
    bad = cell_valid != (gcf > 0)
    n_bad = int(bad.sum())
    zero_px = 0
    if n_bad:
        b04 = np.asarray(kept.values[:, s.bands.index("B04")], dtype=float)
        b8a = np.asarray(kept.values[:, s.bands.index("B8A")], dtype=float)
        denom = np.abs(b8a + b04)
        ti, ci = np.where(bad)
        for t, c in zip(ti, ci):
            r0, c0 = (c // 4) * ch, (c % 4) * cw
            m = sel.mask[t, r0:r0 + ch, c0:c0 + cw]
            zero_px += int((m & (denom[t, r0:r0 + ch, c0:c0 + cw] < 1e-12)).sum())
    return {"cube": os.path.basename(path), "n_bad_cells": n_bad,
            "n_cells": int(cell_valid.size), "zero_reflectance_px": zero_px}


def exclude_unbuildable_cubes(manifest, cube_dir: str, n_jobs: int) -> tuple:
    """Drop cubes whose reflectance is FILL where the published mask says clear.

    WHAT THIS IS. GreenEarthNet's cloud mask reads s2_dlmask and s2_SCL and
    never looks at the reflectance bands. ``encoders.frames.finite_valid_mask``
    already demotes mask-valid pixels whose bands are NON-FINITE, for exactly
    that reason. It cannot catch the case found here: bands that are finite and
    EXACTLY ZERO -- a no-data fill block the mask calls clear. ``data.ndvi.ndvi``
    correctly returns NaN there (its |B8A+B04| < 1e-12 guard), so the cell has
    clear pixels and no finite NDVI, and ``cube_frame_targets`` refuses the cube.

    WHY EXCLUSION AND NOT A MASK FIX. The honest fix is a zero-reflectance rule
    beside ``finite_valid_mask``. That is a SHARED path: P1, P2, P3 and the
    published 32UNU tables all read masks through it, and changing it would move
    numbers this run has no mandate to move. Excluding the cubes touches nothing
    outside this run. NOTHING IS FILLED and nothing is repaired -- the affected
    cubes are removed whole and counted, which is the same posture the rest of
    the project takes towards data it cannot vouch for.

    THIS IS NOT ONE OF THE THREE PRE-AUTHORISED RULES. It is recorded in the
    CSV and called out in the report so it can be overruled.
    """
    from joblib import Parallel, delayed

    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    t0 = time.time()
    say(f"[preflight] checking {len(paths)} cubes can build targets at all "
        f"({n_jobs} workers)")
    res = Parallel(n_jobs=n_jobs)(delayed(_cube_targets_ok)(p) for p in paths)
    bad = [r for r in res if r["n_bad_cells"]]
    say(f"[preflight] {time.time() - t0:.0f}s; {len(bad)}/{len(paths)} cubes "
        "hold grid cells the mask calls clear but that have NO finite NDVI")
    if not bad:
        say("[preflight] nothing excluded")
        return manifest, [], ""
    total_cells = sum(r["n_cells"] for r in res)
    n_bad_cells = sum(r["n_bad_cells"] for r in bad)
    n_zero = sum(r["zero_reflectance_px"] for r in bad)
    for r in bad:
        say(f"[preflight]   {r['cube']}  {r['n_bad_cells']}/{r['n_cells']} "
            f"cells, {r['zero_reflectance_px']} clear pixels at "
            "|B8A+B04| < 1e-12")
    names = [r["cube"] for r in bad]
    before = manifest.cube_id.nunique()
    manifest = manifest[~manifest.cube_id.astype(str).isin(names)].reset_index(
        drop=True)
    why = (f"{len(bad)} of {len(paths)} cubes EXCLUDED: {n_bad_cells} of "
           f"{total_cells} grid cells ({n_bad_cells / total_cells * 100:.4f}%) "
           f"are marked clear by the published s2_dlmask/SCL conjunction yet "
           f"have no finite NDVI, because all {n_zero} of their clear pixels "
           "carry exactly-zero reflectance in B04 and B8A -- a no-data fill "
           "block the cloud mask does not flag. finite_valid_mask cannot demote "
           "them (the bands are finite, just zero). Nothing is filled and the "
           "shared mask path is unchanged; the cubes are dropped whole")
    say(f"[preflight] {why}")
    say(f"[preflight] cubes {before} -> {manifest.cube_id.nunique()}, "
        f"manifest now {manifest.shape}")
    return manifest, names, why


# ---------------------------------------------------------------------------
# Rule 2: which weather block
# ---------------------------------------------------------------------------

def weather_completeness(manifest, p4) -> dict:
    """Per-variable presence and finiteness across the manifest's rows.

    Measured on the MANIFEST rather than assumed from the tile, because the
    30TVN failure was not "the variable is missing" but "the variable is
    missing from 20 of 25 cubes, and a different one loses the same trailing 13
    days in all of them". Only a per-row count sees both.
    """
    out = {}
    n = len(manifest)
    for v in p4.EOBS_VARS:
        if v not in manifest.columns:
            out[v] = {"present": False, "finite_frac": 0.0, "n_cubes_with": 0}
            continue
        col = manifest[v].to_numpy(dtype=float)
        ok = np.isfinite(col)
        by_cube = manifest.assign(_ok=ok).groupby("cube_id")["_ok"].any()
        out[v] = {"present": True, "finite_frac": float(ok.sum()) / n,
                  "n_cubes_with": int(by_cube.sum())}
    return out


def choose_feature_sets(manifest, cube_dir: str, p4) -> tuple:
    """Try weather_full8; fall back to the registered finite intersection.

    Returns (manifest, feature_sets, headline_set, why). The manifest may come
    back with columns DROPPED -- nothing is ever filled, which is the whole
    point of a named finite subset rather than an imputation.
    """
    from encoders.manifest import assert_weather_join

    comp = weather_completeness(manifest, p4)
    n_cubes = manifest.cube_id.nunique()
    say(f"[weather] E-OBS completeness over {len(manifest)} rows / "
        f"{n_cubes} cubes (32UNU reference: all 8 present, 100% finite):")
    for v, c in comp.items():
        flag = "" if c["present"] and c["finite_frac"] == 1.0 else "   <-- GAP"
        say(f"[weather]   {v:<10} present={str(c['present']):<5} "
            f"finite {c['finite_frac'] * 100:6.2f}%  "
            f"in {c['n_cubes_with']:>3}/{n_cubes} cubes{flag}")

    complete = [v for v, c in comp.items()
                if c["present"] and c["finite_frac"] == 1.0]
    incomplete = [v for v in p4.EOBS_VARS if v not in complete]

    if not incomplete:
        try:
            join = assert_weather_join(manifest, cube_dir, verbose=False)
            assert max(join["max_abs_diff"].values()) == 0.0, join
        except AssertionError as exc:
            say(f"[weather] assert_weather_join REFUSED weather_full8: {exc}")
        else:
            why = ("all 8 E-OBS variables are present and 100% finite on this "
                   "tile and the join verifies exactly, so weather_full8 runs "
                   "and the ceiling is directly comparable with 32UNU's")
            say(f"[weather] RULE 2: weather_full8. {why}")
            return manifest, ("weather_full8", "weather_eowm5"), \
                "weather_full8", why

    # Fall back exactly as 30TVN did: DROP the incomplete columns, fill nothing.
    keep = set(p4.WEATHER_FINITE6)
    missing_from_finite6 = sorted(keep - set(complete))
    assert not missing_from_finite6, (
        f"the fully-finite intersection on this tile is {sorted(complete)}, "
        f"which does NOT cover weather_finite6 -- {missing_from_finite6} "
        "is incomplete here. weather_finite6 was registered against 30TVN's "
        "failure mode (eobs_fg absent, eobs_qq truncated) and this tile fails "
        "differently. Inventing a new named subset at runtime is exactly what "
        "probes.p4_ceiling's registry comment forbids, so this stops here "
        "rather than guessing."
    )
    drop = [v for v in incomplete if v in manifest.columns]
    if drop:
        manifest = manifest.drop(columns=drop)
    join = assert_weather_join(manifest, cube_dir, verbose=False)
    assert max(join["max_abs_diff"].values()) == 0.0, join
    why = (f"{incomplete} are incomplete on this tile, so the run uses the "
           "registered fully-finite intersection weather_finite6. NOTHING IS "
           "FILLED. This is the same fix 30TVN already forced, not a new "
           "decision -- and it means this ceiling is NOT directly comparable "
           "with 32UNU's 8-variable number")
    say(f"[weather] RULE 2 FIRED: weather_finite6. {why}")
    return manifest, ("weather_finite6", "weather_eowm5"), \
        "weather_finite6", why


# ---------------------------------------------------------------------------
# Rule 3: what this will actually cost
# ---------------------------------------------------------------------------

def stage_a_once(manifest, cube_dir, p4, feature_sets, n_cubes, k, n_jobs,
                 label):
    """One Stage-A pass over a cube subset, timed by fold mode."""
    import pandas as pd  # noqa: F401  (p4 returns a frame)

    # Drawn from the MANIFEST, not from a glob of the directory: the pre-flight
    # exclusion lives in the manifest, and a glob would quietly re-admit a cube
    # the run has already refused.
    sub_ids = set(sorted(manifest.cube_id.astype(str).unique())[:n_cubes])
    sub = manifest[manifest.cube_id.astype(str).isin(sub_ids)].reset_index(
        drop=True)
    say(f"\n[cal]{label} {sub.cube_id.nunique()} cubes, {len(sub)} frames")

    t0 = time.time()
    data = p4.build_p4_data(sub, cube_dir, feature_sets=feature_sets,
                            verbose=False, plausibility_screen=True)
    t_build = time.time() - t0
    say(f"[cal]{label} build_p4_data {t_build:.0f}s")

    with EvaluateCounter(p4, total=270, label=label) as counter:
        df = p4.run_stage_a(data, feature_sets=feature_sets, k=k,
                            n_jobs=n_jobs, verbose=False)
    return {"n_cubes": int(sub.cube_id.nunique()), "n_frames": int(len(sub)),
            "cell_rows": int(data.targets["cell_mean"].n_rows),
            "t_build": t_build, "t_fit": counter.elapsed,
            "per_mode": dict(counter.per_mode), "n_rows": len(df)}


def project(small: dict, large: dict, n_full: int) -> dict:
    """Fit one power law PER FOLD MODE from two measured cube counts.

    Why per fold mode and why a measured exponent. ``cube`` and
    ``spatial_block`` hold k=5 folds fixed and only grow the rows, so their cost
    grows like rows^p with p >= 1 for HGB and the MLP. ``loco`` is
    leave-one-cube-out: the fold COUNT grows with the cube count as well, so its
    cost grows like N x rows^p -- roughly one whole power faster. Fitting a
    single pooled exponent would average those two behaviours into a number that
    describes neither, and assuming p = 1 anywhere is the linear-scaling error
    this whole check exists to catch.
    """
    ratio = large["n_cubes"] / small["n_cubes"]
    out = {"ratio": ratio, "modes": {}, "t_fit": 0.0}
    for mode in sorted(set(small["per_mode"]) | set(large["per_mode"])):
        a = small["per_mode"].get(mode, float("nan"))
        b = large["per_mode"].get(mode, float("nan"))
        p = math.log(b / a) / math.log(ratio) if a > 0 and b > 0 else float("nan")
        full = b * (n_full / large["n_cubes"]) ** p
        out["modes"][mode] = {"small": a, "large": b, "exponent": p,
                              "projected": full}
        out["t_fit"] += full
    # Cube loading and target assembly are linear in cubes and measured as one.
    out["t_build"] = large["t_build"] * (n_full / large["n_cubes"])
    out["total"] = out["t_fit"] + out["t_build"]
    out["naive_linear"] = ((large["t_fit"] + large["t_build"])
                           * (n_full / large["n_cubes"]))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def headline(df, feature_set: str, tile: str, n_cubes: int) -> None:
    """cell_mean / HGB / <the set that actually ran>, beside 32UNU's."""
    banner(f"HEADLINE -- {tile} {n_cubes} cubes, Stage A, screened, "
           f"cell_mean / HGB / {feature_set}")
    if feature_set != REF["feature_set"]:
        say(f"*** WEATHER BLOCK DIFFERS: this run used {feature_set}, 32UNU "
            f"used {REF['feature_set']}. The two ceilings are NOT directly")
        say("*** comparable -- a different weather block is a different "
            "predictor set, not a different tile.")
    say(f"  {'mode/kind':<28} {tile + ' R2':>12} {'obs margin':>11} "
        f"{'DOY margin':>11} | {'32UNU R2':>10} {'obs':>8} {'doy':>8}")
    for mode in ("cube", "loco"):
        for kind in ("weather", "observation", "doy",
                     "weather_plus_observation", "permutation"):
            q = ((df.stage == "A") & (df.target == "cell_mean")
                 & (df.fold_mode == mode) & (df.estimator == "hgb")
                 & (df.feature_set == feature_set) & (df.model_kind == kind))
            sub = df[q]
            assert len(sub) == 1, f"headline not unique for {mode}/{kind}"
            r = sub.iloc[0]
            ref = REF[mode]
            old = (f"{ref[kind]:+10.3f} {ref['margin_over_control']:+8.3f} "
                   f"{ref['margin_over_doy']:+8.3f}"
                   if kind == "weather" else
                   f"{ref.get(kind, float('nan')):+10.3f} {'':>8} {'':>8}")
            say(f"  {mode + '/' + kind:<28} "
                f"{r.r2_vs_climatology_mean:+12.3f} "
                f"{r.margin_over_control:+11.3f} "
                f"{r.margin_over_doy:+11.3f} | {old}")
        r = df[(df.stage == "A") & (df.target == "cell_mean")
               & (df.fold_mode == mode) & (df.estimator == "hgb")
               & (df.feature_set == feature_set)
               & (df.model_kind == "weather")].iloc[0]
        say(f"  {'':<28} {mode} weather 95% CI "
            f"[{r.r2_vs_climatology_ci_lo:+.3f}, "
            f"{r.r2_vs_climatology_ci_hi:+.3f}] over {int(r.n_folds)} folds, "
            f"effective n {int(r.effective_n)} cubes")


def controls(df, feature_set: str) -> None:
    banner("CONTROLS -- does weather beat day-of-year, or only echo it?")
    say("If the DOY control sits close behind the weather rows, the ceiling "
        "reads as\nseasonal timing recovered through weather, i.e. CONFOUNDED "
        "rather than climate-driven.\n")
    say(f"  {'target/mode/est':<40} {'weather':>9} {'doy':>9} "
        f"{'margin':>9} {'doy/weather':>12}")
    for target in ("cell_mean", "cube_mean"):
        for mode in ("cube", "loco"):
            for est in ("hgb", "linear", "mlp"):
                base = ((df.stage == "A") & (df.target == target)
                        & (df.fold_mode == mode) & (df.estimator == est)
                        & (df.feature_set == feature_set))
                w = df[base & (df.model_kind == "weather")].iloc[0]
                d = df[base & (df.model_kind == "doy")].iloc[0]
                share = (d.r2_vs_climatology_mean / w.r2_vs_climatology_mean
                         if w.r2_vs_climatology_mean > 0 else float("nan"))
                say(f"  {target + '/' + mode + '/' + est:<40} "
                    f"{w.r2_vs_climatology_mean:+9.3f} "
                    f"{d.r2_vs_climatology_mean:+9.3f} "
                    f"{w.margin_over_doy:+9.3f} {share:11.0%}")
    say(f"\n  32UNU screened reference, cell_mean/HGB: "
        f"cube weather {REF['cube']['weather']:+.3f} vs doy "
        f"{REF['cube']['doy']:+.3f} (margin "
        f"{REF['cube']['margin_over_doy']:+.3f}, doy is "
        f"{REF['cube']['doy'] / REF['cube']['weather']:.0%} of weather); "
        f"loco weather {REF['loco']['weather']:+.3f} vs doy "
        f"{REF['loco']['doy']:+.3f} (margin "
        f"{REF['loco']['margin_over_doy']:+.3f})")

    w = df[(df.stage == "A") & (df.model_kind == "weather")]
    crosses = ((w.r2_vs_climatology_ci_lo <= 0)
               & (w.r2_vs_climatology_ci_hi >= 0)).sum()
    say(f"\n  weather rows whose fold-clustered interval includes zero: "
        f"{int(crosses)}/{len(w)}")
    say(f"  weather rows at or below the observation control: "
        f"{int((w.margin_over_control <= 0).sum())}/{len(w)}")
    say(f"  weather rows at or below the DOY control: "
        f"{int((w.margin_over_doy <= 0).sum())}/{len(w)}")
    perm = df[(df.stage == "A") & (df.model_kind == "permutation")]
    say(f"  permutation control, max r2_vs_clim: "
        f"{perm.r2_vs_climatology_mean.max():+.3f}   "
        "(must stay <= 0: the pipeline never invents skill)")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tile", default=None,
                    help="skip rule 1 and force a tile")
    ap.add_argument("--n", type=int, default=400,
                    help="cubes to ASK for; the 64 px non-overlap rule caps it")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=0, help="0 = cpu_count - 1")
    ap.add_argument("--budget-hours", type=float, default=6.0)
    ap.add_argument("--calibrate", default="20,40",
                    help="two cube counts; the growth exponent is read off both")
    ap.add_argument("--out", default=None)
    ap.add_argument("--csv-name", default="p4_extreme_results.csv")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    t_run = time.time()
    n_jobs = args.n_jobs or max(1, (os.cpu_count() or 2) - 1)

    from data.download_greenearthnet import s3fs_client, select_non_overlapping
    s3 = s3fs_client()

    banner("EXTREME-TILE P4 PILOT -- Stage A weather-attributability ceiling")
    say(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}, {n_jobs} workers, "
        f"budget {args.budget_hours:.1f} CPU-hours")

    # -- rule 1 ------------------------------------------------------------
    if args.tile:
        from data.download_greenearthnet import list_cubes
        tile = args.tile
        keys = select_non_overlapping(
            list_cubes(tile=tile, split=SPLIT, s3=s3), n=args.n)
        why_tile = f"tile forced to {tile} on the command line"
        say(f"[tile] {why_tile}: {len(keys)} non-overlapping")
    else:
        tile, keys, counts, why_tile = choose_tile(args.n, s3=s3)

    out_root = args.out or os.path.join("data", f"scaled_{tile}")
    cube_dir = os.path.join(out_root, "raw")
    os.makedirs(out_root, exist_ok=True)

    log_path = os.path.join(out_root, "p4_extreme_run.log")
    open_tee(log_path)
    say(f"\n[log] tee -> {log_path} (everything above was replayed into it)")
    say(f"[tile] SELECTED {tile}: {why_tile}")

    csv = os.path.join(out_root, args.csv_name)
    assert args.overwrite or not os.path.exists(csv), (
        f"{csv} already exists; pass --overwrite for an intentional rerun")

    # -- download ----------------------------------------------------------
    banner(f"DOWNLOAD -- {len(keys)} non-overlapping cubes of {tile}/{SPLIT}",
           "-")
    fetch(keys, cube_dir, s3=s3)

    from data.loader import load_cube
    from encoders.manifest import assert_strata_present, build_manifest
    from probes import p4_ceiling as p4

    paths = sorted(glob.glob(os.path.join(cube_dir, "*.nc")))
    banner(f"MANIFEST -- {len(paths)} cubes on disk", "-")
    t0 = time.time()
    manifest = build_manifest([load_cube(p, verbose=False) for p in paths],
                              verbose=False)
    assert_strata_present(manifest)
    import pandas as pd
    ts = pd.to_datetime(manifest.timestamp)
    doy = ts.dt.dayofyear
    say(f"[manifest] {manifest.shape}, {manifest.cube_id.nunique()} cubes, "
        f"built in {time.time() - t0:.0f}s")
    say(f"[manifest] {len(manifest) / manifest.cube_id.nunique():.1f} "
        f"frames/cube (32UNU screened: {REF['frames_per_cube']})")
    say(f"[manifest] years {sorted(ts.dt.year.unique().tolist())}, "
        f"day-of-year span {int(doy.min())}-{int(doy.max())}, "
        f"median {int(doy.median())}")
    info = p4.detect_seasonal_split(manifest, verbose=True)
    say("[manifest] Stage B is REPORTED, not gated on: Stage A runs under the "
        "single-year\n[manifest] proxy regime either way, which is the regime "
        "32UNU's number was computed in.")

    # -- pre-flight --------------------------------------------------------
    banner("PRE-FLIGHT -- can every cube build targets?", "-")
    manifest, excluded, why_excluded = exclude_unbuildable_cubes(
        manifest, cube_dir, n_jobs)
    if excluded:
        say(f"[preflight] frames/cube after exclusion: "
            f"{len(manifest) / manifest.cube_id.nunique():.1f}")

    # -- rule 2 ------------------------------------------------------------
    banner("WEATHER COMPLETENESS", "-")
    manifest, feature_sets, headline_set, why_weather = choose_feature_sets(
        manifest, cube_dir, p4)

    # -- rule 3 ------------------------------------------------------------
    sizes = [int(x) for x in args.calibrate.split(",")]
    n_full = int(manifest.cube_id.nunique())
    banner(f"RUNTIME CALIBRATION -- Stage A at {sizes} cubes, "
           f"to project {n_full}", "-")
    say("Measured on THIS tile's row counts. The memo's linear extrapolation "
        "from 32UNU\nis exactly what this replaces: loco's fold count grows "
        "with the cube count.")
    cal = [stage_a_once(manifest, cube_dir, p4, feature_sets, s, args.k,
                        n_jobs, f"[{s}c]") for s in sizes]
    for c, s in zip(cal, sizes):
        say(f"[cal] {s:>3} cubes: {c['n_frames']:>5} frames, "
            f"{c['cell_rows']:>6} cell_mean rows, build {c['t_build']:.0f}s, "
            f"fit {c['t_fit'] / 60:.1f} min, "
            f"{c['t_fit'] / 270:.2f} s/row")

    proj = project(cal[0], cal[1], n_full)
    banner(f"PROJECTION to {n_full} cubes", "-")
    say(f"  {'fold mode':<16} {f'{sizes[0]}c':>9} {f'{sizes[1]}c':>9} "
        f"{'exponent':>9} {'projected':>12}")
    for mode, m in proj["modes"].items():
        say(f"  {mode:<16} {m['small']:>8.0f}s {m['large']:>8.0f}s "
            f"{m['exponent']:>9.2f} {m['projected'] / 3600:>10.2f} h")
    say(f"  {'build (linear)':<16} {'':>9} "
        f"{cal[1]['t_build']:>8.0f}s {1.0:>9.2f} "
        f"{proj['t_build'] / 3600:>10.2f} h")
    say(f"\n  PROJECTED FULL-TILE STAGE A: {proj['total'] / 3600:.2f} "
        f"CPU-hours on {n_jobs} workers")
    say(f"  (naive linear-in-cubes projection, for contrast: "
        f"{proj['naive_linear'] / 3600:.2f} h)")
    say(f"  budget: {args.budget_hours:.2f} CPU-hours")

    if proj["total"] / 3600 > args.budget_hours:
        banner("STOP -- RULE 3 FIRED. The full run is NOT started.")
        say(f"Projected {proj['total'] / 3600:.2f} CPU-hours against a "
            f"{args.budget_hours:.1f}-hour budget.")
        say("This is the cost model being wrong, not a routine branch: the "
            "memo's\n1.3-1.9 h estimate assumed linear scaling in cubes, and "
            "the measured\nexponents above say otherwise. Reported rather than "
            "run blind, to protect\nthe Aug 18 budget.")
        say(f"\nThe {len(paths)} cubes are downloaded and staged at {cube_dir}; "
            "a narrowed\nrerun (fewer cubes, or loco dropped) needs no new "
            "download.")
        say(f"\ntotal elapsed {(time.time() - t_run) / 60:.1f} min")
        return

    say("\nUnder budget. Continuing straight through to the full run.")

    # -- the run -----------------------------------------------------------
    banner(f"FULL RUN -- {n_full} cubes, Stage A, plausibility screen ON")
    t0 = time.time()
    data = p4.build_p4_data(manifest, cube_dir, feature_sets=feature_sets,
                            verbose=True, plausibility_screen=True)
    say(f"[run] build_p4_data {time.time() - t0:.0f}s")
    p4.print_doy_weather_collinearity(manifest, data.weather[feature_sets[0]])

    t0 = time.time()
    with EvaluateCounter(p4, total=270) as counter:
        df = p4.run_stage_a(data, feature_sets=feature_sets, k=args.k,
                            n_jobs=n_jobs, verbose=False)
    say(f"\n[run] run_stage_a: {len(df)} rows in "
        f"{(time.time() - t0) / 60:.1f} min on {n_jobs} workers "
        f"(projected {proj['t_fit'] / 60:.1f} min)")

    df = p4.add_margins(df, verbose=True)
    # Stage B never ran; the deferral check reads that off the manifest and
    # refuses a table that relabelled Stage A as H1.
    p4.assert_stage_b_ran_or_deferred(df, info)

    # Which weather block ran, on every row, so the CSV cannot be read next to
    # 32UNU's without the difference being visible.
    df["weather_feature_set_headline"] = headline_set
    df["weather_feature_set_reason"] = why_weather
    df["tile"] = tile
    df["tile_reason"] = why_tile
    df["n_cubes_excluded_fill"] = len(excluded)
    df["cubes_excluded_fill"] = ";".join(excluded)
    df["cubes_excluded_reason"] = why_excluded or "none_excluded"

    df.to_csv(csv, index=False)
    say(f"[run] wrote {csv}; invariants run next")

    p4.assert_results_complete(df, feature_sets=feature_sets)
    p4.assert_plausibility_screen_declared(df, required=True)
    back = pd.read_csv(csv)
    assert back.shape == df.shape, (back.shape, df.shape)
    p4.assert_results_complete(back, feature_sets=feature_sets)
    p4.assert_stage_b_ran_or_deferred(back, info)
    p4.assert_plausibility_screen_declared(back, required=True)
    say("[run] all invariants pass in memory and on the CSV")

    # -- report ------------------------------------------------------------
    banner("SCREEN GEOMETRY", "-")
    say(f"  {int(back.n_implausible_frames.iloc[0])} implausible frames of "
        f"{len(manifest)} ({int(back.n_implausible_frames.iloc[0]) / len(manifest) * 100:.2f}%); "
        f"32UNU: {REF['implausible_frames']}/{REF['n_frames']} "
        f"({REF['implausible_frames'] / REF['n_frames'] * 100:.2f}%)")
    for target in p4.TARGETS:
        say(f"  {target:<12} -"
            f"{int(back.loc[back.target == target, 'n_rows_dropped_implausible'].iloc[0])} rows")

    headline(df, headline_set, tile, n_full)
    controls(df, headline_set)

    banner("RUN COMPLETE")
    say(f"tile {tile}, {n_full} cubes, {len(manifest)} frames, "
        f"weather block {headline_set}")
    say(f"artefacts: {csv}")
    say(f"           {log_path}")
    say(f"total elapsed {(time.time() - t_run) / 60:.1f} min")


if __name__ == "__main__":
    main()
