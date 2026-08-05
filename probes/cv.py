"""THE single definition of leakage-safe evaluation splits.

Any number produced outside this module does not exist. Do not write an ad-hoc
train/test split anywhere else in the repo; import from here.

WHAT A SPLIT OPERATES ON
------------------------
The (cube, frame) manifest from ``encoders.manifest.build_manifest``: one row
per RETAINED frame, with cube_id / tile / year / timestamp /
original_axis_index / pixel_bbox / clear_frac (and more). Every fold generator
yields ``(train_idx, test_idx)``: integer POSITIONS into the manifest's rows
(``df.iloc`` indexing), never pandas index labels.

JOIN CONTRACT
-------------
A probe joins a manifest row to an embedding row on

    (cube_id, original_axis_index) == (cube, kept_idx)

against the ``.npz`` files under ``data/phase1_2/embeddings/``. Use
``join_embeddings`` for that join: it asserts the contract on every field the
two sides share (kept_idx, timestamps, clear_frac) and carries
``window_span_days`` through to the caller. The multi-image encoder's lookback
is a variable number of DAYS (measured on this subset: min 0, median 55,
max 105) and correlates with cloud, so it is a covariate a probe must be able
to condition on; this module must not silently drop it.

THE TWO LEAKAGE RULES, AND WHERE THEY COLLIDE
---------------------------------------------
1. Frames of the same CUBE must never appear on both sides of a fold
   (spatial/identity leakage). Enforced INSIDE every generator, whatever mode
   was requested.
2. The same YEAR should not appear on both sides when the claim is cross-year
   (temporal leakage). This is the climatology's leave-target-year-out rule.

On the extreme split (the current 20 cubes) cube and year are perfectly
confounded: all cubes are 2018, so grouping by cube IS grouping by year. On
the seasonal split ONE cube internally spans 2017-2020, so grouping by year
would split WITHIN a cube -- rule 1 and rule 2 collide. The resolution is the
``crossed`` mode: a test fold contains only (cube, year) pairs whose cube AND
whose year are both unseen in train. That preserves both rules simultaneously
and matches GreenEarthNet's own leave-target-year-out protocol in
``data.climatology``, so the CV is consistent with the climatology baseline
rather than fighting it.

MODES, AND WHAT RUNS ON THE CURRENT SUBSET
------------------------------------------
    cube          DEFAULT. Group k-fold with cube_id as the group.
    crossed       cube AND year jointly held out. P4's mode whenever the
                  manifest spans more than one year. RAISES on a single year.
    year          year as the group. RAISES on a single year; RAISES via the
                  same-cube check when a cube spans years (seasonal split).
    tile          whole geographic tiles held out. RAISES on a single tile.
    spatial_block prototype-scale SUBSTITUTE for tile: cluster cubes by
                  pixel_bbox distance within the tile, hold out whole
                  clusters. True tile holdout is deferred to scale-up.
    temporal      train strictly before / test strictly after a cutoff.
                  NOT a default; a robustness variant for P3 only.

The 20-cube prototype is one tile (32UNU), one year (2018). Therefore ``cube``,
``spatial_block`` and ``temporal`` run, and ``year``, ``tile`` and ``crossed``
all correctly RAISE. A raise from those three here is designed behaviour, not
a bug -- do not work around it with a random split, which would put the same
season on both sides and inflate every number.

If a stratified variant is ever added: the per-cell strata have a thin tail
(over 320 cells: cropland 127, tree_cover 99, grassland 88, built_up 5,
bare_sparse 1). Stratified logic must cover ``REPLICATION_STRATA`` only
(cropland / tree_cover / grassland) and report-but-exclude the other two: a
"replication" over 1 cell is not a replication.
"""

from __future__ import annotations

import os

import numpy as np

from data.climatology import SingleYearError
from encoders.frames import MIN_CLEAR_FRACTION

__all__ = [
    # Phase 1.2-era grouping keys, kept as-is.
    "cube_years", "assert_multi_year", "year_groups", "SingleYearError",
    # Phase 1.3 fold generators and their machinery.
    "LeakageError", "SingleTileError", "LOCO", "MODES", "REPLICATION_STRATA",
    "filter_clear", "folds", "cube_folds", "leave_one_cube_out",
    "crossed_folds", "year_folds", "tile_folds", "spatial_block_folds",
    "temporal_folds", "join_embeddings",
]

# The k sentinel for leave-one-cube-out: the honest choice at 20-cube scale,
# where k=5 leaves only 4 cubes per test fold.
LOCO = "loco"

MODES = ("cube", "crossed", "year", "tile", "spatial_block", "temporal")

# Per-stratum replication covers these three ONLY. built_up (5 cells) and
# bare_sparse (1 cell) are reported for completeness and excluded from any
# replication claim.
REPLICATION_STRATA = ("cropland", "tree_cover", "grassland")

_REQUIRED_COLUMNS = ("cube_id", "tile", "year", "timestamp",
                     "original_axis_index", "pixel_bbox", "clear_frac")


class LeakageError(RuntimeError):
    """Raised when a split would put frames of one cube on both sides."""


class SingleTileError(RuntimeError):
    """Raised when a cross-tile quantity is asked of a single-tile subset."""


# ---------------------------------------------------------------------------
# Phase 1.2-era grouping keys. Kept verbatim; the generators below build
# around them.
# ---------------------------------------------------------------------------

def cube_years(paths) -> np.ndarray:
    """Acquisition year per cube, parsed from the GreenEarthNet cube id.

    Ids look like 32UNU_2018-03-09_2018-08-05_1081_1209_3641_3769_16_96_56_136
    where field 1 is the window start.
    """
    years = []
    for p in paths:
        stem = os.path.basename(str(p))
        stem = stem[:-3] if stem.endswith(".nc") else stem
        parts = stem.split("_")
        assert len(parts) > 2, f"cannot parse a year from {stem!r}"
        years.append(int(parts[1][:4]))
    out = np.asarray(years, dtype=int)
    assert out.shape == (len(list(paths)),) or out.size == len(years)
    return out


def assert_multi_year(paths) -> np.ndarray:
    """Raise unless the manifest spans more than one year.

    The year-leakage check is deferred to scale-up: every cube in tile 32UNU is
    from 2018. Failing loudly here is the point. A year-grouped split that
    silently degenerates to a single group would report a leakage-free number
    that had never been tested for year leakage at all.
    """
    years = cube_years(paths)
    unique = sorted(set(years.tolist()))
    if len(unique) < 2:
        raise SingleYearError(
            f"year-grouped CV needs at least 2 years, the manifest has {unique}. "
            f"{len(years)} cubes, all from {unique[0]}. Every cube in tile 32UNU "
            "is from 2018, so this is expected on the current subset and the "
            "year-leakage check is DEFERRED TO SCALE-UP. Use spatial grouping "
            "for Phase 1.3. Do not fall back to a random split: it would put the "
            "same season on both sides."
        )
    return years


def year_groups(paths) -> np.ndarray:
    """Group labels for year-grouped CV, one per cube.

    Raises SingleYearError on a single-year manifest. See assert_multi_year.
    """
    years = assert_multi_year(paths)
    print(f"[cv] year groups: {len(years)} cubes over {sorted(set(years.tolist()))}")
    return years


# ---------------------------------------------------------------------------
# Manifest validation and the clear-fraction filter
# ---------------------------------------------------------------------------

def _validate_manifest(df) -> None:
    """Refuse a manifest that could corrupt a split, before any fold exists."""
    assert len(df), "empty manifest"
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, (
        f"manifest is missing {missing}; build it with "
        "encoders.manifest.build_manifest, do not hand-roll one"
    )
    dup = df.duplicated(subset=["cube_id", "timestamp"])
    if dup.any():
        rows = df.loc[dup, ["cube_id", "timestamp"]].head(3)
        raise LeakageError(
            f"{int(dup.sum())} duplicate (cube_id, timestamp) rows in the "
            f"manifest, e.g.\n{rows}\nA duplicated frame lands in every fold "
            "its copies are assigned to, which silently double-counts it and "
            "can place the same observation on both sides. Deduplicate at the "
            "source (build_manifest over each cube exactly once)."
        )
    # NOT year: a seasonal-split cube legitimately spans years, and a correct
    # manifest carries the per-row year. The year/id consistency check lives
    # in _row_years, where year-aware modes enforce it strictly.
    for col in ("tile", "pixel_bbox"):
        per_cube = df.groupby("cube_id")[col].nunique()
        bad = per_cube[per_cube > 1]
        assert bad.empty, (
            f"{col} is not constant within a cube for {list(bad.index)[:3]}; "
            "the manifest is inconsistent and no grouping key is trustworthy"
        )


def _timestamps(df) -> np.ndarray:
    ts = np.asarray(df["timestamp"].to_numpy(), dtype="datetime64[ns]")
    assert ts.shape == (len(df),)
    return ts


def _row_years(df, strict: bool, context: str) -> np.ndarray:
    """Per-ROW year from the timestamp, cross-checked against the year column.

    The manifest's ``year`` column is parsed from the cube id (the window
    START year). On the seasonal split one cube spans 2017-2020, so the two
    can disagree. Year-aware modes must split on what the frame actually is,
    and a silent disagreement is exactly the kind of wrong label that produces
    a confident, leaky split -- so in those modes (``strict=True``) a mismatch
    RAISES, naming the fix (per-row year in build_manifest), rather than
    quietly preferring either column.
    """
    from_ts = _timestamps(df).astype("datetime64[Y]").astype(int) + 1970
    from_col = df["year"].to_numpy().astype(int)
    n_bad = int((from_ts != from_col).sum())
    if n_bad and strict:
        raise AssertionError(
            f"{context}: {n_bad}/{len(df)} rows have year (from cube id) != "
            "calendar year of their timestamp. This happens when a cube spans "
            "a year boundary (the seasonal split). Fix build_manifest to "
            "derive year per row from the timestamp before using a year-aware "
            "mode; splitting on the wrong label would leak."
        )
    if n_bad:
        print(f"[cv] WARNING: {n_bad}/{len(df)} rows have year != timestamp "
              f"year ({context}); year-aware modes will refuse this manifest")
    assert from_ts.shape == (len(df),)
    return from_ts


def filter_clear(df, min_clear: float = MIN_CLEAR_FRACTION,
                 verbose: bool = True) -> np.ndarray:
    """Row POSITIONS surviving the frame-level clear-fraction filter.

    Strictly greater-than, matching ``encoders.frames.select_clear_frames``.
    ``min_clear`` is configurable UPWARD only: the embeddings were encoded
    under > MIN_CLEAR_FRACTION, so a looser cut would select manifest rows for
    which no embedding exists. A stricter cut uses the stored per-frame
    clear_frac and needs no re-encoding.
    """
    assert min_clear >= MIN_CLEAR_FRACTION, (
        f"min_clear={min_clear} is below the encoding threshold "
        f"{MIN_CLEAR_FRACTION}; embeddings only exist for frames with "
        f"clear_frac > {MIN_CLEAR_FRACTION}, so a looser filter would select "
        "rows that have no embedding. Configurable upward only."
    )
    cf = df["clear_frac"].to_numpy().astype(float)
    assert cf.shape == (len(df),)
    assert np.isfinite(cf).all() and (cf > 0).all() and (cf <= 1).all(), (
        "clear_frac outside (0, 1] in the manifest"
    )
    keep = np.flatnonzero(cf > min_clear)
    if verbose:
        print(f"[cv] clear_frac filter (> {min_clear}): kept {keep.size}/{len(df)} "
              f"rows, removed {len(df) - keep.size}")
    assert keep.size, (
        f"no manifest row has clear_frac > {min_clear}; loosen the filter "
        f"(not below {MIN_CLEAR_FRACTION}) or check the manifest"
    )
    return keep


# ---------------------------------------------------------------------------
# Shared fold machinery
# ---------------------------------------------------------------------------

def _group_kfold_assign(labels: np.ndarray, k: int) -> dict:
    """Deterministic balanced group assignment: {group label: fold index}.

    Greedy row-count balancing (largest group first, into the currently
    lightest fold), which is sklearn's GroupKFold strategy -- implemented here
    with numpy so the split definition has no dependency this repo does not
    already carry, and no RNG: the same manifest always yields the same folds.
    """
    uniq, counts = np.unique(labels, return_counts=True)
    assert 2 <= k <= uniq.size, (
        f"k={k} needs 2 <= k <= n_groups={uniq.size} (groups: "
        f"{uniq[:5].tolist()}{'...' if uniq.size > 5 else ''})"
    )
    order = np.argsort(-counts, kind="stable")   # largest first; ties keep label order
    loads = np.zeros(k, dtype=int)
    fold_of = {}
    for j in order:
        i = int(np.argmin(loads))
        fold_of[uniq[j]] = i
        loads[i] += int(counts[j])
    assert len(fold_of) == uniq.size and (loads > 0).all()
    return fold_of


def _clamp_k(k: int, n_groups: int, unit: str, verbose: bool) -> int:
    if k > n_groups:
        if verbose:
            print(f"[cv] k clamped {k} -> {n_groups}: only {n_groups} {unit} "
                  "groups exist")
        return n_groups
    return k


def _emit(mode: str, i: int, k: int, df, train_idx: np.ndarray,
          test_idx: np.ndarray, verbose: bool, leak_hint: str = "") -> tuple:
    """Final gate every fold passes through, whatever mode built it.

    Asserts row disjointness and CUBE disjointness (raising LeakageError), and
    prints the per-fold summary: n rows, n cubes and the years on each side.
    """
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    assert train_idx.size and test_idx.size, (
        f"{mode} fold {i + 1}/{k}: empty side (train {train_idx.size}, "
        f"test {test_idx.size})"
    )
    assert not np.intersect1d(train_idx, test_idx).size, (
        f"{mode} fold {i + 1}/{k}: row indices overlap"
    )
    tr_cubes = set(df["cube_id"].to_numpy()[train_idx].tolist())
    te_cubes = set(df["cube_id"].to_numpy()[test_idx].tolist())
    both = sorted(tr_cubes & te_cubes)
    if both:
        raise LeakageError(
            f"{mode} fold {i + 1}/{k} puts frames of {len(both)} cube(s) on "
            f"BOTH sides, e.g. {both[0]!r}. Frames of one cube must never be "
            f"split across a fold, whatever mode asked for it. {leak_hint}"
        )
    years = _timestamps(df).astype("datetime64[Y]").astype(int) + 1970
    tr_years = sorted(set(years[train_idx].tolist()))
    te_years = sorted(set(years[test_idx].tolist()))
    if verbose:
        print(f"[cv] {mode} fold {i + 1}/{k}: "
              f"train {train_idx.size} rows / {len(tr_cubes)} cubes, "
              f"test {test_idx.size} rows / {len(te_cubes)} cubes | "
              f"years train {tr_years} test {te_years}")
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# The modes
# ---------------------------------------------------------------------------

def cube_folds(manifest, k=5, min_clear: float = MIN_CLEAR_FRACTION,
               verbose: bool = True):
    """DEFAULT. Group k-fold with cube_id as the group.

    Handles spatial leakage at prototype scale: cubes are non-overlapping by
    construction (64 px separation), so the cube is the smallest honest
    spatial unit. ``k=LOCO`` gives leave-one-cube-out, the honest choice at
    the 20-cube prototype scale.
    """
    _validate_manifest(manifest)
    rows = filter_clear(manifest, min_clear, verbose=verbose)
    cubes = manifest["cube_id"].to_numpy()[rows]
    n_cubes = np.unique(cubes).size
    k = n_cubes if k == LOCO else int(k)
    fold_of = _group_kfold_assign(cubes, k)
    fold = np.array([fold_of[c] for c in cubes])
    for i in range(k):
        yield _emit("cube", i, k, manifest,
                    rows[fold != i], rows[fold == i], verbose)


def leave_one_cube_out(manifest, min_clear: float = MIN_CLEAR_FRACTION,
                       verbose: bool = True):
    """Leave-one-cube-out: cube mode with k = number of cubes."""
    yield from cube_folds(manifest, k=LOCO, min_clear=min_clear, verbose=verbose)


def crossed_folds(manifest, k=5, min_clear: float = MIN_CLEAR_FRACTION,
                  verbose: bool = True):
    """Cube AND year jointly held out: the resolution of the collision.

    A test fold contains only (cube, year) pairs whose cube is unseen in train
    AND whose year is unseen in train. Rows matching on one axis only (test
    cube in a train year, or train cube in a test year) belong to NEITHER side
    of that fold. This is P4's mode whenever the manifest spans more than one
    year, and it matches the climatology's leave-target-year-out protocol.

    A fold whose held-out cubes have no frames in its held-out years is
    skipped with a loud print (possible when cube windows cluster by year);
    at least one fold must survive or this raises.
    """
    _validate_manifest(manifest)
    rows = filter_clear(manifest, min_clear, verbose=verbose)
    sub = manifest.iloc[rows]
    years = _row_years(sub, strict=True, context="crossed mode")
    uniq_years = np.unique(years)
    if uniq_years.size < 2:
        raise SingleYearError(
            f"crossed (cube x year) CV needs at least 2 years, the manifest "
            f"has {uniq_years.tolist()}. On a single-year subset cube and year "
            "are perfectly confounded, so grouping by cube already holds the "
            "year out: use the default 'cube' mode instead. Do not fall back "
            "to a random split."
        )
    cubes = sub["cube_id"].to_numpy()
    k = _clamp_k(int(k), min(uniq_years.size, np.unique(cubes).size),
                 "year/cube", verbose)
    year_fold = _group_kfold_assign(years, k)
    cube_fold = _group_kfold_assign(cubes, k)
    yf = np.array([year_fold[y] for y in years])
    cf = np.array([cube_fold[c] for c in cubes])
    emitted = 0
    for i in range(k):
        test = rows[(cf == i) & (yf == i)]
        train = rows[(cf != i) & (yf != i)]
        if not test.size or not train.size:
            print(f"[cv] crossed fold {i + 1}/{k}: SKIPPED -- held-out cubes "
                  "have no frames in the held-out years (or nothing is left "
                  "to train on); this fold tests nothing")
            continue
        tr, te = _emit("crossed", i, k, manifest, train, test, verbose)
        # The crossed guarantee, asserted explicitly on every emitted fold.
        all_years = _timestamps(manifest).astype("datetime64[Y]").astype(int) + 1970
        assert not set(all_years[te]) & set(all_years[tr]), "crossed: year leaked"
        emitted += 1
        yield tr, te
    assert emitted, (
        "crossed mode emitted no folds: no (cube, year) pair survived the "
        "joint holdout. The cube windows cluster too tightly by year for "
        f"k={k}; try a smaller k or the 'cube' mode."
    )


def year_folds(manifest, k=5, min_clear: float = MIN_CLEAR_FRACTION,
               verbose: bool = True):
    """Year as the group. Raises on a single year; collides on seasonal cubes.

    On the extreme split (single year) this raises SingleYearError, mirroring
    ``data.climatology.ndvi_climatology``. On the seasonal split, where one
    cube spans years, the same-cube gate raises LeakageError and names
    ``crossed`` as the resolution -- year mode is only cleanly satisfiable
    when every cube sits inside one year.
    """
    _validate_manifest(manifest)
    rows = filter_clear(manifest, min_clear, verbose=verbose)
    sub = manifest.iloc[rows]
    years = _row_years(sub, strict=True, context="year mode")
    uniq = np.unique(years)
    if uniq.size < 2:
        raise SingleYearError(
            f"year-grouped CV needs at least 2 years, the manifest has "
            f"{uniq.tolist()} ({np.unique(sub['cube_id']).size} cubes). Every "
            "cube in tile 32UNU is from 2018 (the extreme split), so this is "
            "expected on the current subset. Use 'cube' mode (spatial "
            "grouping) here, or 'crossed' once the manifest spans years. Do "
            "not fall back to a random split: it would put the same season on "
            "both sides."
        )
    k = _clamp_k(int(k), uniq.size, "year", verbose)
    fold_of = _group_kfold_assign(years, k)
    fold = np.array([fold_of[y] for y in years])
    hint = ("A cube spanning a year boundary cannot be split by year (the "
            "seasonal-split collision); use 'crossed' mode, which holds cube "
            "and year out jointly.")
    for i in range(k):
        yield _emit("year", i, k, manifest,
                    rows[fold != i], rows[fold == i], verbose, leak_hint=hint)


def tile_folds(manifest, k=5, min_clear: float = MIN_CLEAR_FRACTION,
               verbose: bool = True):
    """Whole geographic tiles held out. Raises on a single-tile manifest."""
    _validate_manifest(manifest)
    rows = filter_clear(manifest, min_clear, verbose=verbose)
    tiles = manifest["tile"].to_numpy()[rows]
    uniq = np.unique(tiles)
    if uniq.size < 2:
        raise SingleTileError(
            f"tile-grouped CV needs at least 2 tiles, the manifest has "
            f"{uniq.tolist()}. The 20-cube prototype is tile 32UNU only, so "
            "this is expected: use 'spatial_block' as the prototype-scale "
            "substitute (whole within-tile clusters held out). True tile "
            "holdout is DEFERRED TO SCALE-UP. Do not fall back to a random "
            "split."
        )
    k = _clamp_k(int(k), uniq.size, "tile", verbose)
    fold_of = _group_kfold_assign(tiles, k)
    fold = np.array([fold_of[t] for t in tiles])
    for i in range(k):
        tr, te = _emit("tile", i, k, manifest,
                       rows[fold != i], rows[fold == i], verbose)
        all_tiles = manifest["tile"].to_numpy()
        assert not set(all_tiles[te]) & set(all_tiles[tr]), "tile leaked"
        yield tr, te


def _bbox_centroids(sub) -> tuple:
    """Per-cube (cube_id array, centroid array (n, 2)) from pixel_bbox."""
    per_cube = sub.drop_duplicates("cube_id")[["cube_id", "pixel_bbox"]]
    ids = per_cube["cube_id"].to_numpy()
    cents = np.array([[(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0]
                      for b in per_cube["pixel_bbox"]])
    assert cents.shape == (ids.size, 2)
    return ids, cents


def _complete_linkage(cents: np.ndarray, n_blocks: int) -> np.ndarray:
    """Deterministic complete-linkage agglomerative clustering to n_blocks.

    O(n^3) worst case, which is fine at prototype scale (tens of cubes); at
    scale-up spatial_block is superseded by true tile holdout anyway.
    """
    n = cents.shape[0]
    assert 1 <= n_blocks <= n
    D = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(-1))
    clusters = [[i] for i in range(n)]
    while len(clusters) > n_blocks:
        best, pair = np.inf, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = D[np.ix_(clusters[a], clusters[b])].max()   # complete linkage
                if d < best:
                    best, pair = d, (a, b)
        a, b = pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = np.empty(n, dtype=int)
    for ci, members in enumerate(clusters):
        labels[members] = ci
    return labels


def spatial_block_folds(manifest, k=5, n_blocks=None,
                        min_clear: float = MIN_CLEAR_FRACTION,
                        verbose: bool = True):
    """Prototype-scale SUBSTITUTE for tile mode; true tile holdout is deferred
    to scale-up.

    Single-tile manifest (the prototype): cubes are clustered by pixel_bbox
    centroid distance (complete linkage, deterministic) into ``n_blocks``
    (default k) within-tile blocks, and whole blocks are held out. This is
    weaker than tile holdout -- test cubes still share the tile's weather and
    phenology with train -- which is exactly why it is a substitute and not
    the real thing.

    Multi-tile manifest: the spatial block IS the tile. Clustering within
    tiles while whole tiles are available would be strictly weaker than
    holding the tiles out, so this mode collapses to tile holdout there (and
    no tile ever appears on both sides).
    """
    _validate_manifest(manifest)
    rows = filter_clear(manifest, min_clear, verbose=verbose)
    sub = manifest.iloc[rows]
    tiles = sub["tile"].to_numpy()
    n_tiles = np.unique(tiles).size
    if n_tiles > 1:
        if verbose:
            print(f"[cv] spatial_block: {n_tiles} tiles present -- the block "
                  "IS the tile (within-tile clustering would be weaker than "
                  "the tile holdout that is available)")
        block_of_cube = {c: t for c, t in
                         zip(sub["cube_id"].to_numpy(), tiles)}
    else:
        ids, cents = _bbox_centroids(sub)
        nb = int(n_blocks) if n_blocks is not None else int(k)
        assert nb <= ids.size, (
            f"n_blocks={nb} but only {ids.size} cubes exist in the tile"
        )
        labels = _complete_linkage(cents, nb)
        block_of_cube = {c: f"{tiles[0]}/block{l}" for c, l in zip(ids, labels)}
        if verbose:
            sizes = np.bincount(labels, minlength=nb)
            print(f"[cv] spatial_block: 1 tile, {ids.size} cubes clustered by "
                  f"pixel_bbox into {nb} blocks (complete linkage), "
                  f"sizes {sizes.tolist()} -- SUBSTITUTE for tile holdout, "
                  "which is deferred to scale-up")
    blocks = np.array([block_of_cube[c] for c in sub["cube_id"].to_numpy()])
    k = _clamp_k(int(k), np.unique(blocks).size, "spatial block", verbose)
    fold_of = _group_kfold_assign(blocks, k)
    fold = np.array([fold_of[b] for b in blocks])
    for i in range(k):
        tr, te = _emit("spatial_block", i, k, manifest,
                       rows[fold != i], rows[fold == i], verbose)
        if n_tiles > 1:
            all_tiles = manifest["tile"].to_numpy()
            assert not set(all_tiles[te]) & set(all_tiles[tr]), "tile leaked"
        yield tr, te


def temporal_folds(manifest, cutoff, min_clear: float = MIN_CLEAR_FRACTION,
                   verbose: bool = True):
    """Train strictly before, test strictly after a cutoff timestamp.

    NOT a default. A robustness variant for P3 only, because it starves
    training data when cube grouping already holds -- and here is where the
    starvation bites: the same-cube rule binds in this mode too, so every cube
    is assigned WHOLE to the side holding the majority of its frames (ties go
    to train), and its frames on the other side of the cutoff are DROPPED,
    loudly. On this subset no cutoff separates complete cubes (the 16 windows
    overlap), so the dropped-frame count is the real price of the mode and it
    is printed per side. Frames exactly at the cutoff belong to neither side.

    Yields exactly one (train_idx, test_idx) fold.
    """
    _validate_manifest(manifest)
    rows = filter_clear(manifest, min_clear, verbose=verbose)
    cutoff = np.datetime64(cutoff)
    ts = _timestamps(manifest)[rows]
    cubes = manifest["cube_id"].to_numpy()[rows]
    before, after = ts < cutoff, ts > cutoff

    uniq = np.unique(cubes)
    side = {}   # cube -> "train" | "test"
    for c in uniq:
        m = cubes == c
        n_b, n_a = int(before[m].sum()), int(after[m].sum())
        side[c] = "train" if n_b >= n_a else "test"
    on_train_cube = np.array([side[c] == "train" for c in cubes])

    train = rows[on_train_cube & before]
    test = rows[~on_train_cube & after]
    n_at = int((~before & ~after).sum())
    n_dropped = rows.size - train.size - test.size
    if verbose:
        n_tr_cubes = sum(1 for c in uniq if side[c] == "train")
        print(f"[cv] temporal cutoff {np.datetime64(cutoff, 'D')}: "
              f"{n_tr_cubes}/{uniq.size} cubes to train, "
              f"{uniq.size - n_tr_cubes} to test; DROPPED {n_dropped} frames "
              f"({n_at} at the cutoff, the rest on the wrong side of their "
              "cube's majority) -- this starvation is why temporal is a P3 "
              "robustness variant, not a default")
    assert train.size, (
        f"temporal cutoff {cutoff}: no training frames survive. Every cube's "
        "majority sits after the cutoff; move it later (retained frames span "
        f"{ts.min()} .. {ts.max()})."
    )
    assert test.size, (
        f"temporal cutoff {cutoff}: no test frames survive. Every cube's "
        "majority sits before the cutoff; move it earlier (retained frames "
        f"span {ts.min()} .. {ts.max()})."
    )
    assert (_timestamps(manifest)[train] < cutoff).all()
    assert (_timestamps(manifest)[test] > cutoff).all()
    yield _emit("temporal", 0, 1, manifest, train, test, verbose)


def folds(manifest, mode: str = "cube", k=5,
          min_clear: float = MIN_CLEAR_FRACTION, cutoff=None, n_blocks=None,
          verbose: bool = True):
    """Dispatch to one of MODES. See the module docstring for what runs where.

    Every generator yields (train_idx, test_idx): integer positions into the
    manifest's rows.
    """
    assert mode in MODES, f"mode {mode!r} not in {MODES}"
    if mode == "cube":
        return cube_folds(manifest, k=k, min_clear=min_clear, verbose=verbose)
    if mode == "crossed":
        return crossed_folds(manifest, k=k, min_clear=min_clear, verbose=verbose)
    if mode == "year":
        return year_folds(manifest, k=k, min_clear=min_clear, verbose=verbose)
    if mode == "tile":
        return tile_folds(manifest, k=k, min_clear=min_clear, verbose=verbose)
    if mode == "spatial_block":
        return spatial_block_folds(manifest, k=k, n_blocks=n_blocks,
                                   min_clear=min_clear, verbose=verbose)
    assert cutoff is not None, "temporal mode needs cutoff=<timestamp>"
    return temporal_folds(manifest, cutoff, min_clear=min_clear, verbose=verbose)


# ---------------------------------------------------------------------------
# The manifest <-> embeddings join
# ---------------------------------------------------------------------------

def join_embeddings(manifest, encoded, verbose: bool = True) -> dict:
    """Join one cube's manifest rows to its EncodedCube, contract asserted.

    The contract: (cube_id, original_axis_index) == (cube, kept_idx). Both
    sides also carry timestamps and clear_frac, so those are asserted equal
    rather than trusted. Returns a dict whose arrays are all aligned to the
    same T_kept order:

        manifest_idx      (T_kept,) int, POSITIONS into the manifest's rows
        embeddings        (T_kept, D) float32
        grid              (T_kept, 16, D_grid) float16
        grid_clear_frac   (T_kept, 16)
        clear_frac        (T_kept,)
        timestamps        (T_kept,)
        window_span_days  (T_kept,)  -- NEVER dropped: the MI encoder's
                          lookback varies over ~0-105 days, correlates with
                          cloud, and a probe must be able to condition on it.
    """
    _validate_manifest(manifest)
    cube = encoded.cube
    pos = np.flatnonzero((manifest["cube_id"] == cube).to_numpy())
    assert pos.size, f"manifest has no rows for cube {cube!r}"
    sub = manifest.iloc[pos]
    order = np.argsort(sub["original_axis_index"].to_numpy(), kind="stable")
    pos = pos[order]
    sub = manifest.iloc[pos]

    m_idx = sub["original_axis_index"].to_numpy().astype(int)
    k_idx = np.asarray(encoded.kept_idx).astype(int)
    assert m_idx.shape == k_idx.shape and (m_idx == k_idx).all(), (
        f"JOIN CONTRACT VIOLATED for {cube!r}: manifest original_axis_index "
        f"{m_idx.tolist()} != embedding kept_idx {k_idx.tolist()}. The two "
        "were built under different frame selections (mask or min_clear "
        "changed since encoding?); reset_phase('phase1_2') and re-encode."
    )
    m_ts = np.asarray(sub["timestamp"].to_numpy(), dtype="datetime64[ns]")
    assert (m_ts == np.asarray(encoded.timestamps)).all(), (
        f"{cube!r}: manifest timestamps != embedding timestamps at equal "
        "kept_idx -- the cube on disk changed since encoding"
    )
    np.testing.assert_allclose(
        sub["clear_frac"].to_numpy().astype(float), encoded.clear_frac,
        rtol=0, atol=1e-9,
        err_msg=f"{cube!r}: clear_frac disagrees between manifest and embedding",
    )
    wsd = encoded.window_span_days
    assert wsd is not None, (
        f"{cube!r} x {encoded.encoder}: window_span_days is missing from the "
        "embedding (pre-v3 cache?). It is a required covariate -- the MI "
        "lookback is weather-correlated -- and this join refuses to drop it "
        "silently. reset_phase('phase1_2') and re-encode."
    )
    T = pos.size
    out = {
        "manifest_idx": pos,
        "embeddings": encoded.embeddings,
        "grid": encoded.grid,
        "grid_clear_frac": encoded.grid_clear_frac,
        "clear_frac": encoded.clear_frac,
        "timestamps": np.asarray(encoded.timestamps),
        "window_span_days": np.asarray(wsd),
        "cube": cube,
        "encoder": encoded.encoder,
    }
    assert out["embeddings"].shape[0] == T
    assert out["window_span_days"].shape == (T,)
    if verbose:
        print(f"[cv] join {cube} x {encoded.encoder}: manifest_idx {pos.shape} "
              f"| embeddings {out['embeddings'].shape} | "
              f"grid {None if out['grid'] is None else out['grid'].shape} | "
              f"window_span_days {out['window_span_days'].shape} "
              f"(min {out['window_span_days'].min():.0f} "
              f"max {out['window_span_days'].max():.0f} days)")
    return out
