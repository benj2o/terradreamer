"""Fetch <=20 pre-processed GreenEarthNet minicubes for one MGRS tile.

    python -m data.download_greenearthnet --out data/raw --n 20 --tile 32UNU

Why this instead of live Sentinel-2 extraction:

* Speed. A tile-filtered pull of 20 cubes is ~70 MB and about 15 seconds.
  Running earthnet-minicuber over the same 20 cubes measured 14.7 hours,
  because the cloud-mask U-Net runs on CPU for 36 monthly queries per cube.
* Mask provenance. These cubes ship GreenEarthNet's published masks. Numbers
  computed on them are comparable with the benchmark; numbers computed on
  masks we derived ourselves are not.

Note on `earthnet.download(..., limit=N)`: `limit` slices a lexicographic
listing of every file in the split, so it returns N cubes from whichever tile
sorts first, not from the tile you want. Hence this module.

Tile choice: 32UPU, which contains Munich itself, is NOT in the dataset. The
closest true Alpine-foreland tile is 32UNU (9.00-10.49 E, 47.76-48.75 N,
Allgaeu and Upper Swabia), same latitude band as Munich and about 135 km west.
32TPT is nearer in km but is high Alps, a different biome.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ENDPOINT = "https://s3.bgc-jena.mpg.de:9000"
REGION = "thuringia"
BUCKET = "earthnet/earthnet2021x"

DEFAULT_TILE = "32UNU"
DEFAULT_SPLIT = "train"

# Alpine-foreland / Bavaria MGRS tiles that exist in the dataset, nearest first.
# 32UPU (Munich) is absent upstream.
BAVARIA_TILES = ("32UNU", "32TPT", "33TUN")

__all__ = ["list_cubes", "parse_footprint", "select_non_overlapping", "s3fs_client"]


def s3fs_client():
    import s3fs

    return s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": ENDPOINT, "region_name": REGION},
    )


def list_cubes(tile: str = DEFAULT_TILE, split: str = DEFAULT_SPLIT, s3=None) -> list:
    """Every cube key for one tile. Anonymous, no credentials needed."""
    s3 = s3 or s3fs_client()
    keys = sorted(s3.ls(f"{BUCKET}/{split}/{tile}"))
    assert keys, (
        f"no cubes under {BUCKET}/{split}/{tile}. Available Bavaria-area tiles: "
        f"{BAVARIA_TILES}"
    )
    return keys


def parse_footprint(name: str) -> tuple:
    """Pixel box (r0, r1, c0, c1) from a cube id.

    Ids look like
    32UNU_2018-03-09_2018-08-05_1081_1209_3641_3769_16_96_56_136
    tile  start       end        r0   r1   c0   c1  <eobs indices>
    """
    parts = os.path.basename(name)[:-3].split("_") if name.endswith(".nc") \
        else os.path.basename(name).split("_")
    r0, r1, c0, c1 = (int(p) for p in parts[3:7])
    assert r1 > r0 and c1 > c0, f"bad footprint in {name}: {(r0, r1, c0, c1)}"
    return r0, r1, c0, c1


def _time_window(key: str) -> str:
    return "_".join(os.path.basename(key)[:-3].split("_")[1:3])


def select_non_overlapping(keys, n: int, spread_windows: bool = True,
                           min_gap_px: int = 64) -> list:
    """Take `n` cubes whose pixel boxes do not intersect.

    Two cubes sharing pixels would put identical data on both sides of whatever
    split probes/cv.py draws later. Cheaper to prevent here than to detect after
    the fact, though data.loader.assert_no_overlap still checks on disk.

    With `spread_windows`, cubes are drawn round-robin across the tile's
    distinct time windows. All cubes in a tile are from the same year, so the
    window is the only seasonal variation on offer: taking them in listing order
    would hand back 20 cubes that all start in March.
    """
    keys = list(keys)
    assert keys, "no cubes to select from"

    if spread_windows:
        by_window: dict = {}
        for k in keys:
            by_window.setdefault(_time_window(k), []).append(k)
        ordered, buckets = [], list(by_window.values())
        for row in range(max(len(b) for b in buckets)):
            for b in buckets:
                if row < len(b):
                    ordered.append(b[row])
        keys = ordered

    g = min_gap_px
    chosen, boxes = [], []
    for k in keys:
        r0, r1, c0, c1 = parse_footprint(k)
        # Reject not just overlap but adjacency: cubes touching edge to edge
        # share no pixel yet are the same field, so a spatial split would still
        # be leaking. `min_gap_px` reproduces the 64 px (half a cube) separation
        # the live-extraction grid used.
        if any(r0 - g < R1 and R0 - g < r1 and c0 - g < C1 and C0 - g < c1
               for R0, R1, C0, C1 in boxes):
            continue
        boxes.append((r0, r1, c0, c1))
        chosen.append(k)
        if len(chosen) >= n:
            break
    assert chosen, "no non-overlapping cubes found"
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--n", type=int, default=20, help="number of cubes (<=20)")
    ap.add_argument("--tile", default=DEFAULT_TILE, help=f"MGRS tile, e.g. {BAVARIA_TILES}")
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    assert 1 <= args.n <= 20, "Munich-first: keep this <= 20 cubes"

    os.makedirs(args.out, exist_ok=True)
    s3 = s3fs_client()

    print(f"[list] {BUCKET}/{args.split}/{args.tile}")
    keys = list_cubes(args.tile, args.split, s3)
    chosen = select_non_overlapping(keys, args.n)
    print(f"[list] {len(keys)} cubes in tile, selected {len(chosen)} non-overlapping")

    windows = sorted({"_".join(os.path.basename(k)[:-3].split("_")[1:3]) for k in chosen})
    print(f"[list] time windows covered: {len(windows)}")
    for w in windows:
        print(f"         {w}")

    ok, failed = [], []
    t_batch = time.time()
    for i, key in enumerate(chosen):
        name = os.path.basename(key)
        path = os.path.join(args.out, name)
        if os.path.exists(path) and not args.overwrite:
            print(f"[{i + 1:02d}/{len(chosen)}] exists, skipping {name}")
            ok.append(path)
            continue
        tmp = path + ".partial"
        t0 = time.time()
        try:
            s3.download(key, tmp)
            os.replace(tmp, path)
            print(f"[{i + 1:02d}/{len(chosen)}] {name} "
                  f"({os.path.getsize(path) / 1e6:.1f} MB, {time.time() - t0:.1f}s)",
                  flush=True)
            ok.append(path)
        except Exception as e:
            print(f"[{i + 1:02d}/{len(chosen)}] FAILED {name}: {type(e).__name__}: {e}")
            failed.append(name)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    total_mb = sum(os.path.getsize(p) for p in ok) / 1e6 if ok else 0.0
    print(f"\n[done] {len(ok)}/{len(chosen)} cubes in {args.out} "
          f"({total_mb:.0f} MB, {time.time() - t_batch:.0f}s)")

    if not ok:
        print("[done] NO CUBES WERE DOWNLOADED.")
        sys.exit(1)
    if failed:
        print(f"[done] {len(failed)} failed: {failed}")
        print("[done] re-run the SAME command, finished cubes are skipped.")
        sys.exit(1)

    # Prove the first cube opens and carries what the loader needs.
    import xarray as xr

    with xr.open_dataset(ok[0]) as ds:
        need = ["s2_B02", "s2_B03", "s2_B04", "s2_B8A", "s2_mask"]
        missing = [v for v in need if v not in ds.variables]
        assert not missing, f"{ok[0]} is missing {missing}"
        print(f"[check] {os.path.basename(ok[0])} dims {dict(ds.sizes)}")
        print(f"[check] all of {need} present")


if __name__ == "__main__":
    main()
