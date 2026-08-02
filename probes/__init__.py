"""Probes on frozen EO foundation-model representations.

No pretrained model is ever fine-tuned here. Encoders are frozen: `.eval()` and
`torch.no_grad()`, always. Splits come from `probes.cv` and nowhere else.

SCOPE UNDER THE SINGLE-YEAR SUBSET
----------------------------------
Every cube in tile 32UNU is from 2018, and each covers one ~150-day window with
about 29 acquisitions. What that does and does not cost:

* P2, 5-day deltas: UNAFFECTED. The 5-day step is the Sentinel-2 revisit and
  sits well inside a single cube.
* P3, horizons up to ~100 days: UNAFFECTED. A 150-day window holds a 100-day
  horizon with context to spare.
* The ceiling claim: NARROWED to "within-season". It cannot speak to
  interannual variability or multi-year seasonality on this subset. This is the
  EarthNet benchmark's own design, so the narrowed claim is the conventional
  one, but it must be stated rather than assumed away.

The PCA "latent clock" figure cannot come from one cube: ~29 frames over ~150
days is a fragment of an annual cycle, and a clock drawn from it would be a
line, not a loop. Pool embeddings ACROSS cubes spanning the tile's 16 distinct
time windows (starts from 2018-03-09 to 2018-07-07, so 2018-03 to 2018-12
coverage in aggregate) and colour by month or season. The spread across windows
is the only seasonal axis this subset offers.

Cross-year quantities raise rather than degenerate. See
`data.climatology.ndvi_climatology` and `probes.cv.year_groups`, both of which
raise `SingleYearError` here.
"""
