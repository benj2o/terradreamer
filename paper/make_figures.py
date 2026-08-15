#!/usr/bin/env python3
"""Rebuild paper/figures from published Tier-1 / screened CSVs.

Run from repo root:
  MPLCONFIGDIR=.pycache/mpl python paper/make_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#222222",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
    }
)

HORIZONS = [5, 25, 50, 100]
COLORS = {
    "raw_features + weather": "#1b4f72",
    "raw_rgb_only + weather": "#2874a6",
    "[NDVI(t), weather]": "#148f77",
    "best frozen network": "#b9770e",
    "persistence": "#7b241c",
    "proxy climatology": "#7f8c8d",
    "observation control": "#95a5a6",
}
LABELS = {
    "imagenet_vit_b16": "ImageNet ViT",
    "imagenet_vit_b16_cir": "ImageNet ViT (CIR)",
    "dinov2_vitb14": "DINOv2",
    "dinov2_vitb14_cir": "DINOv2 (CIR)",
    "satlas_s2_swinb_rgb": "Satlas SI",
    "satlas_s2_swinb_rgb_cir": "Satlas SI (CIR)",
    "satlas_s2_swinb_mi_rgb": "Satlas MI",
    "satlas_s2_swinb_mi_rgb_cir": "Satlas MI (CIR)",
}


def load_p3() -> pd.DataFrame:
    path = ROOT / "data/scaled_32UNU/p3_tier1_results.csv"
    if not path.exists():
        sys.exit(f"missing {path} (gitignored locally; needed to rebuild)")
    p3 = pd.read_csv(path)
    m = (
        (p3.aggregation == "cube_mean")
        & (p3.fold_mode == "cube")
        & (p3.metric == "r2")
        & (p3.plausibility_screen == True)
        & (p3.target_level == "frame")
    )
    return p3[m].copy()


def pick_row(df: pd.DataFrame, **kwargs) -> pd.Series:
    q = df.copy()
    for k, v in kwargs.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise ValueError(f"expected 1 row for {kwargs}, got {len(q)}")
    return q.iloc[0]


def save(fig: plt.Figure, stem: str) -> None:
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", pdf)


def fig_headline_r2(p3: pd.DataFrame) -> None:
    rows = []
    for d in HORIZONS:
        dd = p3[p3.delta_days == d]
        specs = [
            (
                "raw_features + weather",
                dict(
                    model_kind="raw_features_weather",
                    feature_base="ndvi_weather",
                    alpha_rule="nested_cv",
                    estimator="linear",
                ),
            ),
            (
                "raw_rgb_only + weather",
                dict(
                    model_kind="raw_rgb_only_weather",
                    feature_base="ndvi_weather",
                    alpha_rule="nested_cv",
                    estimator="linear",
                ),
            ),
            (
                "[NDVI(t), weather]",
                dict(
                    model_kind="weather_only",
                    feature_base="ndvi_weather",
                    alpha_rule="nested_cv",
                    estimator="linear",
                ),
            ),
            ("persistence", dict(model_kind="persistence")),
            ("proxy climatology", dict(model_kind="climatology_proxy")),
            (
                "observation control",
                dict(
                    model_kind="observation",
                    estimator="linear",
                    alpha_rule="nested_cv",
                ),
            ),
        ]
        for label, kw in specs:
            r = pick_row(dd, **kw)
            rows.append((d, label, r.r2_pooled, r.r2_pooled_ci_lo, r.r2_pooled_ci_hi))
        fc = dd[
            (dd.model_kind == "forecast")
            & (dd.feature_base == "ndvi_weather")
            & (dd.alpha_rule == "nested_cv")
            & (dd.estimator == "linear")
        ]
        best = fc.loc[fc.r2_pooled.idxmax()]
        rows.append(
            (
                d,
                "best frozen network",
                best.r2_pooled,
                best.r2_pooled_ci_lo,
                best.r2_pooled_ci_hi,
            )
        )

    tab = pd.DataFrame(rows, columns=["delta", "label", "r2", "lo", "hi"])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for label in COLORS:
        s = tab[tab.label == label].sort_values("delta")
        ax.plot(
            s.delta,
            s.r2,
            marker="o",
            lw=1.8,
            ms=5,
            color=COLORS[label],
            label=label,
        )
        ax.fill_between(
            s.delta, s.lo, s.hi, color=COLORS[label], alpha=0.12, linewidth=0
        )
    ax.axhline(0, color="#444", lw=0.8, ls=":")
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel(r"Pooled out-of-fold $R^2$")
    ax.set_title("Tier-1 P3 headline skill (cube folds, plausibility screen on)")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(-0.25, 1.0)
    fig.tight_layout()
    save(fig, "fig1_headline_r2")


def fig_paired_bandmatched(p3: pd.DataFrame) -> None:
    fc = p3[
        (p3.model_kind == "forecast")
        & (p3.feature_base == "ndvi_weather")
        & (p3.alpha_rule == "nested_cv")
        & (p3.estimator == "linear")
    ].copy()
    encoders = [
        "imagenet_vit_b16",
        "dinov2_vitb14",
        "satlas_s2_swinb_rgb",
        "satlas_s2_swinb_mi_rgb",
        "imagenet_vit_b16_cir",
        "dinov2_vitb14_cir",
        "satlas_s2_swinb_rgb_cir",
        "satlas_s2_swinb_mi_rgb_cir",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 3.6), sharey=True)
    xpos = np.arange(len(encoders))
    for ax, d in zip(axes, HORIZONS):
        sub = fc[fc.delta_days == d].set_index("encoder")
        diffs, los, his, sep = [], [], [], []
        for e in encoders:
            r = sub.loc[e]
            diffs.append(float(r.paired_diff_vs_band_matched))
            los.append(float(r.paired_ci_lo_vs_band_matched))
            his.append(float(r.paired_ci_hi_vs_band_matched))
            sep.append(bool(r.separable_vs_band_matched))
        diffs_a = np.asarray(diffs)
        yerr = np.vstack([diffs_a - np.asarray(los), np.asarray(his) - diffs_a])
        colors = [
            "#a93226"
            if s and dff < 0
            else ("#1e8449" if s and dff > 0 else "#5d6d7e")
            for s, dff in zip(sep, diffs_a)
        ]
        ax.barh(
            xpos,
            diffs_a,
            xerr=yerr,
            color=colors,
            alpha=0.85,
            height=0.7,
            error_kw=dict(ecolor="#333", lw=0.8, capsize=2),
        )
        ax.axvline(0, color="#222", lw=0.9)
        ax.set_title(f"{d} d")
        ax.set_yticks(xpos)
        ax.invert_yaxis()
    axes[0].set_yticklabels([LABELS[e] for e in encoders])
    fig.supxlabel(
        r"Paired per-fold $\Delta R^2$ vs band-matched raw\_rgb\_only", y=0.02
    )
    fig.suptitle(
        "No frozen encoder view separably beats the band-matched baseline", y=1.02
    )
    axes[-1].legend(
        handles=[
            Patch(color="#a93226", label="separably below"),
            Patch(color="#1e8449", label="separably above"),
            Patch(color="#5d6d7e", label="not separable"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout()
    save(fig, "fig2_paired_vs_bandmatched")


def fig_sign_vs_magnitude() -> None:
    path = ROOT / "data/scaled_32UNU/p2_screened_results.csv"
    if not path.exists():
        sys.exit(f"missing {path}")
    p2 = pd.read_csv(path)
    base = p2[
        (p2.part == "B_delta")
        & (p2.fold_mode == "cube")
        & (p2.readout == "linear")
        & (p2.feature_level == "pooled")
        & (p2.metric == "spearman")
    ]
    order = [
        ("dinov2_vitb14", "embedding", "DINOv2"),
        ("satlas_s2_swinb_rgb", "embedding", "Satlas SI"),
        ("imagenet_vit_b16", "embedding", "ImageNet"),
        ("satlas_s2_swinb_mi_rgb", "embedding", "Satlas MI"),
        ("raw_features", "raw_rgb_only", "raw_rgb_only"),
        ("none", "gap_days", "gap-length control"),
    ]
    rows = []
    for enc, fs, label in order:
        for kind, target in [
            ("sign", "cube_mean_sign"),
            ("magnitude", "cube_mean_magnitude"),
        ]:
            r = base[
                (base.encoder == enc)
                & (base.feature_set == fs)
                & (base.target == target)
            ]
            assert len(r) == 1, (enc, fs, target, len(r))
            r = r.iloc[0]
            rows.append((label, kind, r.spearman_mean, r.spearman_ci_lo, r.spearman_ci_hi))
    tab = pd.DataFrame(rows, columns=["label", "kind", "rho", "lo", "hi"])
    labels = [o[2] for o in order]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for kind, color in [("sign", "#1a5276"), ("magnitude", "#b9770e")]:
        s = tab[tab.kind == kind].set_index("label").loc[labels]
        offset = -width / 2 if kind == "sign" else width / 2
        yerr = np.vstack([s.rho - s.lo, s.hi - s.rho])
        ax.bar(
            x + offset,
            s.rho,
            width=width,
            color=color,
            alpha=0.9,
            label=kind,
            yerr=yerr,
            error_kw=dict(ecolor="#333", lw=0.8, capsize=2),
        )
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ (pooled features)")
    ax.set_title("P2 screened: direction recoverable, rate is not")
    ax.legend(frameon=False)
    ax.set_ylim(-0.15, 0.95)
    fig.tight_layout()
    save(fig, "fig3_sign_vs_magnitude")


def fig_extreme_tails(p3: pd.DataFrame) -> None:
    specs = [
        (
            "raw_rgb_only",
            dict(
                model_kind="raw_rgb_only_weather",
                feature_base="ndvi_weather",
                alpha_rule="nested_cv",
                estimator="linear",
            ),
        ),
        (
            "DINOv2",
            dict(
                model_kind="forecast",
                encoder="dinov2_vitb14",
                feature_base="ndvi_weather",
                alpha_rule="nested_cv",
                estimator="linear",
            ),
        ),
        (
            "ImageNet CIR",
            dict(
                model_kind="forecast",
                encoder="imagenet_vit_b16_cir",
                feature_base="ndvi_weather",
                alpha_rule="nested_cv",
                estimator="linear",
            ),
        ),
        (
            "Satlas SI CIR",
            dict(
                model_kind="forecast",
                encoder="satlas_s2_swinb_rgb_cir",
                feature_base="ndvi_weather",
                alpha_rule="nested_cv",
                estimator="linear",
            ),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, tail, title in zip(
        axes,
        [
            "skill_vs_persistence_extreme_low",
            "skill_vs_persistence_extreme_high",
        ],
        ["extreme_low", "extreme_high"],
    ):
        for label, kw in specs:
            ys = [
                getattr(pick_row(p3[p3.delta_days == d], **kw), tail) for d in HORIZONS
            ]
            ax.plot(HORIZONS, ys, marker="o", lw=1.7, ms=5, label=label)
        ax.axhline(0, color="#444", lw=0.9, ls=":")
        ax.set_xticks(HORIZONS)
        ax.set_title(title)
        ax.set_xlabel("Horizon (days)")
    axes[0].set_ylabel("Skill vs persistence")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Extreme-tail skill vs persistence (cube-mean, Tier-1)", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_extreme_tails")


def copy_latent_clock() -> None:
    src = ROOT / "data/phase1_4/figures/figure1_latent_clock.png"
    if not src.exists():
        print("skip latent clock (missing)", src)
        return
    dst = OUT / "fig0_latent_clock.png"
    shutil.copy2(src, dst)
    print("copied", dst)


def main() -> None:
    p3 = load_p3()
    copy_latent_clock()
    fig_headline_r2(p3)
    fig_paired_bandmatched(p3)
    fig_sign_vs_magnitude()
    fig_extreme_tails(p3)


if __name__ == "__main__":
    main()
