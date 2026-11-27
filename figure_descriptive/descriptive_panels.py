#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive 6-panel comparisons for Results 3-5.

For each of six biopsy-level morphometric summaries, compares values
across biopsy-level strata with a Kruskal-Wallis test, applies
Benjamini-Hochberg FDR control across the six panels of the figure, and
summarises the monotonic association with Spearman correlation
(Methods 11, Results 3-5). Draws the 2x3 box+strip panel figure.

Presets (--spec):
  glom    Figure 3  — glomerular summaries across pooled Tervaert groups
  tubule  Fig. S2   — tubular summaries across biopsy-level IFTA strata
  inflam  Fig. S3   — inflammatory summaries across the model-derived
                      inflammation grade analogue
Or fully custom: --features a,b,c... --strata_col X --strata_order u,v,w

Expected patient-level columns (one row per biopsy; aliases resolved):
  glom preset:  podocyte-class nuclear count (median), mesangial area
                fraction, total nuclear cell density, tuft-area
                fraction, eccentricity, circularity
  tubule:       median per-tubule minimum TBM thickness, median tubule
                area, median lumen area, median luminal circularity,
                within-biopsy SD of TBM thickness, SD of TBM area
  inflam:       tubule-to-inflammatory-cell distance, peritubular
                inflammatory density, inflammatory share, hotspot-
                tubule fraction, atrophy-directed share, periglomerular
                inflammatory density
Strata columns come from assign_strata.py
(tervaert_group / ifta_stratum / inflam_grade).

Usage:
  python descriptive_panels.py --patient patient_with_strata.csv \
      --spec glom --out_prefix fig3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

MM = 1 / 25.4
NK, NL = "#2D2D2D", "#B0B0B0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 6.6,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

PRESETS = {
    "glom": {
        "strata_col": "tervaert_group",
        "strata_order": ["I-IIB", "III", "IV"],
        "features": [
            ("podocyte-class nuclear count", ["podocyte_nuc_med", "podocyte_count_med", "pod_nuc_med"]),
            ("mesangial area fraction", ["mesangial_area_frac", "mes_frac", "mesangial_frac"]),
            ("nuclear cell density", ["nuc_cell_density", "cell_density", "nuc_density"]),
            ("tuft-area fraction", ["tuft_area_frac", "tuft_frac"]),
            ("eccentricity", ["eccentricity", "glom_eccentricity"]),
            ("circularity", ["circularity", "glom_circularity"]),
        ],
    },
    "tubule": {
        "strata_col": "ifta_stratum",
        "strata_order": [0, 1, 2, 3],
        "strata_labels": ["<=10%", "10-25%", "25-50%", ">50%"],
        "features": [
            ("min TBM thickness (median)", ["tbm_thick_min_med", "tbm_min_med"]),
            ("tubule area (median)", ["tub_area_med", "tubule_area_med"]),
            ("lumen area (median)", ["lumen_area_med", "tub_lumen_med"]),
            ("luminal circularity (median)", ["lumen_circ_med", "lum_circ_med"]),
            ("SD TBM thickness", ["tbm_thick_sd", "sd_tbm_thick"]),
            ("SD TBM area", ["tbm_area_sd", "sd_tbm_area"]),
        ],
    },
    "inflam": {
        "strata_col": "inflam_grade",
        "strata_order": [0, 1, 2],
        "features": [
            ("tubule-inflam distance", ["tub_inflam_dist_med", "nearest_um_med", "dist_med"]),
            ("peritubular density", ["peritub_dens", "dens_50_med", "inflam_dens_med"]),
            ("inflammatory share", ["inflam_share", "inflam_frac_50_med"]),
            ("hotspot fraction", ["hotspot_frac", "frac_hotspot", "hotspot_med"]),
            ("atrophy-directed share", ["atrophy_directed_share", "atrophy_inflam_f_med"]),
            ("periglomerular density", ["periglom_dens", "peri_glom_dens_med"]),
        ],
    },
}


def pick_col(df, cands, label):
    for c in cands:
        if c in df.columns:
            return c
    print(f"  [skip] missing column for '{label}' (tried {cands})", file=sys.stderr)
    return None


def kw_and_spearman(df, col, scol, order):
    groups = [df.loc[df[scol] == o, col].dropna().to_numpy(float) for o in order]
    groups = [g for g in groups if len(g)]
    if len(groups) < 2:
        return np.nan, np.nan, np.nan
    kw_p = float(stats.kruskal(*groups).pvalue)
    d = df[[scol, col]].dropna()
    rho, sp_p = stats.spearmanr(d[scol].cat.codes.to_numpy(float), d[col].to_numpy(float))
    return kw_p, float(rho), float(sp_p)


def main():
    ap = argparse.ArgumentParser(description="Results 3-5 descriptive 6-panel figures")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--spec", choices=list(PRESETS) + ["custom"], default="glom")
    ap.add_argument("--features", default="", help="custom: comma-separated column names")
    ap.add_argument("--strata_col", default="")
    ap.add_argument("--strata_order", default="", help="custom: comma-separated values")
    ap.add_argument("--out_prefix", default="panels")
    args = ap.parse_args()

    df = pd.read_csv(args.patient)
    if args.spec == "custom":
        feats = [(c, [c]) for c in args.features.split(",") if c]
        spec = {
            "strata_col": args.strata_col,
            "strata_order": args.strata_order.split(","),
            "features": feats,
        }
    else:
        spec = PRESETS[args.spec]

    scol = spec["strata_col"]
    if scol not in df.columns:
        sys.exit(f"strata column '{scol}' not in patient table (run assign_strata.py first)")
    order = spec["strata_order"]
    labels = spec.get("strata_labels", [str(o) for o in order])
    df = df[df[scol].isin(order)].copy()
    df[scol] = pd.Categorical(df[scol].astype(type(order[0])), categories=order, ordered=True)

    panels = []
    for label, cands in spec["features"]:
        col = pick_col(df, cands, label)
        if col is not None:
            panels.append((label, col))
    if not panels:
        sys.exit("no feature columns resolved — nothing to plot")
    stats_rows = []
    for label, col in panels:
        kw_p, rho, sp_p = kw_and_spearman(df, col, scol, order)
        stats_rows.append({"feature": label, "col": col, "kw_p": kw_p, "rho": rho, "spearman_p": sp_p})
    st = pd.DataFrame(stats_rows)
    st["kw_q"] = multipletests(st["kw_p"].fillna(1.0), method="fdr_bh")[1]
    print(st[["feature", "kw_p", "kw_q", "rho", "spearman_p"]].to_string(index=False))
    st.to_csv(f"{args.out_prefix}_stats.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(168 * MM, 110 * MM))
    letters = "abcdef"
    rng = np.random.default_rng(0)
    for k, (label, col) in enumerate(panels):
        ax = axes[k // 3, k % 3]
        data = [df.loc[df[scol] == o, col].dropna().to_numpy(float) for o in order]
        pos = np.arange(len(order))
        bp = ax.boxplot(data, positions=pos, widths=0.55, showfliers=False,
                        medianprops=dict(color=NK, lw=1.0),
                        boxprops=dict(color=NK, lw=0.7),
                        whiskerprops=dict(color=NK, lw=0.7),
                        capprops=dict(color=NK, lw=0.7))
        for i, vals in enumerate(data):
            ax.scatter(pos[i] + rng.uniform(-0.18, 0.18, len(vals)), vals,
                       s=4, color="#0072B5", alpha=0.5, linewidths=0, zorder=3)
        r = st.iloc[k]
        star = "*" if r.kw_q < 0.05 else ""
        ax.set_title(f"{label}\nKW P={r.kw_p:.3g} q={r.kw_q:.3g}{star}  rho={r.rho:.2f}",
                     fontsize=6.0, color=NK)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=5.8)
        ax.text(-0.28, 1.05, letters[k], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    for k in range(len(panels), 6):
        axes[k // 3, k % 3].axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}.{ext}", bbox_inches="tight", facecolor="white")
    print("->", f"{args.out_prefix}.png/.pdf")


if __name__ == "__main__":
    main()
