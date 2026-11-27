#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 8 reproduction — all 15 pre-specified combinations of the locked
per-structure pseudotime percentiles and the cortical IFTA proportion,
for the 1- and 2-year composite kidney endpoints.

Input: patient-level tables produced by the external_features pipeline
(scripts 01–05). One row per biopsy. Required columns (aliases resolved):

    sample            de-identified biopsy code
    DCr               1-year composite endpoint (0/1)
    2Y_DCr            2-year composite endpoint (0/1)
    IFTA_model        cortical IFTA proportion = 100*KI1_SC2/(KI1_SC3+KI1_SC4)
    tub_p87/tub_p80   tubular pseudotime percentiles (locked 1y/2y)
    glo_p90/glo_p93   glomerular pseudotime percentiles (locked 1y/2y)
    psi_p87/psi_p95   inflammatory pseudotime percentiles (locked 1y/2y)
                      (alias inflam_psi_p87 / inflam_psi_p95 accepted)

Per model and horizon this reports
    apparent AUC       resubstitution on the discovery set (+ Hanley-McNeil CI)
    LOO-CV AUC         leave-one-out cross-validation on the discovery set
    external AUC       locked model applied to the external cohort without
                       refitting (+ Hanley-McNeil CI and one-sided P vs 0.5)

Logistic models are StandardScaler -> LogisticRegression(penalty=None,
max_iter=2000), fitted once on the discovery cohort; standardisation
parameters and coefficients are then held fixed (Methods sections 7-9).
Confidence intervals use the Hanley-McNeil approximation, limits
restricted to [0, 1]. The "P vs IFTA(model)" column follows the figure
footnote: univariable -> Wald; X+IFTA -> Wald of X|IFTA; pairs without
IFTA -> joint LRT vs null; models containing IFTA with >=3 predictors ->
LRT vs IFTA alone.

Usage:
    python 369_inflam_forest_357style_p87_p95.py \
        --discovery discovery_patient.csv [--external external_patient.csv]

Fallbacks: PSMP_DISCOVERY_CSV / PSMP_EXTERNAL_CSV environment variables,
or ./discovery_patient.csv / ./external_patient.csv inside this pack.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PACK = Path(__file__).resolve().parents[1]
OUT_RES = PACK / "04_results"
FIG = PACK / "05_figure"
for d in (OUT_RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MM = 1 / 25.4
IFTA_LAB = "IFTA(model)"
NB, NR, NK, NL = "#0072B5", "#BC3C29", "#2D2D2D", "#B0B0B0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.6,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

ALIASES = {
    "tub_p87": ["tub_p87"],
    "tub_p80": ["tub_p80"],
    "glo_p90": ["glo_p90"],
    "glo_p93": ["glo_p93"],
    "psi_p87": ["psi_p87", "inflam_psi_p87"],
    "psi_p95": ["psi_p95", "inflam_psi_p95"],
    "IFTA": ["IFTA_model", "IFTA"],
    "DCr": ["DCr", "1y_DCr"],
    "2Y_DCr": ["2Y_DCr", "2y_DCr"],
}

# locked upper-tail percentile per compartment and horizon (Results 7)
COLMAP = {
    "1y": {"T": "tub_p87", "I": "psi_p87", "G": "glo_p90", "F": "IFTA"},
    "2y": {"T": "tub_p80", "I": "psi_p95", "G": "glo_p93", "F": "IFTA"},
}
LBL = {
    "1y": {"T": "T-p87", "I": "I-p87", "G": "G-p90", "F": IFTA_LAB},
    "2y": {"T": "T-p80", "I": "I-p95", "G": "G-p93", "F": IFTA_LAB},
}

# all 15 non-empty combinations, in display order (singles first, per Results 7)
MODELS = (
    [("T",), ("F",), ("G",), ("I",)]
    + [("T", "I"), ("F", "I"), ("G", "I"), ("T", "F"), ("T", "G"), ("F", "G")]
    + [("T", "F", "I"), ("T", "F", "G"), ("T", "G", "I"), ("F", "G", "I")]
    + [("T", "F", "G", "I")]
)


def resolve(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for canon, cands in ALIASES.items():
        for c in cands:
            if c in df.columns:
                ren[c] = canon
                break
    return df.rename(columns=ren)


def load_patient(path: Path, horizon_cols: list[str], label: str) -> pd.DataFrame:
    df = resolve(pd.read_csv(path))
    need = sorted(set(horizon_cols + ["DCr", "2Y_DCr"]))
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"[{label}] {path}: missing required columns {missing}")
    dat = df.dropna(subset=need).copy()
    for c in need:
        dat[c] = pd.to_numeric(dat[c], errors="coerce")
    return dat.dropna(subset=need)


def logit_pipe() -> Pipeline:
    # Methods 7: scikit-learn pipeline of StandardScaler followed by
    # unpenalised logistic regression (penalty=None, max_iter=2000)
    return Pipeline(
        [("s", StandardScaler()), ("m", LogisticRegression(penalty=None, max_iter=2000))]
    )


def loo_oof(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, int)
    oof = np.zeros(len(y), float)
    for tr, te in LeaveOneOut().split(X):
        pipe = logit_pipe()
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def hanley_mcneil_se(auc, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUC (Methods 7.2)."""
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def auc_ci_hm(y, score):
    """AUC with Hanley-McNeil 95% CI, limits restricted to [0, 1]."""
    auc = float(roc_auc_score(y, score))
    se = hanley_mcneil_se(auc, int(np.sum(y == 1)), int(np.sum(y == 0)))
    lo = max(0.0, auc - 1.96 * se)
    hi = min(1.0, auc + 1.96 * se)
    return auc, lo, hi, se


def fit_logit(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, int)
    Xs = sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")
    fit = sm.Logit(y, Xs).fit(disp=False, maxiter=200)
    return fit, np.exp(fit.params[1:]), np.exp(fit.conf_int()[1:]), fit.pvalues[1:]


def nested_lrt(X_small, X_big, y):
    y = np.asarray(y, int)

    def _ll(X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.size == 0 or X.shape[1] == 0:
            return float(sm.Logit(y, np.ones((len(y), 1))).fit(disp=False, maxiter=200).llf), 0
        Xs = sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")
        return float(sm.Logit(y, Xs).fit(disp=False, maxiter=200).llf), X.shape[1]

    ll0, d0 = _ll(X_small)
    ll1, d1 = _ll(X_big)
    return float(1 - stats.chi2.cdf(2 * (ll1 - ll0), df=max(1, d1 - d0)))


def p_vs_ifta(combo, X, y, ifta):
    """P-value scheme of the figure footnote (see header docstring)."""
    if combo == ("F",):
        return np.nan, "reference", "-"
    _, ors, cis, ps = fit_logit(X, y)
    detail = "; ".join(
        f"{c}:OR={ors[j]:.2f} P={ps[j]:.3f}" for j, c in enumerate(X.columns)
    )
    if len(combo) == 1:
        return float(ps[0]), detail, "Wald(uni)"
    if "F" in combo and len(combo) == 2:
        j = [i for i, k in enumerate(combo) if k != "F"][0]
        return float(ps[j]), detail, "Wald|IFTA(model)"
    if "F" in combo:
        return nested_lrt(ifta, X.to_numpy(float), y), detail, "LRT vs IFTA(model)"
    return nested_lrt(np.ones((len(y), 0)), X.to_numpy(float), y), detail, "LRT vs null"


def eval_models(dat, dat_ext, ycol, cmap, horizon):
    rows = []
    y = dat[ycol].astype(int).to_numpy()
    ye = dat_ext[ycol].astype(int).to_numpy() if dat_ext is not None else None
    ifta = dat[["IFTA"]].to_numpy(float)
    for combo in MODELS:
        cols = [cmap[k] for k in combo]
        name = " + ".join(LBL[horizon][k] for k in combo)
        X = dat[cols].to_numpy(float)
        # apparent (resubstitution) AUC on the discovery set
        pipe = logit_pipe().fit(X, y)
        app, alo, ahi, _ = auc_ci_hm(y, pipe.predict_proba(X)[:, 1])
        # leave-one-out cross-validated AUC
        loo = float(roc_auc_score(y, loo_oof(X, y)))
        # locked external AUC (+ H-M CI, one-sided P vs 0.5)
        ext_auc = elo = ehi = ext_p = np.nan
        if dat_ext is not None:
            Xe = dat_ext[cols].to_numpy(float)
            s_ext = pipe.predict_proba(Xe)[:, 1]
            ext_auc, elo, ehi, se_e = auc_ci_hm(ye, s_ext)
            if se_e > 0:
                ext_p = float(stats.norm.sf((ext_auc - 0.5) / se_e))
        p, detail, method = p_vs_ifta(combo, dat[cols], y, ifta)
        rows.append(
            dict(
                model=name,
                cols="+".join(cols),
                key_route=int(len(combo) >= 2),
                auc_app=app,
                app_lo=alo,
                app_hi=ahi,
                auc_loo=loo,
                ext_auc=ext_auc,
                ext_lo=elo,
                ext_hi=ehi,
                ext_p=ext_p,
                p_vs_ifta_model=p,
                p_method=method,
                detail=detail,
                n=len(dat),
                n_event=int(y.sum()),
            )
        )
        print(
            f"  {name:38s} app={app:.3f}({alo:.2f}-{ahi:.2f}) loo={loo:.3f}"
            + (
                f" ext={ext_auc:.3f}({elo:.2f}-{ehi:.2f}) P={ext_p:.4f}"
                if dat_ext is not None
                else ""
            ),
            flush=True,
        )
    return pd.DataFrame(rows)


def draw_panel(ax, res: pd.DataFrame, title: str, letter: str, has_ext: bool):
    d = res.copy().reset_index(drop=True)
    y = len(d) - 1 - np.arange(len(d))
    ax.axvline(0.5, color=NL, ls=":", lw=0.7, zorder=0)
    for i, r in enumerate(d.itertuples()):
        yy = y[i]
        if r.key_route:
            ax.axhspan(yy - 0.5, yy + 0.5, color="#FCE8E6", zorder=0)
        ax.plot([r.app_lo, r.app_hi], [yy + 0.22, yy + 0.22], color=NK, lw=1.0, zorder=2)
        ax.scatter(
            [r.auc_app],
            [yy + 0.22],
            s=26,
            facecolors=NR if r.key_route else "white",
            edgecolors=NR if r.key_route else NK,
            linewidths=0.8,
            zorder=3,
        )
        ax.scatter([r.auc_loo], [yy], s=18, facecolors="none", edgecolors=NK, linewidths=0.8, zorder=3)
        if has_ext:
            ax.plot([r.ext_lo, r.ext_hi], [yy - 0.22, yy - 0.22], color=NB, lw=1.0, zorder=2)
            ax.scatter([r.ext_auc], [yy - 0.22], s=24, marker="^", color=NB, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"], fontsize=5.5)
    for tick, r in zip(ax.get_yticklabels(), d.itertuples()):
        p = r.p_vs_ifta_model
        if p == p and p < 0.05:
            tick.set_color(NR)
            tick.set_fontweight("bold")
        else:
            tick.set_color(NK)
    ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("AUC", fontsize=6.5)
    ax.set_xlim(0.30, 1.02)
    ax.set_title(title, fontsize=7.2, fontstyle="italic", loc="left", color=NK)
    ax.text(-0.32, 1.03, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    labels, colors = [], []
    for r in d.itertuples():
        txt = f"{r.auc_app:.3f} ({r.app_lo:.2f}-{r.app_hi:.2f}) | {r.auc_loo:.3f}"
        if has_ext:
            txt += f" | {r.ext_auc:.3f} ({r.ext_lo:.2f}-{r.ext_hi:.2f})"
        labels.append(txt)
        colors.append(NK)
    ax2.set_yticklabels(labels, fontsize=4.8)
    for tick, c in zip(ax2.get_yticklabels(), colors):
        tick.set_color(c)
    ax2.tick_params(axis="y", length=0)
    ax.text(
        1.02,
        1.02,
        "apparent (95% CI) | LOO" + (" | external (95% CI)" if has_ext else ""),
        transform=ax.transAxes,
        fontsize=5.2,
        ha="left",
        va="bottom",
        color=NK,
        clip_on=False,
    )


def main():
    ap = argparse.ArgumentParser(description="Figure 8 forest: 15 locked models, discovery + locked external")
    ap.add_argument(
        "--discovery",
        default=os.environ.get("PSMP_DISCOVERY_CSV", str(PACK / "discovery_patient.csv")),
        help="discovery patient-level CSV (one row per biopsy)",
    )
    ap.add_argument(
        "--external",
        default=os.environ.get("PSMP_EXTERNAL_CSV", str(PACK / "external_patient.csv")),
        help="external patient-level CSV (optional)",
    )
    args = ap.parse_args()

    need = sorted({c for cm in COLMAP.values() for c in cm.values()})
    disc_path = Path(args.discovery)
    if not disc_path.exists():
        sys.exit(f"discovery CSV not found: {disc_path}\nexpected columns: {need} + DCr + 2Y_DCr")
    dat = load_patient(disc_path, need, "discovery")
    ext_path = Path(args.external)
    dat_ext = (
        load_patient(ext_path, need, "external") if ext_path.exists() else None
    )
    print(f"discovery n={len(dat)}  external n={len(dat_ext) if dat_ext is not None else 0}")

    print("\n[A] 1-year composite endpoint", flush=True)
    res1 = eval_models(dat, dat_ext, "DCr", COLMAP["1y"], "1y")
    print("\n[B] 2-year composite endpoint", flush=True)
    res2 = eval_models(dat, dat_ext, "2Y_DCr", COLMAP["2y"], "2y")
    res1.to_csv(OUT_RES / f"369_forest_all15_1y_{TS}.csv", index=False)
    res2.to_csv(OUT_RES / f"369_forest_all15_2y_{TS}.csv", index=False)

    has_ext = dat_ext is not None
    fig, axes = plt.subplots(2, 1, figsize=(180 * MM, 200 * MM))
    fig.suptitle(
        "Prognostic discrimination for the composite kidney endpoint\n"
        f"(discovery n={len(dat)}"
        + (f"; external n={len(dat_ext)}" if has_ext else "")
        + "; 15 pre-specified models; unpenalised logistic regression)",
        fontsize=7.0,
        fontweight="bold",
        color=NK,
        y=0.995,
    )
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NR, markeredgecolor=NR, markersize=6, label="Apparent (key route)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=NK, markersize=6, label="Apparent"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=NK, markersize=6, label="LOO-CV"),
    ]
    if has_ext:
        leg.append(Line2D([0], [0], marker="^", color="w", markerfacecolor=NB, markeredgecolor=NB, markersize=6, label="External (locked)"))
    fig.legend(handles=leg, loc="upper right", frameon=False, fontsize=5.6, bbox_to_anchor=(0.98, 0.97))
    draw_panel(axes[0], res1, "a  1-year composite endpoint", "a", has_ext)
    draw_panel(axes[1], res2, "b  2-year composite endpoint", "b", has_ext)
    fig.text(
        0.5,
        0.008,
        "StandardScaler -> unpenalised logistic regression (penalty=None, max_iter=2000); "
        "95% CI = Hanley-McNeil, limits [0,1]; "
        "red labels: P<0.05 vs IFTA(model) (Wald/LRT per footnote scheme).",
        ha="center",
        fontsize=4.4,
        color=NL,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.03, 0.78, 0.94])
    out = FIG / f"369_forest_all15_{TS}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    main()
