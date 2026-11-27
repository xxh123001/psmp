#!/usr/bin/env python3
# Exact 357 forest style · current cluster5 ψ · Inflam peaks p87 / p95
from __future__ import annotations
from datetime import datetime
import os
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
# default: run inside this pack; fall back to original project if missing
_proj = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data")) / "小管差异分析" / "03_results"
RES = PACK / "03_source_features" if (PACK / "03_source_features").exists() else _proj
OUT_RES = PACK / "04_results"
FIG = PACK / "05_figure"
RPT = PACK / "04_results"
TEA = PACK
for d in (OUT_RES, FIG):
    d.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MM = 1 / 25.4
IFTA_COL = "IFTA_model"
IFTA_LAB = "IFTA(model)"
NB, NR, NK, NL = "#0072B5", "#BC3C29", "#2D2D2D", "#B0B0B0"
P_INF_1Y, P_INF_2Y = 87, 95

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

def loo_oof(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    y = np.asarray(y, int)
    oof = np.zeros(len(y), float)
    for tr, te in LeaveOneOut().split(X):
        pipe = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=3000, C=1.0))])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof

def auc_ci(y, oof, n_boot=800, seed=0):
    auc = float(roc_auc_score(y, oof))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); bs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2: continue
        bs.append(roc_auc_score(y[b], oof[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return auc, float(lo), float(hi)

def fit_logit(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1: X = X.reshape(-1, 1)
    y = np.asarray(y, int)
    Xs = sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")
    fit = sm.Logit(y, Xs).fit(disp=False, maxiter=200)
    return fit, np.exp(fit.params[1:]), np.exp(fit.conf_int()[1:]), fit.pvalues[1:]

def nested_lrt(X_small, X_big, y):
    y = np.asarray(y, int)
    def _ll(X):
        X = np.asarray(X, float)
        if X.ndim == 1: X = X.reshape(-1, 1)
        if X.size == 0 or X.shape[1] == 0:
            return float(sm.Logit(y, np.ones((len(y), 1))).fit(disp=False, maxiter=200).llf), 0
        Xs = sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")
        return float(sm.Logit(y, Xs).fit(disp=False, maxiter=200).llf), X.shape[1]
    ll0, d0 = _ll(X_small); ll1, d1 = _ll(X_big)
    return float(1 - stats.chi2.cdf(2 * (ll1 - ll0), df=max(1, d1 - d0)))

def build_patient():
    can = pd.read_csv(sorted(RES.glob("305_canonical_n45_with_p80_*.csv"))[-1]).copy()
    dual = pd.read_csv(RES / "237_tubule_dual_state_20260730_055710.csv")
    cal = pd.read_csv(RES / "351_IFTA_softband_calibration_20260802_192823.csv")
    # prefer explicit cluster5; fallback latest 327
    c5 = sorted(RES.glob("365_patch_psi_Ge_cluster5_*.csv"))
    patch = pd.read_csv(c5[-1] if c5 else sorted(RES.glob("327_patch_dict_psi_allfeat_*.csv"))[-1])
    print("ψ source:", Path(patch.attrs.get('path','')).name if False else (c5[-1].name if c5 else '327 latest'))

    morph_rows = []
    for sid, g in dual.groupby("sample"):
        psi = g["psi_morph"].dropna().to_numpy(float)
        if len(psi) < 5: continue
        morph_rows.append({"sample": sid, "tub_p87": float(np.percentile(psi, 87)), "tub_p80": float(np.percentile(psi, 80))})
    inflam_rows = []
    for sid, g in patch.groupby("sample"):
        psi = g["psi_inflam"].dropna().to_numpy(float)
        if len(psi) < 5: continue
        inflam_rows.append({
            "sample": sid,
            f"inflam_psi_p{P_INF_1Y}": float(np.percentile(psi, P_INF_1Y)),
            f"inflam_psi_p{P_INF_2Y}": float(np.percentile(psi, P_INF_2Y)),
        })
    ifta = cal[["sample", "Banff_raw"]].rename(columns={"Banff_raw": IFTA_COL})
    pat = (
        can.drop(columns=["tub_p90", "tub_p80"], errors="ignore")
        .merge(pd.DataFrame(morph_rows), on="sample", how="left")
        .merge(pd.DataFrame(inflam_rows), on="sample", how="left")
        .merge(ifta, on="sample", how="left")
    )
    for c in ["DCr", "2Y_DCr", IFTA_COL, "tub_p87", "tub_p80", f"inflam_psi_p{P_INF_1Y}", f"inflam_psi_p{P_INF_2Y}"]:
        pat[c] = pd.to_numeric(pat[c], errors="coerce")
    return pat

def _needs_inflam(cols):
    return any(str(c).startswith("inflam_") for c in cols)

def eval_models(dat45, dat_inf, ycol, models, seed0=0):
    rows = []
    for i, (name, cols, key, focus) in enumerate(models):
        use_inf = _needs_inflam(cols)
        dat = dat_inf if use_inf else dat45
        y = dat[ycol].astype(int).to_numpy()
        X = dat[list(cols)].to_numpy(float)
        if len(cols) == 1 and str(cols[0]).startswith("inflam_"):
            oof = np.asarray(X, float).ravel()
        else:
            oof = loo_oof(X, y)
        auc, lo, hi = auc_ci(y, oof, seed=seed0 + i)
        if focus == "uni" and list(cols) == [IFTA_COL]:
            p, detail, method = np.nan, "reference", "—"
        elif focus == "uni":
            _, ors, cis, ps = fit_logit(X, y)
            p = float(ps[0]); detail = f"OR={ors[0]:.2f} ({cis[0,0]:.2f}-{cis[0,1]:.2f})"; method = "Wald(uni)"
        elif focus == "wald_add":
            _, ors, cis, ps = fit_logit(X, y)
            p = float(ps[0]); detail = f"OR={ors[0]:.2f} ({cis[0,0]:.2f}-{cis[0,1]:.2f})"; method = "Wald|IFTA(model)"
        elif focus == "lrt_null":
            p = nested_lrt(np.ones((len(y), 0)), X, y)
            _, ors, cis, ps = fit_logit(X, y)
            detail = "; ".join(f"{c}:OR={ors[j]:.2f} P={ps[j]:.3f}" for j, c in enumerate(cols))
            method = "LRT vs null"
        else:
            p = nested_lrt(dat[[IFTA_COL]].to_numpy(float), X, y)
            _, ors, cis, ps = fit_logit(X, y)
            detail = "; ".join(f"{c}:P={ps[j]:.3f}" for j, c in enumerate(cols))
            method = "LRT vs IFTA(model)"
        rows.append(dict(model=name, key_route=int(key), auc=auc, lo=lo, hi=hi, p_vs_ifta_model=p, p_method=method, detail=detail, n=len(dat)))
        star = "*" if (p == p and p < 0.05) else ""
        print(f"  {name:42s} AUC={auc:.3f}  P={p:.3f}{star}" if p==p else f"  {name:42s} AUC={auc:.3f}  P=—")
    return pd.DataFrame(rows)

def draw_panel(ax, res, title, letter):
    d = res.copy()
    d["_ord"] = np.arange(len(d))
    d = d.sort_values("_ord", ascending=False)
    y = np.arange(len(d))
    ax.axvline(0.5, color=NL, ls=":", lw=0.7, zorder=0)
    for i, r in enumerate(d.itertuples()):
        if r.key_route:
            ax.axhspan(i - 0.45, i + 0.45, color="#FCE8E6", zorder=0)
        ax.plot([r.lo, r.hi], [i, i], color=NK, lw=1.0, zorder=2)
        ax.scatter([r.auc], [i], s=28,
                   facecolors=NR if r.key_route else "white",
                   edgecolors=NR if r.key_route else NK, linewidths=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"], fontsize=5.5)
    for tick, r in zip(ax.get_yticklabels(), d.itertuples()):
        p = r.p_vs_ifta_model
        if p == p and p < 0.05:
            tick.set_color(NR); tick.set_fontweight("bold")
        else:
            tick.set_color(NK)
    ax.set_xlabel("LOO-CV AUC", fontsize=6.5)
    ax.set_xlim(0.35, 0.95)
    ax.set_title(title, fontsize=7.2, fontstyle="italic", loc="left", color=NK)
    ax.text(-0.32, 1.05, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(y)
    labels, colors = [], []
    for r in d.itertuples():
        p = r.p_vs_ifta_model
        if p != p:
            labels.append(f"{r.auc:.3f} ({r.lo:.2f}–{r.hi:.2f})   —"); colors.append(NK)
        else:
            star = "*" if p < 0.05 else ""
            labels.append(f"{r.auc:.3f} ({r.lo:.2f}–{r.hi:.2f})   {p:.3f}{star}")
            colors.append(NR if p < 0.05 else NK)
    ax2.set_yticklabels(labels, fontsize=5.0)
    for tick, c in zip(ax2.get_yticklabels(), colors):
        tick.set_color(c)
        if c == NR: tick.set_fontweight("bold")
    ax2.tick_params(axis="y", length=0)
    ax.text(1.02, 1.02, f"AUC (95% CI)     P vs {IFTA_LAB}†",
            transform=ax.transAxes, fontsize=5.2, ha="left", va="bottom", color=NK, clip_on=False)

# ---- main ----
pat = build_patient()
need = ["tub_p87", "tub_p80", IFTA_COL, "DCr", "2Y_DCr"]
inf1, inf2 = f"inflam_psi_p{P_INF_1Y}", f"inflam_psi_p{P_INF_2Y}"
dat45 = pat.dropna(subset=need).copy()
dat_inf = pat.dropna(subset=need + [inf1, inf2]).copy()
print(f"n45={len(dat45)} n42={len(dat_inf)}")

models_1y = [
    (f"PT p87 + Inflam ψ p{P_INF_1Y} + {IFTA_LAB}", ["tub_p87", inf1, IFTA_COL], True, "lrt"),
    (f"Inflam ψ p{P_INF_1Y} + {IFTA_LAB}", [inf1, IFTA_COL], True, "wald_add"),
    (f"PT p87 + {IFTA_LAB}", ["tub_p87", IFTA_COL], True, "wald_add"),
    (f"PT p87 + Inflam ψ p{P_INF_1Y}", ["tub_p87", inf1], True, "lrt_null"),
    (f"Inflam ψ p{P_INF_1Y}", [inf1], False, "uni"),
    ("PT p87", ["tub_p87"], False, "uni"),
    (IFTA_LAB, [IFTA_COL], False, "uni"),
]
models_2y = [
    (f"PT p80 + Inflam ψ p{P_INF_2Y} + {IFTA_LAB}", ["tub_p80", inf2, IFTA_COL], True, "lrt"),
    (f"Inflam ψ p{P_INF_2Y} + {IFTA_LAB}", [inf2, IFTA_COL], True, "wald_add"),
    (f"PT p80 + {IFTA_LAB}", ["tub_p80", IFTA_COL], True, "wald_add"),
    (f"PT p80 + Inflam ψ p{P_INF_2Y}", ["tub_p80", inf2], True, "lrt_null"),
    (f"Inflam ψ p{P_INF_2Y}", [inf2], False, "uni"),
    ("PT p80", ["tub_p80"], False, "uni"),
    (IFTA_LAB, [IFTA_COL], False, "uni"),
]

print("\n[A] 1y"); res1 = eval_models(dat45, dat_inf, "DCr", models_1y, seed0=1)
print("\n[B] 2y"); res2 = eval_models(dat45, dat_inf, "2Y_DCr", models_2y, seed0=100)
res1.to_csv(OUT_RES / f"369_forest_357style_1y_p{P_INF_1Y}_{TS}.csv", index=False)
res2.to_csv(OUT_RES / f"369_forest_357style_2y_p{P_INF_2Y}_{TS}.csv", index=False)

fig, axes = plt.subplots(2, 1, figsize=(168 * MM, 165 * MM))
fig.suptitle(
    "Prognostic discrimination for kidney-function decline\n"
    f"(raw {IFTA_LAB}=100×SC2/K2; Ge*+cluster≥5 ψ; 1y Inflam p{P_INF_1Y} / 2y Inflam p{P_INF_2Y}; "
    f"P vs {IFTA_LAB}; n=45 / inflam n={len(dat_inf)})",
    fontsize=7.0, fontweight="bold", color=NK, y=0.995,
)
leg = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=NK, markersize=6, label="Univariable"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=NR, markeredgecolor=NR, markersize=6, label=f"Key route (multi / +{IFTA_LAB})"),
]
fig.legend(handles=leg, loc="upper right", frameon=False, fontsize=5.6, bbox_to_anchor=(0.98, 0.97))
draw_panel(axes[0], res1, "a  1-year creatinine doubling", "a")
draw_panel(axes[1], res2, "b  2-year creatinine doubling", "b")
fig.text(
    0.5, 0.008,
    f"{IFTA_LAB} = 100×SC2/K2 RAW (reference row). AUC = LOO-CV. "
    f"†P vs {IFTA_LAB}: X+IFTA = Wald P of X|IFTA; X+Y+IFTA = LRT vs IFTA alone; "
    "PT+Inflam (no IFTA) = LRT vs null (joint); univariable = Wald vs null; "
    "IFTA row = —. * P<0.05.",
    ha="center", fontsize=4.4, color=NL, style="italic",
)
fig.tight_layout(rect=[0, 0.04, 0.76, 0.94])
out = FIG / f"369_inflam_forest_357style_p{P_INF_1Y}_p{P_INF_2Y}_{TS}"
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
plt.close(fig)
print("→", out)
print("→ 蒋老师/figures/forest_drop4.png")
