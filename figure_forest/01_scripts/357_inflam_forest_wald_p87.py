#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
357 — Forest with p87 + raw IFTA(model); P column vs baseline IFTA(model).

AUC = LOO-CV (discrimination display).
P vs IFTA(model):
  - IFTA(model) alone: — (reference)
  - univariable morph/inflam: Wald P (assoc. with outcome; shown for completeness)
  - X + IFTA(model): Wald P of X adjusted for IFTA(model)
  - X + Y (no IFTA): nested LRT vs null (joint; not nested in IFTA)
  - X + Y + IFTA(model): nested LRT vs IFTA(model) alone
* P < 0.05
"""
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

ROOT = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data"))
RES = ROOT / "小管差异分析" / "03_results"
FIG = ROOT / "小管差异分析" / "04_figures"
RPT = ROOT / "小管差异分析" / "20260730汇报"
for d in (RES, FIG, RPT):
    d.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MM = 1 / 25.4
IFTA_COL = "IFTA_model"
IFTA_LAB = "IFTA(model)"
NB, NR, NK, NL = "#0072B5", "#BC3C29", "#2D2D2D", "#B0B0B0"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.6,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)


def loo_oof(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, int)
    oof = np.zeros(len(y), float)
    for tr, te in LeaveOneOut().split(X):
        pipe = Pipeline(
            [("s", StandardScaler()), ("m", LogisticRegression(max_iter=3000, C=1.0))]
        )
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return oof


def auc_ci(y, oof, n_boot=800, seed=0):
    auc = float(roc_auc_score(y, oof))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    bs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        bs.append(roc_auc_score(y[b], oof[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return auc, float(lo), float(hi)


def fit_logit(X, y):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, int)
    Xs = sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")
    fit = sm.Logit(y, Xs).fit(disp=False, maxiter=200)
    ors = np.exp(fit.params[1:])
    cis = np.exp(fit.conf_int()[1:])
    return fit, ors, cis, fit.pvalues[1:]


def nested_lrt(X_small, X_big, y):
    y = np.asarray(y, int)

    def _ll(X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.size == 0 or X.shape[1] == 0:
            Xs = np.ones((len(y), 1))
            return float(sm.Logit(y, Xs).fit(disp=False, maxiter=200).llf), 0
        Xs = sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")
        return float(sm.Logit(y, Xs).fit(disp=False, maxiter=200).llf), X.shape[1]

    ll0, d0 = _ll(X_small)
    ll1, d1 = _ll(X_big)
    lr = 2 * (ll1 - ll0)
    p = float(1 - stats.chi2.cdf(lr, df=max(1, d1 - d0)))
    return p


def build_patient():
    can = pd.read_csv(sorted(RES.glob("305_canonical_n45_with_p80_*.csv"))[-1]).copy()
    dual = pd.read_csv(RES / "237_tubule_dual_state_20260730_055710.csv")
    cal = pd.read_csv(RES / "351_IFTA_softband_calibration_20260802_192823.csv")
    patch = pd.read_csv(sorted(RES.glob("327_patch_dict_psi_allfeat_*.csv"))[-1])

    morph_rows = []
    for sid, g in dual.groupby("sample"):
        psi = g["psi_morph"].dropna().to_numpy(float)
        if len(psi) < 5:
            continue
        morph_rows.append(
            {
                "sample": sid,
                "tub_p87": float(np.percentile(psi, 87)),
                "tub_p80": float(np.percentile(psi, 80)),
            }
        )
    inflam_rows = []
    for sid, g in patch.groupby("sample"):
        psi = g["psi_inflam"].dropna().to_numpy(float)
        if len(psi) < 5:
            continue
        inflam_rows.append(
            {
                "sample": sid,
                "inflam_psi_p88": float(np.percentile(psi, 88)),
                "inflam_psi_p91": float(np.percentile(psi, 91)),
            }
        )
    ifta = cal[["sample", "Banff_raw"]].rename(columns={"Banff_raw": IFTA_COL})
    pat = (
        can.drop(columns=["tub_p90", "tub_p80"], errors="ignore")
        .merge(pd.DataFrame(morph_rows), on="sample", how="left")
        .merge(pd.DataFrame(inflam_rows), on="sample", how="left")
        .merge(ifta, on="sample", how="left")
    )
    for c in [
        "DCr",
        "2Y_DCr",
        IFTA_COL,
        "tub_p87",
        "tub_p80",
        "inflam_psi_p88",
        "inflam_psi_p91",
    ]:
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
            p = np.nan
            detail = "reference"
            method = "—"
        elif focus == "uni":
            # univariable association (not nested); kept so uni rows have a P
            _, ors, cis, ps = fit_logit(X, y)
            p = float(ps[0])
            detail = f"OR={ors[0]:.2f} ({cis[0,0]:.2f}-{cis[0,1]:.2f})"
            method = "Wald(uni)"
        elif focus == "wald_add":
            _, ors, cis, ps = fit_logit(X, y)
            p = float(ps[0])
            detail = f"OR={ors[0]:.2f} ({cis[0,0]:.2f}-{cis[0,1]:.2f})"
            method = "Wald|IFTA(model)"
        elif focus == "lrt_null":
            # PT + Inflam (no IFTA): joint LRT vs intercept; not nested in IFTA
            X0 = np.ones((len(y), 0))
            p = nested_lrt(X0, X, y)
            _, ors, cis, ps = fit_logit(X, y)
            detail = "; ".join(
                f"{c}:OR={ors[j]:.2f} P={ps[j]:.3f}" for j, c in enumerate(cols)
            )
            method = "LRT vs null"
        else:
            p = nested_lrt(dat[[IFTA_COL]].to_numpy(float), X, y)
            _, ors, cis, ps = fit_logit(X, y)
            detail = "; ".join(f"{c}:P={ps[j]:.3f}" for j, c in enumerate(cols))
            method = "LRT vs IFTA(model)"

        rows.append(
            dict(
                model=name,
                cols="+".join(cols),
                key_route=int(key),
                auc=auc,
                lo=lo,
                hi=hi,
                p_vs_ifta_model=p,
                p_method=method,
                detail=detail,
                n=len(dat),
                n_event=int(y.sum()),
                cohort="inflam_n42" if use_inf else "morph_n45",
            )
        )
        star = "*" if (p == p and p < 0.05) else ""
        ptxt = f"{p:.3f}{star}" if p == p else "—"
        print(
            f"  [{rows[-1]['cohort']}] {name:42s} AUC={auc:.3f}  "
            f"P_vs_IFTA(model)={ptxt} ({method})  {detail}",
            flush=True,
        )
    return pd.DataFrame(rows)


def draw_panel(ax, res: pd.DataFrame, title: str, letter: str):
    d = res.copy()
    d["_ord"] = np.arange(len(d))
    d = d.sort_values("_ord", ascending=False)
    y = np.arange(len(d))
    ax.axvline(0.5, color=NL, ls=":", lw=0.7, zorder=0)
    for i, r in enumerate(d.itertuples()):
        if r.key_route:
            ax.axhspan(i - 0.45, i + 0.45, color="#FCE8E6", zorder=0)
        ax.plot([r.lo, r.hi], [i, i], color=NK, lw=1.0, zorder=2)
        ax.scatter(
            [r.auc],
            [i],
            s=28,
            facecolors=NR if r.key_route else "white",
            edgecolors=NR if r.key_route else NK,
            linewidths=0.8,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"], fontsize=5.5)
    for tick, r in zip(ax.get_yticklabels(), d.itertuples()):
        p = r.p_vs_ifta_model
        if p == p and p < 0.05:
            tick.set_color(NR)
            tick.set_fontweight("bold")
        else:
            tick.set_color(NK)
    ax.set_xlabel("LOO-CV AUC", fontsize=6.5)
    ax.set_xlim(0.35, 0.95)
    ax.set_title(title, fontsize=7.2, fontstyle="italic", loc="left", color=NK)
    ax.text(-0.32, 1.05, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y)
    labels, colors = [], []
    for r in d.itertuples():
        p = r.p_vs_ifta_model
        if p != p:
            labels.append(f"{r.auc:.3f} ({r.lo:.2f}–{r.hi:.2f})   —")
            colors.append(NK)
        else:
            star = "*" if p < 0.05 else ""
            labels.append(f"{r.auc:.3f} ({r.lo:.2f}–{r.hi:.2f})   {p:.3f}{star}")
            colors.append(NR if p < 0.05 else NK)
    ax2.set_yticklabels(labels, fontsize=5.0)
    for tick, c in zip(ax2.get_yticklabels(), colors):
        tick.set_color(c)
        if c == NR:
            tick.set_fontweight("bold")
    ax2.tick_params(axis="y", length=0)
    ax.text(
        1.02,
        1.02,
        f"AUC (95% CI)     P vs {IFTA_LAB}†",
        transform=ax.transAxes,
        fontsize=5.2,
        ha="left",
        va="bottom",
        color=NK,
        clip_on=False,
    )


def main():
    print("=" * 70)
    print("357 Wald forest · p87 + raw IFTA(model)")
    print("=" * 70)
    pat = build_patient()
    need = ["tub_p87", "tub_p80", IFTA_COL, "DCr", "2Y_DCr"]
    dat45 = pat.dropna(subset=need).copy()
    dat_inf = pat.dropna(subset=need + ["inflam_psi_p91", "inflam_psi_p88"]).copy()
    print(f"n45={len(dat45)} n42={len(dat_inf)}", flush=True)

    models_1y = [
        (f"PT p87 + Inflam ψ p91 + {IFTA_LAB}", ["tub_p87", "inflam_psi_p91", IFTA_COL], True, "lrt"),
        (f"Inflam ψ p91 + {IFTA_LAB}", ["inflam_psi_p91", IFTA_COL], True, "wald_add"),
        (f"PT p87 + {IFTA_LAB}", ["tub_p87", IFTA_COL], True, "wald_add"),
        ("PT p87 + Inflam ψ p91", ["tub_p87", "inflam_psi_p91"], True, "lrt_null"),
        ("Inflam ψ p91", ["inflam_psi_p91"], False, "uni"),
        ("PT p87", ["tub_p87"], False, "uni"),
        (IFTA_LAB, [IFTA_COL], False, "uni"),
    ]
    models_2y = [
        (f"PT p80 + Inflam ψ p88 + {IFTA_LAB}", ["tub_p80", "inflam_psi_p88", IFTA_COL], True, "lrt"),
        (f"Inflam ψ p88 + {IFTA_LAB}", ["inflam_psi_p88", IFTA_COL], True, "wald_add"),
        (f"PT p80 + {IFTA_LAB}", ["tub_p80", IFTA_COL], True, "wald_add"),
        ("PT p80 + Inflam ψ p88", ["tub_p80", "inflam_psi_p88"], True, "lrt_null"),
        ("Inflam ψ p88", ["inflam_psi_p88"], False, "uni"),
        ("PT p80", ["tub_p80"], False, "uni"),
        (IFTA_LAB, [IFTA_COL], False, "uni"),
    ]

    print("\n[A] 1y", flush=True)
    res1 = eval_models(dat45, dat_inf, "DCr", models_1y, seed0=1)
    print("\n[B] 2y", flush=True)
    res2 = eval_models(dat45, dat_inf, "2Y_DCr", models_2y, seed0=100)

    res1.to_csv(RES / f"357_forest_wald_p87_1y_{TS}.csv", index=False)
    res2.to_csv(RES / f"357_forest_wald_p87_2y_{TS}.csv", index=False)
    for p in (RES / f"357_forest_wald_p87_1y_{TS}.csv", RES / f"357_forest_wald_p87_2y_{TS}.csv"):
        (RPT / p.name).write_bytes(p.read_bytes())

    fig, axes = plt.subplots(2, 1, figsize=(168 * MM, 165 * MM))
    fig.suptitle(
        "Prognostic discrimination for kidney-function decline\n"
        f"(raw {IFTA_LAB}=100×SC2/K2; 1y morph p87; P vs {IFTA_LAB}; "
        f"n=45 / inflam n={len(dat_inf)})",
        fontsize=7.0,
        fontweight="bold",
        color=NK,
        y=0.995,
    )
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=NK, markersize=6, label="Univariable"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=NR,
            markeredgecolor=NR,
            markersize=6,
            label=f"Key route (multi / +{IFTA_LAB})",
        ),
    ]
    fig.legend(handles=leg, loc="upper right", frameon=False, fontsize=5.6, bbox_to_anchor=(0.98, 0.97))
    draw_panel(axes[0], res1, "a  1-year creatinine doubling", "a")
    draw_panel(axes[1], res2, "b  2-year creatinine doubling", "b")
    fig.text(
        0.5,
        0.008,
        f"{IFTA_LAB} = 100×SC2/K2 RAW (reference row). AUC = LOO-CV. "
        f"†P vs {IFTA_LAB}: X+IFTA = Wald P of X|IFTA; X+Y+IFTA = LRT vs IFTA alone; "
        "PT+Inflam (no IFTA) = LRT vs null (joint); univariable = Wald vs null; "
        "IFTA row = —. * P<0.05.",
        ha="center",
        fontsize=4.4,
        color=NL,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.04, 0.76, 0.94])
    out = FIG / f"357_inflam_forest_Pvs_IFTAmodel_p87_{TS}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
        (RPT / f"{out.name}.{ext}").write_bytes(Path(f"{out}.{ext}").read_bytes())
    plt.close(fig)

    md = RES / f"357_wald_p87_forest_{TS}.md"
    md.write_text(
        f"""# 357 森林图（p87 + raw IFTA；P vs IFTA(model)）

- IFTA(model) = 100×SC2/K2 未校准（参考行 P = —）
- 1y 形态 = p87
- **P vs IFTA(model)**：X+IFTA = Wald(X|IFTA)；三变量 = LRT vs IFTA；PT+Inflam = LRT vs null；单因素 = Wald vs null

## 1y
{res1.to_string(index=False)}

## 2y
{res2.to_string(index=False)}

图：`{out.name}.png`
""",
        encoding="utf-8",
    )
    (RPT / md.name).write_text(md.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"→ {out}.png")
    print("DONE")


if __name__ == "__main__":
    main()
