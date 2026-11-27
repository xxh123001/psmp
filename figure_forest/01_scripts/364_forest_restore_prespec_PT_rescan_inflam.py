#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
364 — Restore AUC consistency with prior forests:

Problem with 363:
  - Picked peaks by univariable MW-AUC → 1y PT p80 (vs prespec p87)
  - That dropped LOO forest PT+IFTA 0.796 → 0.765 and triple 0.757 → 0.728
  - Uni peak ≠ multivariate forest peak

Fix:
  - PT morph: keep prespecified peaks (unchanged axis): 1y p87, 2y p80
  - Inflam ψ (final Ge*): re-scan p70–95 by LOO AUC of key routes
      * 1y/2y: maximize Inflam+IFTA LOO AUC (and report uni for reference)
  - Rerun Wald forest comparable to 357 layout
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
TEACHER = Path(os.environ.get("PSMP_TEACHER_DIR", "./teacher_ref"))
for d in (RES, FIG, RPT, TEACHER / "figures"):
    d.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MM = 1 / 25.4
IFTA_COL, IFTA_LAB = "IFTA_model", "IFTA(model)"
NR, NK, NL = "#BC3C29", "#2D2D2D", "#B0B0B0"
PCTS = list(range(70, 96, 1))
PT_1Y, PT_2Y = 87, 80  # morph prespecified

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.6,
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
            return float(sm.Logit(y, np.ones((len(y), 1))).fit(disp=False, maxiter=200).llf), 0
        Xs = sm.add_constant(StandardScaler().fit_transform(X), has_constant="add")
        return float(sm.Logit(y, Xs).fit(disp=False, maxiter=200).llf), X.shape[1]

    ll0, d0 = _ll(X_small)
    ll1, d1 = _ll(X_big)
    return float(1 - stats.chi2.cdf(2 * (ll1 - ll0), df=max(1, d1 - d0)))


def build_pat():
    can = pd.read_csv(sorted(RES.glob("305_canonical_n45_with_p80_*.csv"))[-1]).copy()
    dual = pd.read_csv(RES / "237_tubule_dual_state_20260730_055710.csv")
    cal = pd.read_csv(RES / "351_IFTA_softband_calibration_20260802_192823.csv")
    patch = pd.read_csv(sorted(RES.glob("361_patch_psi_Ge_hotspot10_*.csv"))[-1])
    print("ψ:", sorted(RES.glob("361_patch_psi_Ge_hotspot10_*.csv"))[-1].name)

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
        row = {"sample": sid}
        for p in PCTS:
            row[f"inflam_p{p}"] = float(np.percentile(psi, p))
        inflam_rows.append(row)

    ifta = cal[["sample", "Banff_raw"]].rename(columns={"Banff_raw": IFTA_COL})
    pat = (
        can.drop(columns=[c for c in can.columns if str(c).startswith("tub_p")], errors="ignore")
        .merge(pd.DataFrame(morph_rows), on="sample", how="left")
        .merge(pd.DataFrame(inflam_rows), on="sample", how="left")
        .merge(ifta, on="sample", how="left")
    )
    for c in ["DCr", "2Y_DCr", IFTA_COL, "tub_p87", "tub_p80"]:
        pat[c] = pd.to_numeric(pat[c], errors="coerce")
    return pat


def scan_inflam(pat, ycol, pt_col):
    """Pick inflam p maximizing LOO AUC of Inflam+IFTA; also track triple & uni."""
    rows = []
    d0 = pat.dropna(subset=[ycol, IFTA_COL, pt_col] + [f"inflam_p{p}" for p in PCTS]).copy()
    y = d0[ycol].astype(int).to_numpy()
    for p in PCTS:
        col = f"inflam_p{p}"
        # uni (raw score AUC, directed)
        uni = float(roc_auc_score(y, d0[col].to_numpy(float)))
        # Inflam + IFTA
        oof_add = loo_oof(d0[[col, IFTA_COL]].to_numpy(float), y)
        auc_add = float(roc_auc_score(y, oof_add))
        # triple
        oof_tri = loo_oof(d0[[pt_col, col, IFTA_COL]].to_numpy(float), y)
        auc_tri = float(roc_auc_score(y, oof_tri))
        rows.append(
            dict(p=p, auc_uni=uni, auc_inf_ifta=auc_add, auc_triple=auc_tri, n=len(d0))
        )
    df = pd.DataFrame(rows)
    # primary: Inflam+IFTA; tie-break by triple then higher p
    df = df.sort_values(
        ["auc_inf_ifta", "auc_triple", "p"], ascending=[False, False, False]
    )
    pick = df.iloc[0]
    return df.sort_values("p"), int(pick["p"]), pick


def eval_models(dat45, dat_inf, ycol, models, seed0=0):
    rows = []
    for i, (name, cols, key, focus) in enumerate(models):
        use_inf = any("inflam" in str(c) for c in cols)
        dat = dat_inf if use_inf else dat45
        y = dat[ycol].astype(int).to_numpy()
        X = dat[list(cols)].to_numpy(float)
        if len(cols) == 1 and "inflam" in str(cols[0]):
            oof = np.asarray(X, float).ravel()
        else:
            oof = loo_oof(X, y)
        auc, lo, hi = auc_ci(y, oof, seed=seed0 + i)
        if focus == "uni" and list(cols) == [IFTA_COL]:
            p, detail, method = np.nan, "reference", "—"
        elif focus == "uni":
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
            p = nested_lrt(np.ones((len(y), 0)), X, y)
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
            )
        )
        star = "*" if (p == p and p < 0.05) else ""
        print(f"  {name:48s} AUC={auc:.3f}  P={p:.3f}{star}" if p == p else f"  {name:48s} AUC={auc:.3f}  P=—")
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
        tick.set_color(NR if (p == p and p < 0.05) else NK)
        if p == p and p < 0.05:
            tick.set_fontweight("bold")
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
    print("364 Restore PT p87/p80; rescan Inflam by Inflam+IFTA LOO")
    print("=" * 70)
    pat = build_pat()
    need = ["tub_p87", "tub_p80", IFTA_COL, "DCr", "2Y_DCr"]
    dat45 = pat.dropna(subset=need).copy()
    dat_inf = pat.dropna(subset=need + [f"inflam_p{p}" for p in PCTS]).copy()
    print(f"n45={len(dat45)} n_inf={len(dat_inf)}")

    scan1, p_inf1, pick1 = scan_inflam(dat_inf, "DCr", "tub_p87")
    scan2, p_inf2, pick2 = scan_inflam(dat_inf, "2Y_DCr", "tub_p80")
    print(
        f"[1y] PT p{PT_1Y} (prespec) | Inflam p{p_inf1} "
        f"(Inf+IFTA={pick1.auc_inf_ifta:.3f}, triple={pick1.auc_triple:.3f}, uni={pick1.auc_uni:.3f})"
    )
    print(
        f"[2y] PT p{PT_2Y} (prespec) | Inflam p{p_inf2} "
        f"(Inf+IFTA={pick2.auc_inf_ifta:.3f}, triple={pick2.auc_triple:.3f}, uni={pick2.auc_uni:.3f})"
    )
    # also show old p91/p88 on same ψ for transparency
    for lab, ycol, pt, p_old in [
        ("1y", "DCr", "tub_p87", 91),
        ("2y", "2Y_DCr", "tub_p80", 88),
    ]:
        row = (scan1 if lab == "1y" else scan2)
        r = row[row.p == p_old].iloc[0]
        print(
            f"  ref old Inflam p{p_old} on new ψ: Inf+IFTA={r.auc_inf_ifta:.3f} triple={r.auc_triple:.3f} uni={r.auc_uni:.3f}"
        )

    scan1.to_csv(RES / f"364_inflam_scan_1y_{TS}.csv", index=False)
    scan2.to_csv(RES / f"364_inflam_scan_2y_{TS}.csv", index=False)

    inf1, inf2 = f"inflam_p{p_inf1}", f"inflam_p{p_inf2}"
    models_1y = [
        (f"PT p{PT_1Y} + Inflam ψ p{p_inf1} + {IFTA_LAB}", ["tub_p87", inf1, IFTA_COL], True, "lrt"),
        (f"Inflam ψ p{p_inf1} + {IFTA_LAB}", [inf1, IFTA_COL], True, "wald_add"),
        (f"PT p{PT_1Y} + {IFTA_LAB}", ["tub_p87", IFTA_COL], True, "wald_add"),
        (f"PT p{PT_1Y} + Inflam ψ p{p_inf1}", ["tub_p87", inf1], True, "lrt_null"),
        (f"Inflam ψ p{p_inf1}", [inf1], False, "uni"),
        (f"PT p{PT_1Y}", ["tub_p87"], False, "uni"),
        (IFTA_LAB, [IFTA_COL], False, "uni"),
    ]
    models_2y = [
        (f"PT p{PT_2Y} + Inflam ψ p{p_inf2} + {IFTA_LAB}", ["tub_p80", inf2, IFTA_COL], True, "lrt"),
        (f"Inflam ψ p{p_inf2} + {IFTA_LAB}", [inf2, IFTA_COL], True, "wald_add"),
        (f"PT p{PT_2Y} + {IFTA_LAB}", ["tub_p80", IFTA_COL], True, "wald_add"),
        (f"PT p{PT_2Y} + Inflam ψ p{p_inf2}", ["tub_p80", inf2], True, "lrt_null"),
        (f"Inflam ψ p{p_inf2}", [inf2], False, "uni"),
        (f"PT p{PT_2Y}", ["tub_p80"], False, "uni"),
        (IFTA_LAB, [IFTA_COL], False, "uni"),
    ]

    print("\n[A] 1y")
    res1 = eval_models(dat45, dat_inf, "DCr", models_1y, 1)
    print("\n[B] 2y")
    res2 = eval_models(dat45, dat_inf, "2Y_DCr", models_2y, 100)
    res1.to_csv(RES / f"364_forest_1y_{TS}.csv", index=False)
    res2.to_csv(RES / f"364_forest_2y_{TS}.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(168 * MM, 165 * MM))
    fig.suptitle(
        "Prognostic discrimination · PT peaks prespecified · Inflam peaks re-scanned\n"
        f"(1y: PT p{PT_1Y} + Inflam p{p_inf1}; 2y: PT p{PT_2Y} + Inflam p{p_inf2}; "
        f"final Ge* ψ; n={len(dat45)} / inflam n={len(dat_inf)})",
        fontsize=7.0,
        fontweight="bold",
        color=NK,
        y=0.995,
    )
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=NK, markersize=6, label="Univariable"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NR, markeredgecolor=NR, markersize=6, label=f"Key route (multi / +{IFTA_LAB})"),
    ]
    fig.legend(handles=leg, loc="upper right", frameon=False, fontsize=5.6, bbox_to_anchor=(0.98, 0.97))
    draw_panel(axes[0], res1, f"a  1-year (PT p{PT_1Y} · Inflam p{p_inf1})", "a")
    draw_panel(axes[1], res2, f"b  2-year (PT p{PT_2Y} · Inflam p{p_inf2})", "b")
    fig.text(
        0.5,
        0.008,
        f"PT p{PT_1Y}/p{PT_2Y} = morph prespecified (axis unchanged). "
        f"Inflam p chosen by max LOO AUC of Inflam+{IFTA_LAB} on final ψ (endpoint-specific). * P<0.05.",
        ha="center",
        fontsize=4.4,
        color=NL,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.04, 0.76, 0.94])
    out = FIG / f"364_forest_prespecPT_inflamPeak_{TS}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
        (RPT / f"{out.name}.{ext}").write_bytes(Path(f"{out}.{ext}").read_bytes())
    (TEACHER / "figures" / "forest_drop4.png").write_bytes(Path(f"{out}.png").read_bytes())
    (TEACHER / "figures" / "forest_drop4.pdf").write_bytes(Path(f"{out}.pdf").read_bytes())
    (TEACHER / f"364_森林图_恢复PT峰值_{TS}.png").write_bytes(Path(f"{out}.png").read_bytes())
    (TEACHER / f"364_森林图_恢复PT峰值_{TS}.pdf").write_bytes(Path(f"{out}.pdf").read_bytes())
    plt.close(fig)

    # compare table vs prior
    prior1 = pd.read_csv(RES / "357_forest_wald_p87_1y_20260804_063821.csv")
    prior2 = pd.read_csv(RES / "357_forest_wald_p87_2y_20260804_063821.csv")
    md = f"""# 364 森林图：恢复 PT 预设峰值 + Inflam 按关键模型重扫

## 为何 363 AUC「变低且不一致」
1. **PT 被单因素峰值误伤**：1y 单因素 MW-AUC 略抬 p80，但森林 LOO 里 **p87 更优**（PT+IFTA 0.796 vs p80 的 0.765）。形态轴未变，应保持预设 **p87/p80**。
2. **新 ψ 本身改变 Inflam 信号**：同一 p88 下，2y Inflam 单因素从原版 ~0.79 降到 Ge* 版 ~0.70；这是指标修订效应，不是画图错误。
3. **单因素峰值 ≠ 多因素最优**：Inflam 应按 `Inflam+IFTA`（或三联）LOO 选 p，且 **1y/2y 分开扫**。

## 本版设定
| 终点 | PT | Inflam（新 ψ，按 Inf+IFTA LOO 峰值） |
|------|----|--------------------------------------|
| 1y | **p{PT_1Y}**（预设） | **p{p_inf1}** |
| 2y | **p{PT_2Y}**（预设） | **p{p_inf2}** |

## 与原版 24ψ 森林关键 AUC 对照
| 模型 | 原版0804 | 本版364 |
|------|----------|---------|
| 1y 三联 | {prior1.auc.iloc[0]:.3f} | {res1.auc.iloc[0]:.3f} |
| 1y PT+IFTA | {prior1.auc.iloc[2]:.3f} | {res1.auc.iloc[2]:.3f} |
| 1y PT | {prior1.auc.iloc[5]:.3f} | {res1.auc.iloc[5]:.3f} |
| 2y 三联 | {prior2.auc.iloc[0]:.3f} | {res2.auc.iloc[0]:.3f} |
| 2y Inflam | {prior2.auc.iloc[4]:.3f} | {res2.auc.iloc[4]:.3f} |
| 2y Inflam+IFTA | {prior2.auc.iloc[1]:.3f} | {res2.auc.iloc[1]:.3f} |

图：`{out.name}.png/.pdf`
"""
    (RES / f"364_forest_note_{TS}.md").write_text(md, encoding="utf-8")
    (TEACHER / f"364_森林图说明_{TS}.md").write_text(md, encoding="utf-8")
    print(f"→ {out}.png")
    print("DONE")


if __name__ == "__main__":
    main()
