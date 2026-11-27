#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
365 — ψ heatmap: Ge* + peri-tub cluster≥5 in axis (Hotspot≥10 descriptive only)
  - drop A2/B3/C1/D2
  - Axis uses D3a_frac_cluster5 (≥5 @50); Hotspot≥10 kept out of ψ
  - G default = epithelial Ge1–Ge4 only (no lumen sum, no G5 in ψ)
  - NaN impute = zero (meeting preference)
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
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

ROOT = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data"))
CLIN = ROOT / "011_data" / "clinical"
RES = ROOT / "小管差异分析" / "03_results"
FIG = ROOT / "小管差异分析" / "04_figures"
RPT = ROOT / "小管差异分析" / "20260730汇报"
TEACHER = Path(os.environ.get("PSMP_TEACHER_DIR", "./teacher_ref"))
for d in (RES, FIG, RPT, TEACHER / "figures"):
    d.mkdir(parents=True, exist_ok=True)

TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MM = 1 / 25.4
NB, NG, NY = "#0072B5", "#20854E", "#E18727"

# feature, English label, higher_worse, Chinese group
PANEL = [
    ("A1_inflam_dens_mm2", "Patch dens /mm²", True, "Patch全局负荷"),
    ("A3_grid_dens_max", "Grid dens max", True, "Patch全局负荷"),
    ("A3b_grid_dens_p90", "Grid dens p90", True, "Patch全局负荷"),
    ("B1_pct_inflam_area", "Inflamed area %", True, "空间范围与灶性"),
    ("B2_focality_top10", "Focality top10%", True, "空间范围与灶性"),
    ("C2_frac_near_tub_R", "Near-tubule frac", True, "距离与接触"),
    ("C3_inflam_nn_median", "Inflam–inflam NN (µm)", False, "距离与接触"),
    ("D1_dens50_mean", "Peri-tub dens@50", True, "小管周围(间质侧)"),
    ("D1c_dens50_cv", "Peri-tub dens CV", True, "小管周围(间质侧)"),
    ("D1d_dens25_mean", "Peri-tub dens@25", True, "小管周围(间质侧)"),
    ("D1e_dens100_mean", "Peri-tub dens@100", True, "小管周围(间质侧)"),
    ("D3a_frac_cluster5", "Peri-tub cluster ≥5 @50", True, "小管周围(间质侧)"),
    ("D4_ring_grad_mean", "Radial ring gradient", True, "小管周围(间质侧)"),
    ("D5_local_disp_mean", "Local distance CV", True, "小管周围(间质侧)"),
    ("Ge1_epi_n_mean", "Epi-tub count", True, "上皮内"),
    ("Ge2_frac_tub_with_epi", "Tubules with epi %", True, "上皮内"),
    ("Ge3_epi_dens_mean", "Epi-tub dens", True, "上皮内"),
    ("Ge4_epi_to_peri_ratio", "Epi/peri ratio (κ=5)", True, "上皮内"),
    ("E3_atrophic_cond_dens", "Atrophy-coloc dens Δ", True, "萎缩共定位"),
]
FEATS = [f for f, _, _, _ in PANEL]
LABELS = {f: lab for f, lab, _, _ in PANEL}
HIGHER = {f: hw for f, _, hw, _ in PANEL}
GROUPS = {f: g for f, _, _, g in PANEL}

heat_cmap = LinearSegmentedColormap.from_list(
    "rb",
    [
        "#2166AC",
        "#4393C3",
        "#92C5DE",
        "#D1E5F0",
        "#F7F7F7",
        "#FDDBC7",
        "#F4A582",
        "#D6604D",
        "#B2182B",
    ],
    N=256,
)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)


def main():
    feat_path = sorted(RES.glob("360_patch_features_GeGl_hotspot10_*.csv"))[-1]
    print(f"load {feat_path.name}")
    df = pd.read_csv(feat_path)
    clin = pd.concat(
        [
            pd.read_csv(CLIN / "V4_patient_train_56.csv"),
            pd.read_csv(CLIN / "V4_patient_valid_30.csv"),
        ],
        ignore_index=True,
    ).rename(columns={"Slide": "sample"})
    clin["eGFR_CKDEPI"] = pd.to_numeric(clin["eGFR_CKDEPI"], errors="coerce")
    clin["IFTA"] = pd.to_numeric(clin.get("IFTA"), errors="coerce")

    use = [f for f in FEATS if f in df.columns and df[f].notna().sum() >= 50]
    print(f"features in ψ: {len(use)}")
    for f in use:
        print(f"  {f:28s} NA%={df[f].isna().mean()*100:5.1f}  zero%={(df[f]==0).mean()*100:5.1f}")

    X = df[use].replace([np.inf, -np.inf], np.nan).copy()
    for f in use:
        if not HIGHER[f]:
            X[f] = -X[f]
    Xi = X.fillna(0.0).to_numpy(float)  # zero impute
    Xs = np.clip(RobustScaler().fit_transform(Xi), -10, 10)
    pca = PCA(n_components=min(10, len(use)), random_state=42)
    Xp = pca.fit_transform(Xs)
    print("cum var%", (np.cumsum(pca.explained_variance_ratio_) * 100).round(1))

    meta = df.merge(clin[["sample", "eGFR_CKDEPI", "IFTA"]], on="sample", how="left")
    eg = meta["eGFR_CKDEPI"].to_numpy(float)
    ift = meta["IFTA"].to_numpy(float)

    def _pc_cov_dir(y):
        m = ~np.isnan(y)
        if m.sum() < 30:
            return np.zeros(Xp.shape[1])
        yz = (y[m] - np.nanmean(y[m])) / (np.nanstd(y[m]) + 1e-9)
        cov = np.array([np.cov(Xp[m, k], yz)[0, 1] for k in range(Xp.shape[1])])
        nrm = np.linalg.norm(cov)
        return cov / nrm if nrm > 1e-12 else cov

    dir_egfr, dir_ifta = _pc_cov_dir(-eg), _pc_cov_dir(ift)
    direction = dir_egfr + dir_ifta
    nrm = np.linalg.norm(direction)
    if nrm < 1e-12:
        direction = dir_egfr if np.linalg.norm(dir_egfr) > 0 else dir_ifta
        nrm = np.linalg.norm(direction) + 1e-12
    direction = direction / nrm
    proj = Xp @ direction
    m_e, m_i = ~np.isnan(eg), ~np.isnan(ift)
    r_e = np.corrcoef(proj[m_e], eg[m_e])[0, 1] if m_e.sum() > 30 else 0.0
    r_i = np.corrcoef(proj[m_i], ift[m_i])[0, 1] if m_i.sum() > 30 else 0.0
    if r_e > 0:
        proj, direction = -proj, -direction
        r_e, r_i = -r_e, -r_i
    if r_i < 0:
        direction = (dir_egfr + 1.5 * dir_ifta)
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        proj = Xp @ direction
        r_e = np.corrcoef(proj[m_e], eg[m_e])[0, 1]
        r_i = np.corrcoef(proj[m_i], ift[m_i])[0, 1]
        if r_e > 0:
            proj, direction = -proj, -direction
            r_e, r_i = -r_e, -r_i
    print(f"oriented r(ψ,eGFR)={r_e:.3f} r(ψ,IFTA)={r_i:.3f}")

    ranks = pd.Series(proj).rank(method="average").to_numpy()
    df["psi_inflam"] = (ranks - 1) / (len(ranks) - 1) * 100.0

    load_df = (
        pd.DataFrame(
            {
                "feature": use,
                "label": [LABELS[f] for f in use],
                "group_c": [GROUPS[f] for f in use],
                "loading": pca.components_.T @ direction,
            }
        )
        .assign(abs_loading=lambda x: x["loading"].abs())
        .sort_values("abs_loading", ascending=False)
    )
    print(load_df.head(12).to_string(index=False))

    pat = (
        df.groupby("sample", as_index=False)
        .agg(
            psi_inflam_mean=("psi_inflam", "mean"),
            psi_inflam_med=("psi_inflam", "median"),
            psi_inflam_p90=("psi_inflam", lambda x: float(np.percentile(x, 90))),
            n_patch=("psi_inflam", "count"),
        )
        .merge(clin, on="sample", how="left")
    )
    sub = pat.dropna(subset=["psi_inflam_p90", "eGFR_CKDEPI"])
    rho, pval = stats.spearmanr(sub["psi_inflam_p90"], sub["eGFR_CKDEPI"])
    sub_i = pat.dropna(subset=["psi_inflam_p90", "IFTA"])
    rho_i, p_i = stats.spearmanr(sub_i["psi_inflam_p90"], sub_i["IFTA"])
    print(f"ψ90 vs eGFR ρ={rho:.3f} p={pval:.2e}; vs IFTA ρ={rho_i:.3f} p={p_i:.2e}")

    psi = df["psi_inflam"].to_numpy(float)
    n_pts, bw = 50, 10.0
    grid = np.linspace(0, 100, n_pts)
    smooth = np.zeros((len(use), n_pts))
    for fi, feat in enumerate(use):
        vals = df[feat].to_numpy(float)
        if not HIGHER[feat]:
            vals = -vals
        ok = ~np.isnan(vals)
        psi_ok, v = psi[ok], vals[ok]
        for pi, g in enumerate(grid):
            w = np.exp(-0.5 * ((psi_ok - g) / bw) ** 2)
            ws = w.sum()
            smooth[fi, pi] = (w * v).sum() / ws if ws > 0 else np.nan
    for fi in range(len(use)):
        m_, s_ = np.nanmean(smooth[fi]), np.nanstd(smooth[fi])
        if s_ > 1e-9:
            smooth[fi] = (smooth[fi] - m_) / s_

    peak_psi = []
    for fi in range(len(use)):
        y = np.nan_to_num(smooth[fi], nan=0.0)
        w = np.exp(y - y.max())
        w = w / (w.sum() + 1e-12)
        peak_psi.append(float(np.sum(w * grid)))
    order = np.argsort(peak_psi)
    feats_s = [use[i] for i in order]
    smooth_s = smooth[order]
    print("row order:", [f"{LABELS[f]}@{peak_psi[i]:.0f}" for i, f in zip(order, feats_s)])

    df2 = df.merge(clin[["sample", "eGFR_CKDEPI"]], on="sample", how="left")
    df2["egfr_grp"] = pd.cut(df2["eGFR_CKDEPI"], [0, 30, 60, 200], labels=["<30", "30–60", ">60"])

    fig_h = max(95, 55 + 2.8 * len(use))
    fig = plt.figure(figsize=(115 * MM, fig_h * MM), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.28], hspace=0.06, left=0.34, right=0.88, top=0.93, bottom=0.09
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[1, 0], sharex=ax)
    x_edges = np.linspace(0, 100, smooth_s.shape[1] + 1)
    y_edges = np.arange(-0.5, len(feats_s) + 0.5, 1.0)
    Xg, Yg = np.meshgrid(x_edges, y_edges)
    pm = ax.pcolormesh(
        Xg, Yg, smooth_s, cmap=heat_cmap, vmin=-1.8, vmax=1.8, shading="flat", rasterized=True
    )
    ax.set_ylim(len(feats_s) - 0.5, -0.5)
    ax.set_yticks(range(len(feats_s)))
    ax.set_yticklabels([LABELS[f] for f in feats_s], fontsize=5.6)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_title(
        f"Inflam ψ ({len(use)}-feat; Ge* epi; cluster≥5 in axis; zero-impute; eGFR↓+IFTA↑)",
        fontsize=6.2,
        fontstyle="italic",
        loc="left",
    )
    ax.text(-0.42, 1.06, "c", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    cb = fig.colorbar(pm, ax=ax, shrink=0.80, pad=0.02, fraction=0.035)
    cb.set_label("z-score", fontsize=6.5)
    cb.set_ticks([-1.5, 0, 1.5])

    px = np.linspace(0, 100, 200)
    for grp, col in {">60": NB, "30–60": NG, "<30": NY}.items():
        subg = df2.loc[df2["egfr_grp"] == grp, "psi_inflam"].dropna()
        if len(subg) < 20:
            continue
        dens = gaussian_kde(subg, bw_method=0.25)(px)
        dens /= dens.max()
        ax_d.plot(px, dens, color=col, lw=1.2, label=f"eGFR {grp}")
        ax_d.fill_between(px, 0, dens, color=col, alpha=0.12)
    ax_d.set_xlabel("ψ_inflam (inflammatory state)", fontsize=7)
    ax_d.set_ylabel("Density", fontsize=6.5)
    ax_d.set_xlim(0, 100)
    ax_d.set_ylim(0, 1.15)
    ax_d.legend(frameon=False, fontsize=5.6, loc="upper left", ncol=3)

    out = FIG / f"365_psi_heatmap_Ge_cluster5_{TS}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
        (RPT / f"{out.name}.{ext}").write_bytes(Path(f"{out}.{ext}").read_bytes())
    (TEACHER / "figures" / "heatmap_full.png").write_bytes(Path(f"{out}.png").read_bytes())
    (TEACHER / "figures" / "heatmap_full.pdf").write_bytes(Path(f"{out}.pdf").read_bytes())
    (TEACHER / "327_炎症拟时序热图.png").write_bytes(Path(f"{out}.png").read_bytes())
    (TEACHER / f"365_拟时序热图_上皮内_cluster5_{TS}.png").write_bytes(Path(f"{out}.png").read_bytes())
    (TEACHER / f"365_拟时序热图_上皮内_cluster5_{TS}.pdf").write_bytes(Path(f"{out}.pdf").read_bytes())
    plt.close(fig)
    print(f"→ {out}.png")

    # also write 327-compatible psi table for forest
    keep = ["sample", "patch_x", "patch_y", "n_inflam", "n_tub", "psi_inflam"] + use
    keep = [c for c in keep if c in df.columns]
    psi_path = RES / f"327_patch_dict_psi_allfeat_{TS}.csv"
    df[keep].sort_values("psi_inflam").to_csv(psi_path, index=False)
    df[keep].sort_values("psi_inflam").to_csv(RES / f"365_patch_psi_Ge_cluster5_{TS}.csv", index=False)
    (TEACHER / "02_拟时序矩阵_patch按psi排序_含原始特征.csv").write_bytes(psi_path.read_bytes())
    pat.to_csv(RES / f"365_patient_psi_Ge_cluster5_{TS}.csv", index=False)
    load_df.to_csv(RES / f"365_loadings_Ge_cluster5_{TS}.csv", index=False)

    heat_wide = pd.DataFrame(smooth_s, index=feats_s, columns=[f"{g:.2f}" for g in grid])
    heat_wide.insert(0, "metric_id", feats_s)
    heat_wide.insert(1, "label", [LABELS[f] for f in feats_s])
    heat_wide.insert(2, "group_c", [GROUPS[f] for f in feats_s])
    heat_wide.to_csv(RES / f"365_热图显示矩阵_{len(use)}x{n_pts}_{TS}.csv", index=False)

    md = f"""# 365 拟时序热图（Ge* + 管周炎簇≥5 入轴；Hotspot≥10 不入ψ）

- 特征 n={len(use)}
- 已删：A2/B3/C1/D2；合并 G1–G4；G5/Gl* 不进主 ψ
- 入轴：`D3a_frac_cluster5`（≥5 @50µm）；Hotspot≥10 仅描述
- 填充：NaN→0
- r(ψ,eGFR)={r_e:.3f}, r(ψ,IFTA)={r_i:.3f}
- ψ90–eGFR ρ={rho:.3f}; ψ90–IFTA ρ={rho_i:.3f}
- 图：`{out.name}.png/.pdf`
"""
    (RES / f"365_psi_heatmap_Ge_cluster5_{TS}.md").write_text(md, encoding="utf-8")
    (TEACHER / f"365_拟时序热图说明_{TS}.md").write_text(md, encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
