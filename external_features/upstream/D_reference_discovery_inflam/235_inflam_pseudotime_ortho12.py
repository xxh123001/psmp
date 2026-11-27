#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
235 — Inflammation pseudotime from 10 tubule-level features

No lymph/neutro/plasma subtype split (pool inflam; exclude 非炎症细胞 from counts,
but KEEP 非炎症 as denominator for proportion).

Dropped cluster_size & slide_enrich (dominated axis / dens_50 clone).

Feature panel
-------------
 1 dens_50            burden: inflam density @50 µm
 2 inflam_frac_50     proportion: inflam / (inflam + non-inflam) @50
 3 nearest_um         proximity to nearest inflam cell
 4 hotspot_ge5        aggregation flag (n@50 ≥ 5)
 5 in_cluster         membership in slide DBSCAN hotspot
 6 peri_glom_dens     inflam dens @50 µm of nearest glomerulus
 7 peri_art_dens      inflam dens @50 µm of nearest artery
 8 atrophy_inflam_f   of local inflam (@100), fraction nearer to W≥2 tubule
 9 local_disp         CV of distances of inflam cells within 100 µm
10 ring_grad          dens(0–25) − dens(50–100)
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, findfont
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data"))
ALG = ROOT / "011_data" / "raw" / "alg_results_86样本"
CSV = ROOT / "009_csv"
CLIN = ROOT / "011_data" / "clinical"
RES = ROOT / "小管差异分析" / "03_results"
FIG = ROOT / "小管差异分析" / "04_figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MM = 1 / 25.4
MPP = 0.2518

INFLAM_LABELS = {
    "淋巴细胞",
    "中性粒细胞",
    "非典型中性粒细胞",
    "浆细胞",
    "嗜酸性-典型",
    "嗜酸性-困难",
    "嗜酸性-三核",
}
DBSCAN_EPS, DBSCAN_MIN = 80.0, 8

INFLAM_FEATS = [
    "dens_50",
    "inflam_frac_50",
    "nearest_um",
    "hotspot_ge5",
    "in_cluster",
    "peri_glom_dens",
    "peri_art_dens",
    "atrophy_inflam_f",
    "local_disp",
    "ring_grad",
]
FEAT_LABELS = {
    "dens_50": "Local cell density @50µm",
    "inflam_frac_50": "Local inflam proportion @50µm",
    "nearest_um": "Closeness to nearest inflam",
    "hotspot_ge5": "Local hotspot flag (≥5 @50µm)",
    "in_cluster": "Inside slide-level aggregate",
    "peri_glom_dens": "Inflam dens at nearest glomerulus",
    "peri_art_dens": "Inflam dens at nearest artery",
    # NOT "how inflamed atrophic tubules are" — share of local cells nearer to W≥2
    "atrophy_inflam_f": "Scar-directed share of local cells",
    "local_disp": "Local spatial spread (CV)",
    "ring_grad": "Near-minus-far density contrast",
}

NB, NR, NG, NY = "#0072B5", "#BC3C29", "#20854E", "#E18727"

NEJM = "Helvetica"
try:
    if "helvetica" not in findfont(FontProperties(family=NEJM)).lower():
        NEJM = "Arial"
except Exception:
    NEJM = "Arial"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [NEJM, "Arial", "DejaVu Sans"],
        "font.size": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    }
)

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


def bbx_center_um(bbx, mpp=MPP):
    if not bbx or len(bbx) < 4:
        return None
    x, y, w, h = bbx[:4]
    return ((x + w / 2.0) * mpp, (y + h / 2.0) * mpp)


def load_cells(sample_dir: Path):
    """Return inflam_xy, noninflam_xy (µm)."""
    jdir = sample_dir / "hr0_alg.json"
    c_path, cc_path = jdir / "KI1_C.json", jdir / "KI1_CC.json"
    if not c_path.exists() or not cc_path.exists():
        return np.zeros((0, 2)), np.zeros((0, 2))
    cells = {c["id"]: c for c in json.load(open(c_path, encoding="utf-8"))}
    ccs = json.load(open(cc_path, encoding="utf-8"))
    inflam, non = [], []
    for cc in ccs:
        lab = cc.get("label", "")
        parent = cells.get(cc.get("kId", ""))
        if parent is None:
            continue
        ctr = bbx_center_um(parent.get("bbx"))
        if ctr is None:
            continue
        if lab == "非炎症细胞":
            non.append(ctr)
        elif lab in INFLAM_LABELS:
            inflam.append(ctr)
    return (
        np.asarray(inflam, float) if inflam else np.zeros((0, 2)),
        np.asarray(non, float) if non else np.zeros((0, 2)),
    )


def features_one(
    sample: str,
    tub_xy: np.ndarray,
    tub_ids: np.ndarray,
    tub_W: np.ndarray,
    glom_xy: np.ndarray,
    art_xy: np.ndarray,
):
    inflam, non = load_cells(ALG / f"{sample}_alg_result")
    n = len(tub_xy)
    out = {k: np.zeros(n, float) for k in INFLAM_FEATS}
    out["tubule_id"] = tub_ids
    out["sample"] = np.array([sample] * n)
    out["n_inflam_slide"] = np.full(n, len(inflam))
    out["n_non_slide"] = np.full(n, len(non))
    if n == 0:
        return pd.DataFrame(out)

    A25, A50, A100 = np.pi * 25**2, np.pi * 50**2, np.pi * 100**2
    A_far = A100 - A50

    # defaults when no inflam
    out["nearest_um"][:] = 250.0
    out["local_disp"][:] = 0.0

    if len(inflam) == 0:
        return pd.DataFrame(out)

    tree_i = cKDTree(inflam)
    tree_n = cKDTree(non) if len(non) else None

    n25 = np.asarray(tree_i.query_ball_point(tub_xy, 25, return_length=True), float)
    n50 = np.asarray(tree_i.query_ball_point(tub_xy, 50, return_length=True), float)
    n100 = np.asarray(tree_i.query_ball_point(tub_xy, 100, return_length=True), float)
    dens25 = n25 / A25
    dens50 = n50 / A50
    dens_far = (n100 - n50).clip(min=0) / A_far

    if tree_n is not None:
        non50 = np.asarray(tree_n.query_ball_point(tub_xy, 50, return_length=True), float)
    else:
        non50 = np.zeros(n)

    nearest, _ = tree_i.query(tub_xy, k=1)
    nearest = np.asarray(nearest, float)

    # DBSCAN clusters on slide
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(inflam)
    _, nn_idx = tree_i.query(tub_xy, k=1)
    nn_idx = np.asarray(nn_idx, int)
    in_cl = np.zeros(n)
    for i, j in enumerate(nn_idx):
        L = int(lab[j])
        if L >= 0 and nearest[i] <= DBSCAN_EPS:
            in_cl[i] = 1.0

    # peri-glom / peri-artery dens at nearest structure
    peri_g = np.zeros(n)
    peri_a = np.zeros(n)
    if len(glom_xy):
        tg = cKDTree(glom_xy)
        _, gi = tg.query(tub_xy, k=1)
        gi = np.atleast_1d(np.asarray(gi, int))
        g_n50 = np.asarray(
            tree_i.query_ball_point(glom_xy, 50, return_length=True), float
        )
        peri_g = (g_n50 / A50)[gi]
    if len(art_xy):
        ta = cKDTree(art_xy)
        _, ai = ta.query(tub_xy, k=1)
        ai = np.atleast_1d(np.asarray(ai, int))
        a_n50 = np.asarray(
            tree_i.query_ball_point(art_xy, 50, return_length=True), float
        )
        peri_a = (a_n50 / A50)[ai]

    # atrophy colocalization: among inflam in 100 µm of tubule,
    # fraction whose nearest tubule has W>=2
    tree_tub = cKDTree(tub_xy)
    _, i_all = tree_tub.query(inflam, k=1)
    inflam_near_atrophy = (tub_W[np.asarray(i_all, int)] >= 2).astype(float)
    atrophy_f = np.zeros(n)
    balls = tree_i.query_ball_point(tub_xy, 100)
    for i, idxs in enumerate(balls):
        if not idxs:
            atrophy_f[i] = 0.0
        else:
            atrophy_f[i] = float(inflam_near_atrophy[idxs].mean())

    # local dispersion: CV of distances of inflam within 100 µm
    local_disp = np.zeros(n)
    for i, idxs in enumerate(balls):
        if len(idxs) < 3:
            local_disp[i] = 0.0
            continue
        d = np.linalg.norm(inflam[idxs] - tub_xy[i], axis=1)
        mu = d.mean()
        local_disp[i] = (d.std() / mu) if mu > 1e-6 else 0.0

    out["dens_50"] = dens50
    out["inflam_frac_50"] = n50 / (n50 + non50 + 1e-9)
    out["nearest_um"] = nearest
    out["hotspot_ge5"] = (n50 >= 5).astype(float)
    out["in_cluster"] = in_cl
    out["peri_glom_dens"] = peri_g
    out["peri_art_dens"] = peri_a
    out["atrophy_inflam_f"] = atrophy_f
    out["local_disp"] = local_disp
    out["ring_grad"] = dens25 - dens_far
    return pd.DataFrame(out)


def _worker(args):
    return features_one(*args)


def loo_auc(X, y, n_boot=600, seed=0):
    oof = np.zeros(len(y), float)
    for tr, te in LeaveOneOut().split(X):
        pipe = Pipeline(
            [("s", StandardScaler()), ("m", LogisticRegression(max_iter=3000, C=1.0))]
        )
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    rng = np.random.default_rng(seed)
    bs = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        bs.append(roc_auc_score(y[b], oof[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return auc, float(lo), float(hi)


def main():
    print("=" * 80)
    print("[1] Load structure tables")
    print("=" * 80)
    tub = pd.read_csv(CSV / "087_tubule_pseudotime_with_coords_20260518_171611.csv")
    tub = tub.dropna(subset=["x_um", "y_um", "tubule_id", "sample"]).copy()
    tub["W_grade"] = pd.to_numeric(tub["W_grade"], errors="coerce").fillna(0)
    glom = pd.read_csv(CSV / "005_20260517_062600-P2_86例三轴instance_glom_V1.csv")
    art = pd.read_csv(CSV / "056_20260517_155623-P2v4_instance_artery_V1.csv")
    clin = pd.concat(
        [
            pd.read_csv(CLIN / "V4_patient_train_56.csv"),
            pd.read_csv(CLIN / "V4_patient_valid_30.csv"),
        ],
        ignore_index=True,
    ).rename(columns={"Slide": "sample"})
    for c in ["eGFR_CKDEPI", "DCr", "IFTA"]:
        if c in clin.columns:
            clin[c] = pd.to_numeric(clin[c], errors="coerce")
    dual = pd.read_csv(sorted(RES.glob("21_patient_tubglom_*.csv"))[-1]).rename(
        columns={"Slide": "sample"}
    )

    print(f"  tubules={len(tub):,} samples={tub['sample'].nunique()}")

    print("\n" + "=" * 80)
    print("[2] Extract orthogonal inflam features")
    print("=" * 80)
    jobs = []
    g_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in glom.groupby("sample")}
    a_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in art.groupby("sample")}
    for sample, g in tub.groupby("sample"):
        jobs.append(
            (
                sample,
                g[["x_um", "y_um"]].to_numpy(float),
                g["tubule_id"].to_numpy(),
                g["W_grade"].to_numpy(float),
                g_by.get(sample, np.zeros((0, 2))),
                a_by.get(sample, np.zeros((0, 2))),
            )
        )

    frames = []
    with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            frames.append(fut.result())
            if i % 10 == 0 or i == len(jobs):
                print(f"  samples {i}/{len(jobs)}")

    feat = pd.concat(frames, ignore_index=True)
    df = tub[["tubule_id", "sample", "pseudotime", "x_um", "y_um", "W_grade"]].merge(
        feat.drop(columns=["sample"]), on="tubule_id", how="inner"
    )
    print(f"  matrix {df.shape}")

    # redundancy check
    c = df[INFLAM_FEATS].corr(method="spearman")
    print("  |ρ|≥0.80 pairs:")
    for i, a in enumerate(INFLAM_FEATS):
        for b in INFLAM_FEATS[i + 1 :]:
            if abs(c.loc[a, b]) >= 0.80:
                print(f"    {a} vs {b}: {c.loc[a, b]:+.2f}")

    print("\n" + "=" * 80)
    print("[3] PCA ψ_inflam")
    print("=" * 80)
    X_raw = df[INFLAM_FEATS].replace([np.inf, -np.inf], np.nan)
    X_or = X_raw.copy()
    X_or["nearest_um"] = -X_or["nearest_um"]  # closer = more severe

    Xs = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_or))
    Xs = np.clip(Xs, -10, 10)
    pca = PCA(n_components=min(8, Xs.shape[1]), random_state=42)
    Xp = pca.fit_transform(Xs)
    print(f"  cum var%: {(np.cumsum(pca.explained_variance_ratio_)*100).round(1)}")

    eg = df.merge(clin[["sample", "eGFR_CKDEPI"]], on="sample", how="left")["eGFR_CKDEPI"].values
    m = ~np.isnan(eg)
    e_z = (eg[m] - np.nanmean(eg[m])) / (np.nanstd(eg[m]) + 1e-9)
    cov = np.array([np.cov(Xp[m, k], e_z)[0, 1] for k in range(Xp.shape[1])])
    direction = cov / (np.linalg.norm(cov) + 1e-9)
    proj = Xp @ direction
    if np.corrcoef(proj[m], eg[m])[0, 1] > 0:
        proj = -proj
        direction = -direction
        print("  flipped → higher ψ = lower eGFR")

    ranks = pd.Series(proj).rank(method="average").values
    df["psi_inflam"] = (ranks - 1) / (len(ranks) - 1) * 100.0
    df["psi_morph"] = df["pseudotime"]

    load_df = (
        pd.DataFrame({"feature": INFLAM_FEATS, "loading": pca.components_.T @ direction})
        .assign(abs_loading=lambda x: x["loading"].abs())
        .sort_values("abs_loading", ascending=False)
    )
    print(load_df.to_string(index=False))

    print("  UMAP...")
    emb = umap.UMAP(
        n_neighbors=30, min_dist=0.25, n_components=2, random_state=42, n_jobs=1
    ).fit_transform(Xp)
    df["inflam_umap1"], df["inflam_umap2"] = emb[:, 0], emb[:, 1]

    pat = (
        df.groupby("sample", as_index=False)
        .agg(
            psi_inflam_med=("psi_inflam", "median"),
            psi_inflam_p90=("psi_inflam", lambda x: np.percentile(x, 90)),
            psi_inflam_mean=("psi_inflam", "mean"),
            frac_hotspot=("hotspot_ge5", "mean"),
            frac_cluster=("in_cluster", "mean"),
            mean_inflam_frac=("inflam_frac_50", "mean"),
            n_tub=("tubule_id", "count"),
        )
        .merge(clin, on="sample", how="left")
        .merge(dual[["sample", "tub_p90", "glo_p90"]].drop_duplicates("sample"), on="sample", how="left")
    )
    sub = pat.dropna(subset=["psi_inflam_p90", "eGFR_CKDEPI"])
    rho, pval = stats.spearmanr(sub["psi_inflam_p90"], sub["eGFR_CKDEPI"])
    rho, pval = float(rho), float(pval)
    print(f"  ψ_inflam p90 vs eGFR: ρ={rho:.3f}, p={pval:.2e}, n={len(sub)}")
    rs = [
        stats.spearmanr(g["psi_morph"], g["psi_inflam"])[0]
        for _, g in df.groupby("sample")
        if len(g) >= 20
    ]
    print(f"  within-biopsy r(morph,inflam) median={np.nanmedian(rs):.3f}")

    print("\n" + "=" * 80)
    print("[4] LOO 1y DCr")
    print("=" * 80)
    auc_rows = []
    for name, cols in [
        ("Inflam ψ90", ["psi_inflam_p90"]),
        ("Inflam frac mean", ["mean_inflam_frac"]),
        ("DUAL morph", ["tub_p90", "glo_p90"]),
        ("DUAL + Inflam ψ90", ["tub_p90", "glo_p90", "psi_inflam_p90"]),
    ]:
        d = pat.dropna(subset=["DCr"] + cols)
        if len(d) < 20 or d["DCr"].nunique() < 2:
            continue
        y = d["DCr"].astype(int).values
        auc, lo, hi = loo_auc(d[cols].to_numpy(float), y)
        print(f"  {name:22s} AUC={auc:.3f} [{lo:.3f},{hi:.3f}] n={len(y)}")
        auc_rows.append({"model": name, "auc": auc, "lo": lo, "hi": hi, "n": len(y)})

    tub_out = RES / f"235_tubule_inflam_ortho10_{TS}.csv"
    pat_out = RES / f"235_patient_inflam_ortho10_{TS}.csv"
    load_out = RES / f"235_inflam_ortho10_loadings_{TS}.csv"
    df.to_csv(tub_out, index=False)
    pat.to_csv(pat_out, index=False)
    load_df.to_csv(load_out, index=False)
    if auc_rows:
        pd.DataFrame(auc_rows).to_csv(RES / f"235_inflam_ortho10_auc_{TS}.csv", index=False)
    print(f"  → {tub_out.name}")

    # =====================================================================
    # Fig5-c heatmap
    # =====================================================================
    print("\n[5] Heatmap + overview figure")
    psi = df["psi_inflam"].to_numpy(float)
    n_pts, bw = 50, 10.0
    grid = np.linspace(0, 100, n_pts)
    smooth = np.zeros((len(INFLAM_FEATS), n_pts))
    for fi, feat in enumerate(INFLAM_FEATS):
        vals = df[feat].to_numpy(float)
        if feat == "nearest_um":
            vals = -vals
        ok = ~np.isnan(vals)
        psi_ok, v = psi[ok], vals[ok]
        for pi, g in enumerate(grid):
            w = np.exp(-0.5 * ((psi_ok - g) / bw) ** 2)
            ws = w.sum()
            smooth[fi, pi] = (w * v).sum() / ws if ws > 0 else np.nan
    for fi in range(len(INFLAM_FEATS)):
        m_, s_ = np.nanmean(smooth[fi]), np.nanstd(smooth[fi])
        if s_ > 1e-9:
            smooth[fi] = (smooth[fi] - m_) / s_
    order = np.argsort([np.corrcoef(smooth[fi], grid)[0, 1] for fi in range(len(INFLAM_FEATS))])
    feats_s = [INFLAM_FEATS[i] for i in order]
    smooth_s = smooth[order]

    df2 = df.merge(clin[["sample", "eGFR_CKDEPI"]], on="sample", how="left")
    df2["egfr_grp"] = pd.cut(df2["eGFR_CKDEPI"], [0, 30, 60, 200], labels=["<30", "30–60", ">60"])

    fig = plt.figure(figsize=(110 * MM, 90 * MM), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.30], hspace=0.08, left=0.32, right=0.88, top=0.90, bottom=0.10
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
    ax.set_yticklabels([FEAT_LABELS[f] for f in feats_s], fontsize=6.0)
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_title(
        "Inflam features along ψ_inflam (10-feat, no cluster_size/enrich)",
        fontsize=7.2,
        fontstyle="italic",
        loc="left",
    )
    ax.text(-0.40, 1.08, "c", transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    cb = fig.colorbar(pm, ax=ax, shrink=0.85, pad=0.02, fraction=0.035)
    cb.set_label("z-score", fontsize=6.5)
    cb.set_ticks([-1.5, 0, 1.5])

    from scipy.stats import gaussian_kde

    colors = {">60": NB, "30–60": NG, "<30": NY}
    px = np.linspace(0, 100, 200)
    for grp, col in colors.items():
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
    ax_d.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=3)

    fig_path = FIG / f"235_inflam_heatmap_ortho10_{TS}.png"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fig_path.name}")

    # overview 2x2
    fig2, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    axA, axB, axC, axD = axes.ravel()
    sc = axA.scatter(
        df["inflam_umap1"],
        df["inflam_umap2"],
        c=df["psi_inflam"],
        cmap="viridis",
        s=3,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    fig2.colorbar(sc, ax=axA, fraction=0.046, pad=0.02).set_label("ψ_inflam")
    axA.set_title("A  UMAP", loc="left", fontweight="bold")
    axA.set_xlabel("UMAP-1")
    axA.set_ylabel("UMAP-2")

    ld = load_df.sort_values("loading")
    axB.barh(
        range(len(ld)),
        ld["loading"],
        color=[NR if v > 0 else NB for v in ld["loading"]],
        height=0.7,
    )
    axB.set_yticks(range(len(ld)))
    axB.set_yticklabels([FEAT_LABELS[f] for f in ld["feature"]], fontsize=7)
    axB.axvline(0, color="k", lw=0.6)
    axB.set_title("B  Loadings", loc="left", fontweight="bold")
    axB.set_xlabel("Loading on ψ_inflam")

    # spearman matrix of feats
    cm = df[INFLAM_FEATS].corr(method="spearman").values
    im = axC.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axC.set_xticks(range(len(INFLAM_FEATS)))
    axC.set_yticks(range(len(INFLAM_FEATS)))
    axC.set_xticklabels([FEAT_LABELS[f] for f in INFLAM_FEATS], rotation=90, fontsize=5.5)
    axC.set_yticklabels([FEAT_LABELS[f] for f in INFLAM_FEATS], fontsize=5.5)
    axC.set_title(
        f"C  Feature Spearman ({len(INFLAM_FEATS)} feats, no cluster_size/enrich)",
        loc="left",
        fontweight="bold",
    )
    fig2.colorbar(im, ax=axC, fraction=0.046, pad=0.02)

    subp = pat.dropna(subset=["psi_inflam_p90", "eGFR_CKDEPI"])
    axD.scatter(
        subp["psi_inflam_p90"], subp["eGFR_CKDEPI"], s=36, c=NY, edgecolor="k", lw=0.35, alpha=0.85
    )
    z = np.polyfit(subp["psi_inflam_p90"], subp["eGFR_CKDEPI"], 1)
    xx = np.linspace(subp["psi_inflam_p90"].min(), subp["psi_inflam_p90"].max(), 50)
    axD.plot(xx, np.polyval(z, xx), color=NR, lw=1.4)
    axD.text(
        0.05,
        0.05,
            f"ρ={rho:.2f}\np={pval:.1e}\nn={len(subp)}",
            transform=axD.transAxes,
            fontsize=7,
            va="bottom",
        )
    axD.set_xlabel("Patient ψ_inflam p90")
    axD.set_ylabel("eGFR")
    axD.set_title("D  Patient ψ vs eGFR", loc="left", fontweight="bold")
    fig2.suptitle(
        "Inflammation pseudotime — 10 features (drop cluster_size & slide_enrich)",
        fontsize=11,
        fontweight="bold",
    )
    fig2.tight_layout()
    fig2_path = FIG / f"235_inflam_ortho10_overview_{TS}.png"
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"  → {fig2_path.name}")
    print("DONE.")


if __name__ == "__main__":
    main()
