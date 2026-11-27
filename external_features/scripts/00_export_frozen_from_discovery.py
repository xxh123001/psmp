#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit & export frozen projectors from discovery cohort (run once on laptop)."""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from lib.inflam_feats import INFLAM_FEATS, features_one  # noqa: E402

# --- discovery roots (edit if needed) ---
ROOT = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data"))
CSV = ROOT / "009_csv"
CLIN = ROOT / "011_data" / "clinical"
ALG = ROOT / "011_data" / "raw" / "alg_results_86样本"
OUT = PKG / "frozen"
OUT.mkdir(parents=True, exist_ok=True)

META_TUB = {
    "sample",
    "tubule_id",
    "W_grade",
    "tubule_type",
    "lumen_level",
    "brush_border",
    "epithelial_damage",
    "TBM_class",
    "cx_um",
    "cy_um",
    "is_cortex",
}
GLOM_FEATS = [
    "glom_area_um2",
    "tuft_area_um2",
    "tuft_area_fraction",
    "bowman_area_um2",
    "perimeter_um",
    "circularity",
    "solidity",
    "eccentricity",
    "elongation",
    "glom_dia_outer_um",
    "tuft_dia_inner_um",
    "major_axis_um",
    "minor_axis_um",
    "mesangial_W_int",
    "n_GBM_patches",
    "n_FSGS_patches",
    "has_global_sclerosis",
    "has_segmental_sclerosis_FSGS",
    "has_crescent_strict",
    "has_necrosis",
    "n_intraglom_inflam_cells",
    "dist_nearest_glom_um",
    "has_GBM_thicken",
    "has_broken_capsule",
    "has_bowman_caps",
]


def pct_map_fit(proj: np.ndarray):
    """Store sorted reference projections for empirical CDF → 0–100."""
    ref = np.sort(proj[np.isfinite(proj)])
    return {"ref_sorted": ref}


def fit_axis(X: np.ndarray, egfr: np.ndarray, n_comp=8):
    """RobustScaler + PCA + eGFR-covariance direction. Higher score = lower eGFR."""
    imp = SimpleImputer(strategy="median")
    Xi = imp.fit_transform(X)
    sc = RobustScaler()
    Xs = np.clip(sc.fit_transform(Xi), -10, 10)
    pca = PCA(n_components=min(n_comp, Xs.shape[1]), random_state=42)
    Xp = pca.fit_transform(Xs)
    m = np.isfinite(egfr)
    e_z = (egfr[m] - np.nanmean(egfr[m])) / (np.nanstd(egfr[m]) + 1e-9)
    cov = np.array([np.cov(Xp[m, k], e_z)[0, 1] for k in range(Xp.shape[1])])
    direction = cov / (np.linalg.norm(cov) + 1e-9)
    proj = Xp @ direction
    if np.corrcoef(proj[m], egfr[m])[0, 1] > 0:
        direction = -direction
        proj = -proj
    return {
        "imputer": imp,
        "scaler": sc,
        "pca": pca,
        "direction": direction,
        "pct_map": pct_map_fit(proj),
    }, proj


def project(bundle, X):
    Xi = bundle["imputer"].transform(X)
    Xs = np.clip(bundle["scaler"].transform(Xi), -10, 10)
    Xp = bundle["pca"].transform(Xs)
    return Xp @ bundle["direction"]


def to_psi100(proj, ref_sorted):
    # empirical CDF vs discovery reference
    ranks = np.searchsorted(ref_sorted, proj, side="left")
    return ranks / max(len(ref_sorted) - 1, 1) * 100.0


def export_inflam():
    print("=== Inflam frozen ===")
    tub = pd.read_csv(CSV / "056_20260517_155623-P2v4_instance_tubule_V1.csv")
    tub = tub.rename(columns={"tub_id": "tubule_id", "cx_um": "x_um", "cy_um": "y_um"})
    tub["W_grade"] = pd.to_numeric(tub["tub_W_int"], errors="coerce").fillna(0)
    parts = []
    for s, g in tub.groupby("sample"):
        if len(g) > 350:
            g = g.sample(n=350, random_state=42)
        parts.append(g)
    tub = pd.concat(parts, ignore_index=True)
    glom = pd.read_csv(CSV / "005_20260517_062600-P2_86例三轴instance_glom_V1.csv")
    art = pd.read_csv(CSV / "056_20260517_155623-P2v4_instance_artery_V1.csv")
    g_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in glom.groupby("sample")}
    a_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in art.groupby("sample")}
    frames = []
    for sample, g in tub.groupby("sample"):
        alg = ALG / f"{sample}_alg_result"
        frames.append(
            features_one(
                sample,
                g[["x_um", "y_um"]].to_numpy(float),
                g["tubule_id"].to_numpy(),
                g["W_grade"].to_numpy(float),
                g_by.get(sample, np.zeros((0, 2))),
                a_by.get(sample, np.zeros((0, 2))),
                alg,
            )
        )
    feat = pd.concat(frames, ignore_index=True)
    df = tub[["tubule_id", "sample"]].merge(feat, on=["tubule_id", "sample"], how="inner")
    clin = pd.concat(
        [pd.read_csv(CLIN / "V4_patient_train_56.csv"), pd.read_csv(CLIN / "V4_patient_valid_30.csv")],
        ignore_index=True,
    ).rename(columns={"Slide": "sample"})
    eg = df.merge(clin[["sample", "eGFR_CKDEPI"]], on="sample", how="left")["eGFR_CKDEPI"].astype(float).values
    X = df[INFLAM_FEATS].replace([np.inf, -np.inf], np.nan).copy()
    X["nearest_um"] = -X["nearest_um"]
    bundle, proj = fit_axis(X.to_numpy(float), eg, n_comp=8)
    bundle["feat_cols"] = INFLAM_FEATS
    bundle["flip_nearest"] = True
    joblib.dump(bundle, OUT / "inflam_psi_frozen.joblib")
    meta = {"n_tubules": int(len(df)), "n_samples": int(df["sample"].nunique()), "feats": INFLAM_FEATS}
    json.dump(meta, open(OUT / "inflam_psi_meta.json", "w"), ensure_ascii=False, indent=2)
    print("  saved inflam_psi_frozen.joblib", meta)


def export_tubule():
    print("=== Tubule morph frozen (Python portable axis) ===")
    feat_csv = sorted(CSV.glob("tubule_features_all_*.csv"))[-1]
    df = pd.read_csv(feat_csv)
    feat_cols = [c for c in df.columns if c not in META_TUB]
    na = df[feat_cols].isna().mean()
    feat_cols = [
        c
        for c in feat_cols
        if na[c] < 0.05 and not c.startswith("fourier_") and not c.endswith(tuple(f"_{i}" for i in range(2, 11)))
    ]
    # subsample 350
    parts = []
    for s, g in df.groupby("sample"):
        if len(g) > 350:
            g = g.sample(n=350, random_state=42)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    clin = pd.concat(
        [pd.read_csv(CLIN / "V4_patient_train_56.csv"), pd.read_csv(CLIN / "V4_patient_valid_30.csv")],
        ignore_index=True,
    ).rename(columns={"Slide": "sample"})
    eg = df.merge(clin[["sample", "eGFR_CKDEPI"]], on="sample", how="left")["eGFR_CKDEPI"].astype(float).values
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    bundle, _ = fit_axis(X, eg, n_comp=30)
    bundle["feat_cols"] = feat_cols
    joblib.dump(bundle, OUT / "tubule_psi_frozen.joblib")
    json.dump(
        {"n": int(len(df)), "n_samples": int(df["sample"].nunique()), "n_feats": len(feat_cols), "source": feat_csv.name},
        open(OUT / "tubule_psi_meta.json", "w"),
        indent=2,
    )
    print("  saved tubule_psi_frozen.joblib feats", len(feat_cols))


def export_glom():
    print("=== Glom morph frozen (Python portable axis) ===")
    g = pd.read_csv(CSV / "078_20260517_183945-P2v4f_instance_glom_filtered100um_V1.csv")
    cols = [c for c in GLOM_FEATS if c in g.columns]
    clin = pd.concat(
        [pd.read_csv(CLIN / "V4_patient_train_56.csv"), pd.read_csv(CLIN / "V4_patient_valid_30.csv")],
        ignore_index=True,
    ).rename(columns={"Slide": "sample"})
    eg = g.merge(clin[["sample", "eGFR_CKDEPI"]], on="sample", how="left")["eGFR_CKDEPI"].astype(float).values
    X = g[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    bundle, _ = fit_axis(X, eg, n_comp=min(15, len(cols)))
    bundle["feat_cols"] = cols
    bundle["min_dia_um"] = 100.0
    joblib.dump(bundle, OUT / "glom_psi_frozen.joblib")
    json.dump({"n": int(len(g)), "n_samples": int(g["sample"].nunique()), "feats": cols}, open(OUT / "glom_psi_meta.json", "w"), indent=2)
    print("  saved glom_psi_frozen.joblib feats", len(cols))


def export_ifta_formula():
    meta = {
        "formula": "IFTA_model = 100 * area(KI1_SC2) / (area(KI1_SC3)+area(KI1_SC4))",
        "json": "hr0_alg.json/KI1_SC.json",
        "labels": ["KI1_SC2", "KI1_SC3", "KI1_SC4"],
    }
    json.dump(meta, open(OUT / "ifta_model_formula.json", "w"), indent=2)
    print("  saved ifta_model_formula.json")


if __name__ == "__main__":
    export_ifta_formula()
    export_inflam()
    export_tubule()
    export_glom()
    print("DONE →", OUT)
