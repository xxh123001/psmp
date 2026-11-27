#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
373 — Rebuild Inflam ψ on full 86 slides (056 tubule coords), then
macro (apparent) AUC on expanded discovery n=55.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data"))
CSV = ROOT / "009_csv"
CLIN = ROOT / "011_data" / "clinical"
RES = ROOT / "小管差异分析" / "03_results"
CODE = ROOT / "小管差异分析" / "02_code"
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
MAX_TUB_PER_SAMPLE = 350
RNG = np.random.default_rng(42)


def load_235():
    path = CODE / "235_inflam_pseudotime_ortho12.py"
    spec = importlib.util.spec_from_file_location("inflam235", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct(s, p):
    a = np.asarray(s, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


def apparent_auc(X, y):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xs, y)
    return float(roc_auc_score(y, clf.predict_proba(Xs)[:, 1]))


def loo_auc(X, y):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        try:
            clf.fit(Xtr, y[tr])
            pred[te] = clf.predict_proba(Xte)[:, 1]
        except Exception:
            pred[te] = y[tr].mean()
    return float(roc_auc_score(y, pred))


def main():
    m235 = load_235()
    print("[1] Load 056 tubules (all 86) + glom/art")
    tub = pd.read_csv(CSV / "056_20260517_155623-P2v4_instance_tubule_V1.csv")
    tub = tub.rename(columns={"tub_id": "tubule_id", "cx_um": "x_um", "cy_um": "y_um"})
    tub["W_grade"] = pd.to_numeric(tub.get("tub_W_int"), errors="coerce").fillna(0)
    tub = tub.dropna(subset=["x_um", "y_um", "tubule_id", "sample"]).copy()

    # subsample for speed / parity with morph ψ
    parts = []
    for s, g in tub.groupby("sample"):
        if len(g) > MAX_TUB_PER_SAMPLE:
            g = g.sample(n=MAX_TUB_PER_SAMPLE, random_state=42)
        parts.append(g)
    tub = pd.concat(parts, ignore_index=True)
    print(f"  tubules={len(tub):,} samples={tub['sample'].nunique()}")

    glom = pd.read_csv(CSV / "005_20260517_062600-P2_86例三轴instance_glom_V1.csv")
    art = pd.read_csv(CSV / "056_20260517_155623-P2v4_instance_artery_V1.csv")

    print("[2] Extract inflam features")
    g_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in glom.groupby("sample")}
    a_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in art.groupby("sample")}
    jobs = []
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

    # serial/thread-safe call (importlib module is not picklable for ProcessPool)
    frames = []
    for i, j in enumerate(jobs, 1):
        frames.append(m235.features_one(*j))
        if i % 10 == 0 or i == len(jobs):
            print(f"  samples {i}/{len(jobs)}")
    feat = pd.concat(frames, ignore_index=True)
    df = tub[["tubule_id", "sample", "x_um", "y_um", "W_grade"]].merge(
        feat.drop(columns=["sample"]), on="tubule_id", how="inner"
    )
    print(f"  matrix {df.shape}")

    print("[3] PCA ψ_inflam")
    clin = pd.concat(
        [
            pd.read_csv(CLIN / "V4_patient_train_56.csv"),
            pd.read_csv(CLIN / "V4_patient_valid_30.csv"),
        ],
        ignore_index=True,
    ).rename(columns={"Slide": "sample"})
    clin["eGFR_CKDEPI"] = pd.to_numeric(clin["eGFR_CKDEPI"], errors="coerce")
    clin["DCr"] = pd.to_numeric(clin["DCr"], errors="coerce")
    clin["2Y_DCr"] = pd.to_numeric(clin["2Y_DCr"], errors="coerce")

    X_raw = df[m235.INFLAM_FEATS].replace([np.inf, -np.inf], np.nan)
    X_or = X_raw.copy()
    X_or["nearest_um"] = -X_or["nearest_um"]
    Xs = RobustScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_or))
    Xs = np.clip(Xs, -10, 10)
    pca = PCA(n_components=min(8, Xs.shape[1]), random_state=42)
    Xp = pca.fit_transform(Xs)

    eg = df.merge(clin[["sample", "eGFR_CKDEPI"]], on="sample", how="left")["eGFR_CKDEPI"].values
    m = ~np.isnan(eg)
    e_z = (eg[m] - np.nanmean(eg[m])) / (np.nanstd(eg[m]) + 1e-9)
    cov = np.array([np.cov(Xp[m, k], e_z)[0, 1] for k in range(Xp.shape[1])])
    direction = cov / (np.linalg.norm(cov) + 1e-9)
    proj = Xp @ direction
    if np.corrcoef(proj[m], eg[m])[0, 1] > 0:
        proj = -proj
        print("  flipped → higher ψ = lower eGFR")
    ranks = pd.Series(proj).rank(method="average").values
    df["psi_inflam"] = (ranks - 1) / max(len(ranks) - 1, 1) * 100.0

    pat = (
        df.groupby("sample", as_index=False)
        .agg(
            n_tub=("tubule_id", "count"),
            psi_inflam_med=("psi_inflam", "median"),
            psi_inflam_mean=("psi_inflam", "mean"),
            psi_p87=("psi_inflam", lambda x: pct(x, 87)),
            psi_p95=("psi_inflam", lambda x: pct(x, 95)),
            frac_hotspot=("hotspot_ge5", "mean"),
        )
        .merge(clin, on="sample", how="left")
    )
    sub = pat.dropna(subset=["psi_p87", "eGFR_CKDEPI"])
    rho, pval = stats.spearmanr(sub["psi_p87"], sub["eGFR_CKDEPI"])
    print(f"  ψ p87 vs eGFR: ρ={rho:.3f}, p={pval:.2e}, n={len(sub)}")
    print(f"  patient samples={len(pat)}")

    tub_out = RES / f"373_tubule_inflam_expand86_{TS}.csv"
    pat_out = RES / f"373_patient_inflam_expand86_{TS}.csv"
    df.to_csv(tub_out, index=False)
    pat.to_csv(pat_out, index=False)
    print("  saved", tub_out.name, pat_out.name)

    print("[4] Macro (apparent) AUC on expanded n=55")
    morph = pd.read_csv(CSV / "034_20260517_071631-P7v2_tubule_pseudotime_V2.csv")
    tg = morph.groupby("sample")["pseudotime"].agg(
        tub_p80=lambda x: pct(x, 80),
        tub_p87=lambda x: pct(x, 87),
    ).reset_index()
    glo = pd.read_csv(CSV / "079_20260517_184308-P7v3_glom_pseudotime_V3.csv")
    gg = glo.groupby("sample")["pseudotime"].agg(
        glo_p90=lambda x: pct(x, 90),
        glo_p93=lambda x: pct(x, 93),
    ).reset_index()
    ifta = pd.read_csv(sorted(RES.glob("335_IFTA_model_vs_grade_patient_*.csv"))[-1])

    train = clin[clin["sample"].isin(pd.read_csv(CLIN / "V4_patient_train_56.csv")["Slide"])].copy()
    exp = (
        train[["sample", "DCr", "2Y_DCr"]]
        .merge(pat[["sample", "psi_p87", "psi_p95", "n_tub"]], on="sample", how="left")
        .merge(tg, on="sample", how="left")
        .merge(gg, on="sample", how="left")
        .merge(ifta[["sample", "IFTA_model"]], on="sample", how="left")
    )
    exp = exp[exp["DCr"].notna() & exp["psi_p87"].notna() & exp["tub_p87"].notna()].copy()
    print(f"  expanded inflam-complete n={len(exp)} 1y={int(exp.DCr.sum())} 2y={int(exp['2Y_DCr'].sum())}")

    models = {
        "1y": [
            ("Inflam ψ p87", ["psi_p87"]),
            ("Inflam ψ p87 + IFTA(model)", ["psi_p87", "IFTA_model"]),
            ("PT p87 + Inflam ψ p87", ["tub_p87", "psi_p87"]),
            ("PT p87 + Inflam ψ p87 + IFTA(model)", ["tub_p87", "psi_p87", "IFTA_model"]),
            ("PT p87 + Glo p90 + IFTA(model)", ["tub_p87", "glo_p90", "IFTA_model"]),
            ("IFTA(model)", ["IFTA_model"]),
        ],
        "2y": [
            ("Inflam ψ p95", ["psi_p95"]),
            ("Inflam ψ p95 + IFTA(model)", ["psi_p95", "IFTA_model"]),
            ("PT p80 + Inflam ψ p95", ["tub_p80", "psi_p95"]),
            ("PT p80 + Inflam ψ p95 + IFTA(model)", ["tub_p80", "psi_p95", "IFTA_model"]),
            ("PT p80 + Glo p93 + IFTA(model)", ["tub_p80", "glo_p93", "IFTA_model"]),
            ("IFTA(model)", ["IFTA_model"]),
        ],
    }
    ymap = {"1y": "DCr", "2y": "2Y_DCr"}
    rows = []
    for ep, ms in models.items():
        ycol = ymap[ep]
        for name, feats in ms:
            sub = exp[["sample", ycol] + feats].dropna()
            y = sub[ycol].astype(int).values
            X = sub[feats].astype(float).values
            app = apparent_auc(X, y)
            loo = loo_auc(X, y)
            rows.append(
                dict(
                    endpoint=ep,
                    model=name,
                    n=len(sub),
                    events=int(y.sum()),
                    auc_macro_apparent=app,
                    auc_loo=loo,
                )
            )
            print(f"  {ep} n={len(sub):2d} app={app:.3f} loo={loo:.3f} | {name}")

    out = pd.DataFrame(rows)
    outp = RES / f"373_expanded55_inflam_macroAUC_{TS}.csv"
    out.to_csv(outp, index=False)
    exp.to_csv(RES / f"373_expanded55_inflam_patient_{TS}.csv", index=False)
    print("Saved", outp)


if __name__ == "__main__":
    main()
