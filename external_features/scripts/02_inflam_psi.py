#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract inflam features + project frozen Inflam ψ → patient psi_p87 / psi_p95."""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from lib.inflam_feats import INFLAM_FEATS, features_one  # noqa: E402


def to_psi100(proj, ref_sorted):
    ranks = np.searchsorted(ref_sorted, proj, side="left")
    return ranks / max(len(ref_sorted) - 1, 1) * 100.0


def pct(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg_root", required=True)
    ap.add_argument("--tubule_csv", required=True, help="instance tubule CSV (sample,tub_id/tubule_id,cx_um,cy_um,tub_W_int)")
    ap.add_argument("--glom_csv", required=True)
    ap.add_argument("--artery_csv", required=True)
    ap.add_argument("--frozen", default=str(PKG / "frozen" / "inflam_psi_frozen.joblib"))
    ap.add_argument("--max_tub", type=int, default=350)
    ap.add_argument("--out_tubule", required=True)
    ap.add_argument("--out_patient", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.frozen)
    tub = pd.read_csv(args.tubule_csv)
    rename = {}
    if "tub_id" in tub.columns and "tubule_id" not in tub.columns:
        rename["tub_id"] = "tubule_id"
    if "cx_um" in tub.columns:
        rename["cx_um"] = "x_um"
        rename["cy_um"] = "y_um"
    tub = tub.rename(columns=rename)
    if "W_grade" not in tub.columns:
        tub["W_grade"] = pd.to_numeric(tub.get("tub_W_int"), errors="coerce").fillna(0)
    parts = []
    for s, g in tub.groupby("sample"):
        if len(g) > args.max_tub:
            g = g.sample(n=args.max_tub, random_state=42)
        parts.append(g)
    tub = pd.concat(parts, ignore_index=True)

    glom = pd.read_csv(args.glom_csv)
    art = pd.read_csv(args.artery_csv)
    g_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in glom.groupby("sample")}
    a_by = {s: g[["cx_um", "cy_um"]].to_numpy(float) for s, g in art.groupby("sample")}
    alg_root = Path(args.alg_root)

    frames = []
    for sample, g in tub.groupby("sample"):
        alg = alg_root / f"{sample}_alg_result"
        if not alg.exists():
            # allow sample folder without suffix
            alg = alg_root / sample
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
    df = tub[["tubule_id", "sample", "x_um", "y_um", "W_grade"]].merge(
        feat, on=["tubule_id", "sample"], how="inner"
    )

    X = df[INFLAM_FEATS].replace([np.inf, -np.inf], np.nan).copy()
    if bundle.get("flip_nearest", True):
        X["nearest_um"] = -X["nearest_um"]
    Xi = bundle["imputer"].transform(X.to_numpy(float))
    Xs = np.clip(bundle["scaler"].transform(Xi), -10, 10)
    proj = bundle["pca"].transform(Xs) @ bundle["direction"]
    df["psi_inflam"] = to_psi100(proj, bundle["pct_map"]["ref_sorted"])

    pat = (
        df.groupby("sample", as_index=False)
        .agg(
            n_tub_inflam=("tubule_id", "count"),
            psi_inflam_med=("psi_inflam", "median"),
            psi_p87=("psi_inflam", lambda x: pct(x, 87)),
            psi_p95=("psi_inflam", lambda x: pct(x, 95)),
            frac_hotspot=("hotspot_ge5", "mean"),
        )
    )
    Path(args.out_tubule).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tubule, index=False)
    pat.to_csv(args.out_patient, index=False)
    print(f"tubules={len(df)} samples={len(pat)} → {args.out_patient}")


if __name__ == "__main__":
    main()
