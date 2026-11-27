#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project frozen glom morph ψ → glo_p90 / glo_p93."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def to_psi100(proj, ref_sorted):
    ranks = np.searchsorted(ref_sorted, proj, side="left")
    return ranks / max(len(ref_sorted) - 1, 1) * 100.0


def pct(a, p):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glom_csv", required=True, help="instance glom CSV (filtered dia>=100 preferred)")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out_glom", required=True)
    ap.add_argument("--out_patient", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.frozen)
    cols = bundle["feat_cols"]
    min_dia = float(bundle.get("min_dia_um", 100.0))
    g = pd.read_csv(args.glom_csv)
    if "glom_dia_outer_um" in g.columns:
        g = g[g["glom_dia_outer_um"] >= min_dia].copy()
    elif "dia_um" in g.columns:
        g = g[g["dia_um"] >= min_dia].copy()

    missing = [c for c in cols if c not in g.columns]
    if missing:
        raise SystemExit(f"Missing glom columns: {missing}")

    X = g[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    Xi = bundle["imputer"].transform(X)
    Xs = np.clip(bundle["scaler"].transform(Xi), -10, 10)
    proj = bundle["pca"].transform(Xs) @ bundle["direction"]
    g["pseudotime"] = to_psi100(proj, bundle["pct_map"]["ref_sorted"])

    keep = [c for c in ["sample", "glom_id", "pseudotime", "cx_um", "cy_um", "glom_dia_outer_um"] if c in g.columns]
    pat = (
        g.groupby("sample", as_index=False)
        .agg(
            n_glo=("pseudotime", "count"),
            glo_p90=("pseudotime", lambda x: pct(x, 90)),
            glo_p93=("pseudotime", lambda x: pct(x, 93)),
            glo_mean=("pseudotime", "mean"),
        )
    )
    Path(args.out_glom).parent.mkdir(parents=True, exist_ok=True)
    g[keep].to_csv(args.out_glom, index=False)
    pat.to_csv(args.out_patient, index=False)
    print(f"gloms={len(g)} samples={len(pat)} → {args.out_patient}")


if __name__ == "__main__":
    main()
