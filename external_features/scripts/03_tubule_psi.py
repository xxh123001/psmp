#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project frozen tubule morph ψ from tubule_features_all CSV → tub_p80/p87/p90."""
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
    ap.add_argument("--features_csv", required=True, help="tubule_features_all-like CSV")
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--max_tub", type=int, default=350)
    ap.add_argument("--out_tubule", required=True)
    ap.add_argument("--out_patient", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.frozen)
    cols = bundle["feat_cols"]
    df = pd.read_csv(args.features_csv)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns ({len(missing)}): {missing[:10]}...")

    parts = []
    for s, g in df.groupby("sample"):
        if len(g) > args.max_tub:
            g = g.sample(n=args.max_tub, random_state=42)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)

    X = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    Xi = bundle["imputer"].transform(X)
    Xs = np.clip(bundle["scaler"].transform(Xi), -10, 10)
    proj = bundle["pca"].transform(Xs) @ bundle["direction"]
    df["pseudotime"] = to_psi100(proj, bundle["pct_map"]["ref_sorted"])

    keep = [c for c in ["sample", "tubule_id", "pseudotime", "W_grade", "cx_um", "cy_um"] if c in df.columns]
    if "tubule_id" not in keep and "tub_id" in df.columns:
        df["tubule_id"] = df["tub_id"]
        keep = [c for c in ["sample", "tubule_id", "pseudotime", "W_grade", "cx_um", "cy_um"] if c in df.columns]

    pat = (
        df.groupby("sample", as_index=False)
        .agg(
            n_tub=("pseudotime", "count"),
            tub_p80=("pseudotime", lambda x: pct(x, 80)),
            tub_p87=("pseudotime", lambda x: pct(x, 87)),
            tub_p90=("pseudotime", lambda x: pct(x, 90)),
            tub_mean=("pseudotime", "mean"),
        )
    )
    Path(args.out_tubule).parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(args.out_tubule, index=False)
    pat.to_csv(args.out_patient, index=False)
    print(f"tubules={len(df)} samples={len(pat)} → {args.out_patient}")


if __name__ == "__main__":
    main()
