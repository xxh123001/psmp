#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge IFTA / tubule / glom / inflam patient tables (+ optional outcome CSV)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifta", required=True)
    ap.add_argument("--tubule", required=True)
    ap.add_argument("--glom", required=True)
    ap.add_argument("--inflam", required=True)
    ap.add_argument("--outcome", default=None, help="optional CSV with sample + endpoints")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.ifta)
    for path in [args.tubule, args.glom, args.inflam]:
        t = pd.read_csv(path)
        df = df.merge(t, on="sample", how="outer")
    if args.outcome:
        oc = pd.read_csv(args.outcome)
        # allow Biopsy_ID as sample
        if "sample" not in oc.columns and "Biopsy_ID" in oc.columns:
            oc = oc.rename(columns={"Biopsy_ID": "sample"})
        if "sample" not in oc.columns and "Slide" in oc.columns:
            oc = oc.rename(columns={"Slide": "sample"})
        df = df.merge(oc, on="sample", how="left")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"n={len(df)} cols={len(df.columns)} → {args.out}")
    need = ["IFTA_model", "tub_p87", "glo_p90", "psi_p87"]
    for c in need:
        if c in df.columns:
            print(f"  {c}: non-null {df[c].notna().sum()}/{len(df)}")


if __name__ == "__main__":
    main()
