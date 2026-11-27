#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble external patient features by swapping freshly recomputed columns
into a base feature table.

IFTA + glom in the base table are already on aligned axes (validated), so by
default we KEEP them and only swap in:
  - re-extracted inflammation (psi_p87 / psi_p95) from script 12
  - (tubule P7v2 is injected later by script 14 via --tubule_p7v2_patient)

Optionally also swap IFTA / glom if you recomputed them on the server.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

PKG = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(PKG / "discovery_ref" / "patient_external_features_base.csv"))
    ap.add_argument("--inflam", default="", help="patient inflam CSV from script 12 (sample,psi_p87,psi_p95,...)")
    ap.add_argument("--glom", default="", help="optional patient glom CSV (sample,glo_p90,glo_p93,...)")
    ap.add_argument("--ifta", default="", help="optional patient IFTA CSV (sample,IFTA_model)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.base)

    def swap(df_base, path, cols):
        if not path:
            return df_base
        new = pd.read_csv(path)
        cols = [c for c in cols if c in new.columns]
        df_base = df_base.drop(columns=[c for c in cols if c in df_base.columns], errors="ignore")
        df_base = df_base.merge(new[["sample"] + cols], on="sample", how="left")
        print(f"  swapped {cols} from {Path(path).name}")
        return df_base

    base = swap(base, args.inflam, ["psi_p87", "psi_p95", "psi_inflam_med", "n_tub_inflam"])
    base = swap(base, args.glom, ["glo_p90", "glo_p93", "glo_mean", "n_glo"])
    base = swap(base, args.ifta, ["IFTA_model"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out, index=False)
    print(f"assembled n={len(base)} → {args.out}")


if __name__ == "__main__":
    main()
