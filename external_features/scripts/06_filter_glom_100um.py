#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter glom instance CSV: keep glom_dia_outer_um (or equivalent) >= 100 µm."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="Raw instance_glom CSV from 057")
    ap.add_argument("--out_csv", required=True, help="Filtered output CSV")
    ap.add_argument("--min_dia_um", type=float, default=100.0)
    args = ap.parse_args()

    g = pd.read_csv(args.in_csv)
    if "glom_dia_outer_um" in g.columns:
        dia = pd.to_numeric(g["glom_dia_outer_um"], errors="coerce")
    elif "dia_um" in g.columns:
        dia = pd.to_numeric(g["dia_um"], errors="coerce")
    elif "glom_area_um2" in g.columns:
        dia = 2 * np.sqrt(pd.to_numeric(g["glom_area_um2"], errors="coerce").clip(lower=1) / np.pi)
        g = g.copy()
        g["dia_um"] = dia
    else:
        raise SystemExit("Need glom_dia_outer_um or dia_um or glom_area_um2")

    out = g[dia >= args.min_dia_um].copy()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"in={len(g)} out={len(out)} min_dia={args.min_dia_um} → {args.out_csv}")


if __name__ == "__main__":
    main()
