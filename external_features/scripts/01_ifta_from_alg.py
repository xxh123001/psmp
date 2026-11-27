#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IFTA(model) = 100 × KI1_SC2 / (KI1_SC3 + KI1_SC4) from alg_results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def ifta_one(alg_dir: Path) -> float:
    hr = alg_dir / "hr0_alg.json"
    sc_path = (hr if hr.is_dir() else alg_dir) / "KI1_SC.json"
    if not sc_path.exists():
        return np.nan
    areas = {}
    for obj in json.load(open(sc_path)):
        if not isinstance(obj, dict):
            continue
        lab = obj.get("label")
        if lab in ("KI1_SC2", "KI1_SC3", "KI1_SC4"):
            areas[lab] = areas.get(lab, 0.0) + float(obj.get("area") or 0.0)
    den = areas.get("KI1_SC3", 0.0) + areas.get("KI1_SC4", 0.0)
    if den <= 0:
        return np.nan
    return 100.0 * areas.get("KI1_SC2", 0.0) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg_root", required=True, help="Directory containing <sample>_alg_result/")
    ap.add_argument("--out", required=True, help="Output patient CSV")
    args = ap.parse_args()
    root = Path(args.alg_root)
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        sample = d.name.replace("_alg_result", "")
        rows.append({"sample": sample, "IFTA_model": ifta_one(d)})
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"n={len(df)} valid={df['IFTA_model'].notna().sum()} → {args.out}")


if __name__ == "__main__":
    main()
