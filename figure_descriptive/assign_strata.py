#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assign the biopsy-level strata used by Results 3-5 to a patient table.

Adds three columns (all descriptive model-derived stratifiers, Methods 5):

  tervaert_group   pooled Tervaert class groups for Figure 3:
                   I/IIA/IIB -> "I-IIB", III -> "III", IV -> "IV"
                   (input column: Tervaert / tervaert_class / class, values
                   I, IA? IIA, IIB, III, IV or 1,2,3,4,5)
  ifta_stratum     biopsy-level IFTA severity stratum for Fig S2:
                   0: <=10%, 1: >10-25%, 2: >25-50%, 3: >50%
                   (input: IFTA_model cortical IFTA proportion, %)
  inflam_grade     model-derived interstitial inflammation grade analogue
                   for Fig S3 (Methods 5):
                   0: <10% of tubules show local inflammation
                   2: inflammation extends into non-atrophic cortex
                   1: otherwise

inflam_grade needs a tubule-level table (one row per QC'd tubule):
  sample, has_local_inflam (0/1; alias: any inflam cell within 50 um,
  e.g. n_inflam_50 > 0 or local_inflam), is_cortex (0/1; alias in_cortex),
  is_atrophic (0/1; alias W_grade >= 2 / atrophy flag).
"Extends into non-atrophic cortex" is approximated as: at least one
non-atrophic cortical tubule with local inflammation; the exact
cell-level criterion needs raw inflammatory-cell coordinates.

Usage:
  python assign_strata.py --patient patient_features.csv \
      [--tubule tubule_level_inflam.csv] \
      [--tervaert_col Tervaert] [--out patient_with_strata.csv]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def _pick(df: pd.DataFrame, cands, what):
    for c in cands:
        if c in df.columns:
            return c
    return None


def tervaert_group(s: pd.Series) -> pd.Series:
    """Pool Tervaert I/IIA/IIB; keep III and IV (Results 3)."""
    m = {
        "I": "I-IIB", "IA": "I-IIB", "IIA": "I-IIB", "IIB": "I-IIB",
        "III": "III", "IV": "IV",
        "1": "I-IIB", "2": "I-IIB", "3": "I-IIB", "4": "III", "5": "IV",
        1: "I-IIB", 2: "I-IIB", 3: "I-IIB", 4: "III", 5: "IV",
    }
    return s.map(lambda v: m.get(v, m.get(str(v).strip().upper(), np.nan)))


def ifta_stratum(s: pd.Series) -> pd.Series:
    """IFTA proportion strata: <=10 / 10-25 / 25-50 / >50 (Methods 5)."""
    v = pd.to_numeric(s, errors="coerce")
    return pd.cut(v, [-np.inf, 10, 25, 50, np.inf], labels=[0, 1, 2, 3]).astype(float)


def inflam_grade(tub: pd.DataFrame) -> pd.Series:
    """Model-derived inflammation grade analogue per biopsy (Methods 5)."""
    li = _pick(tub, ["has_local_inflam", "local_inflam"], "local inflam")
    cx = _pick(tub, ["is_cortex", "in_cortex"], "cortex")
    at = _pick(tub, ["is_atrophic", "atrophic"], "atrophic")
    n50 = _pick(tub, ["n_inflam_50", "n_inflam50"], "n_inflam_50")
    wg = _pick(tub, ["W_grade", "tub_W_int", "w_grade"], "W_grade")

    if li is None and n50 is not None:
        tub = tub.assign(_li=pd.to_numeric(tub[n50], errors="coerce").fillna(0) > 0)
        li = "_li"
    if at is None and wg is not None:
        tub = tub.assign(_at=pd.to_numeric(tub[wg], errors="coerce").fillna(0) >= 2)
        at = "_at"
    if li is None or cx is None or at is None:
        sys.exit("tubule table needs has_local_inflam (or n_inflam_50), "
                 "is_cortex and is_atrophic (or W_grade) columns")

    def grade(g: pd.DataFrame) -> int:
        frac_local = g[li].astype(float).mean()
        if frac_local < 0.10:
            return 0
        ext = ((g[cx].astype(float) > 0) & (g[at].astype(float) == 0) & (g[li].astype(float) > 0)).any()
        return 2 if ext else 1

    return tub.groupby("sample").apply(grade).rename("inflam_grade")


def main():
    ap = argparse.ArgumentParser(description="Assign Results 3-5 strata to patient table")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--tubule", default="", help="tubule-level table for inflam_grade")
    ap.add_argument("--tervaert_col", default="")
    ap.add_argument("--ifta_col", default="")
    ap.add_argument("--out", default="patient_with_strata.csv")
    args = ap.parse_args()

    pat = pd.read_csv(args.patient)
    if "sample" not in pat.columns:
        sys.exit("patient table needs a 'sample' column")

    tc = args.tervaert_col or _pick(pat, ["Tervaert", "tervaert_class", "class", "Class_All"], "tervaert")
    if tc is not None:
        pat["tervaert_group"] = tervaert_group(pat[tc])
        print("tervaert_group:", pat["tervaert_group"].value_counts(dropna=False).to_dict())
    else:
        print("no Tervaert column found; tervaert_group not assigned")

    ic = args.ifta_col or _pick(pat, ["IFTA_model", "IFTA"], "ifta")
    if ic is not None:
        pat["ifta_stratum"] = ifta_stratum(pat[ic])
        print("ifta_stratum:", pat["ifta_stratum"].value_counts(dropna=False).to_dict())
    else:
        print("no IFTA column found; ifta_stratum not assigned")

    if args.tubule:
        tub = pd.read_csv(args.tubule)
        pat = pat.merge(inflam_grade(tub).reset_index(), on="sample", how="left")
        print("inflam_grade:", pat["inflam_grade"].value_counts(dropna=False).to_dict())

    pat.to_csv(args.out, index=False)
    print("->", args.out)


if __name__ == "__main__":
    main()
