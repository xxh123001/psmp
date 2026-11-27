#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure-level quality control for prognostic analysis (Methods 4.4).

Applies the pre-specified QC rules to per-structure instance tables,
then the biopsy-level minimum-yield rule.

Instance-level rules
--------------------
  (i)   tubule area > 50 um^2                    (tubules only)
  (ii)  successful basement-membrane extraction  (tubules only):
        TBM / epithelial / luminal sub-measures obtainable.
        Uses explicit flag column when present (bm_ok/tbm_ok), otherwise
        proxies by non-null inner/lumen measures
        (tub_dia_inner_um + tub_lumen_area_um2 or tbm_thick_min_um).
  (iii) structure-instance classification confidence > 0.6
        (column: conf / confidence / cls_conf / score).
  (iv)  non-overlap with vascular or glomerular masks:
        explicit flag column when present
        (overlap_* -> keep 0, or non_overlap -> keep 1);
        if no overlap column exists the rule is reported as not applied.

Biopsy-level rule
-----------------
  exclude biopsies with < 8 quality-controlled glomeruli
  (outer diameter >= 100 um) or < 20 quality-controlled tubules.

Inputs are the three instance CSVs of the pipeline schema
(sample, *_id, cx_um, cy_um, feature columns...). Glomeruli are counted
after the >=100 um diameter filter of script 06 (applied here unless
--no_dia_filter).

Usage:
  python 07_structure_qc.py --tubule instance_tubule.csv \
      --glom instance_glom.csv --out_dir qc_out/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AREA_MIN_UM2 = 50.0      # rule (i)
CONF_MIN = 0.6           # rule (iii)
GLOM_DIA_MIN_UM = 100.0  # Methods 5/Results 2
MIN_GLOM_PER_BIOPSY = 8
MIN_TUB_PER_BIOPSY = 20

CONF_CANDS = ["conf", "confidence", "cls_conf", "class_conf", "score"]
AREA_CANDS = ["tub_area_um2", "area_um2", "tubule_area_um2"]
DIA_CANDS = ["glom_dia_outer_um", "dia_outer_um", "dia_um", "glom_dia_um"]
BM_FLAG_CANDS = ["bm_ok", "tbm_ok", "bm_extract_ok"]
BM_PROXY_COLS = [["tub_dia_inner_um", "tub_lumen_area_um2"], ["tbm_thick_min_um"]]
OVERLAP_NEG = ["overlap", "overlap_mask", "overlap_vessel_glom", "overlap_vg"]
OVERLAP_POS = ["non_overlap", "no_overlap", "nonoverlap"]


def _pick(df: pd.DataFrame, cands) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def qc_tubules(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply rules (i)-(iv) to a tubule instance table."""
    rep = {}
    keep = pd.Series(True, index=df.index)

    c = _pick(df, AREA_CANDS)
    if c is None:
        sys.exit("tubule CSV: no area column (tub_area_um2/area_um2)")
    ok = pd.to_numeric(df[c], errors="coerce") > AREA_MIN_UM2
    rep["area>50um2"] = int(ok.sum())
    keep &= ok.fillna(False)

    f = _pick(df, BM_FLAG_CANDS)
    if f is not None:
        ok = df[f].astype(float) > 0
        rep["bm_flag"] = int(ok.sum())
        keep &= ok
    else:
        cols = next((cc for cc in BM_PROXY_COLS if all(c in df.columns for c in cc)), None)
        if cols is None:
            sys.exit("tubule CSV: cannot verify BM extraction (no bm_ok/tbm_ok flag, "
                     "no inner/lumen or tbm_thick_min_um columns)")
        ok = df[cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        rep["bm_proxy:" + "+".join(cols)] = int(ok.sum())
        keep &= ok

    c = _pick(df, CONF_CANDS)
    if c is None:
        sys.exit("tubule CSV: no confidence column (conf/confidence/cls_conf/score)")
    ok = pd.to_numeric(df[c], errors="coerce") > CONF_MIN
    rep["conf>0.6"] = int(ok.sum())
    keep &= ok.fillna(False)

    c = _pick(df, OVERLAP_NEG)
    p = _pick(df, OVERLAP_POS)
    if c is not None:
        ok = pd.to_numeric(df[c], errors="coerce").fillna(0) == 0
        rep[f"no_overlap({c}==0)"] = int(ok.sum())
        keep &= ok
    elif p is not None:
        ok = pd.to_numeric(df[p], errors="coerce").fillna(0) == 1
        rep[f"non_overlap({p}==1)"] = int(ok.sum())
        keep &= ok
    else:
        rep["overlap_rule"] = "not applied (no overlap column)"

    rep["kept"] = int(keep.sum())
    return df.loc[keep].copy(), rep


def qc_gloms(df: pd.DataFrame, dia_filter: bool = True) -> tuple[pd.DataFrame, dict]:
    """Gloms: confidence > 0.6 and non-overlap where columns exist; >=100 um."""
    rep = {}
    keep = pd.Series(True, index=df.index)

    if dia_filter:
        c = _pick(df, DIA_CANDS)
        if c is None:
            sys.exit("glom CSV: no diameter column for the >=100 um filter")
        ok = pd.to_numeric(df[c], errors="coerce") >= GLOM_DIA_MIN_UM
        rep["dia>=100um"] = int(ok.sum())
        keep &= ok.fillna(False)

    c = _pick(df, CONF_CANDS)
    if c is not None:
        ok = pd.to_numeric(df[c], errors="coerce") > CONF_MIN
        rep["conf>0.6"] = int(ok.sum())
        keep &= ok.fillna(False)

    c = _pick(df, OVERLAP_NEG)
    p = _pick(df, OVERLAP_POS)
    if c is not None:
        ok = pd.to_numeric(df[c], errors="coerce").fillna(0) == 0
        keep &= ok
        rep[f"no_overlap({c}==0)"] = int(ok.sum())
    elif p is not None:
        ok = pd.to_numeric(df[p], errors="coerce").fillna(0) == 1
        keep &= ok
        rep[f"non_overlap({p}==1)"] = int(ok.sum())

    rep["kept"] = int(keep.sum())
    return df.loc[keep].copy(), rep


def biopsy_qc(tub: pd.DataFrame, glom: pd.DataFrame) -> pd.DataFrame:
    """Per-biopsy yield table + include flag (>=8 gloms, >=20 tubules)."""
    nt = tub.groupby("sample").size().rename("n_tub_qc")
    ng = glom.groupby("sample").size().rename("n_glom_qc")
    tab = pd.concat([nt, ng], axis=1).fillna(0).astype(int).reset_index()
    tab["include"] = (
        (tab["n_glom_qc"] >= MIN_GLOM_PER_BIOPSY)
        & (tab["n_tub_qc"] >= MIN_TUB_PER_BIOPSY)
    ).astype(int)
    tab["exclude_reason"] = np.where(
        tab["n_glom_qc"] < MIN_GLOM_PER_BIOPSY, f"n_glom<{MIN_GLOM_PER_BIOPSY}", ""
    ) + np.where(
        tab["n_tub_qc"] < MIN_TUB_PER_BIOPSY,
        np.where(tab["n_glom_qc"] < MIN_GLOM_PER_BIOPSY, "+", "") + f"n_tub<{MIN_TUB_PER_BIOPSY}",
        "",
    )
    return tab


def main():
    ap = argparse.ArgumentParser(description="Structure/biopsy QC per Methods 4.4")
    ap.add_argument("--tubule", required=True, help="tubule instance CSV")
    ap.add_argument("--glom", required=True, help="glomerulus instance CSV")
    ap.add_argument("--out_dir", default="qc_out")
    ap.add_argument("--no_dia_filter", action="store_true",
                    help="do not re-apply the glom >=100 um diameter filter")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tub = pd.read_csv(args.tubule)
    glom = pd.read_csv(args.glom)
    tub_qc, rep_t = qc_tubules(tub)
    glom_qc, rep_g = qc_gloms(glom, dia_filter=not args.no_dia_filter)

    print(f"tubules: {len(tub)} -> {len(tub_qc)}   {rep_t}")
    print(f"glomeruli: {len(glom)} -> {len(glom_qc)}   {rep_g}")

    tab = biopsy_qc(tub_qc, glom_qc)
    kept = tab.loc[tab.include == 1, "sample"]
    print(f"biopsies: {len(tab)} -> include {int(tab.include.sum())} "
          f"(>= {MIN_GLOM_PER_BIOPSY} gloms & >= {MIN_TUB_PER_BIOPSY} tubules)")

    tub_qc[tub_qc["sample"].isin(kept)].to_csv(out / "instance_tubule_qc.csv", index=False)
    glom_qc[glom_qc["sample"].isin(kept)].to_csv(out / "instance_glom_qc.csv", index=False)
    tab.to_csv(out / "patient_qc_summary.csv", index=False)
    print("->", out)


if __name__ == "__main__":
    main()
