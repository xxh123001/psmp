#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
058 - P34v2: Patient-level aggregation from P2v4 (new JSON) glom+tubule+artery
  直接用 P2v4 instance CSV 聚合, 输出患者级特征
"""
import logging, re, sys, time
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import pandas as pd

proj = Path(os.environ.get("PSMP_PROJ_DIR", "./project") + "/"
            "01.20260517_060800_V5_NGM-DKD-PAS_replica")
csv_dir = proj / "009_csv"
log_dir = proj / "013_log"
for d in (csv_dir, log_dir):
    d.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
def nxt(d):
    n = [int(re.match(r"(\d+)_", p.name).group(1))
         for p in d.iterdir() if re.match(r"(\d+)_", p.name)]
    return f"{max(n, default=0)+1:03d}"

log_file = log_dir / f"{nxt(log_dir)}_{ts}-P34v2_patient_agg_P2v4.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


def continuous_summary(series, prefix):
    s = series.dropna()
    if len(s) == 0:
        return {f"{prefix}_median": np.nan, f"{prefix}_IQR": np.nan,
                f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
    return {f"{prefix}_median": q2, f"{prefix}_IQR": q3 - q1,
            f"{prefix}_mean": s.mean(), f"{prefix}_std": s.std()}


def binary_fraction(series, prefix):
    s = series.dropna()
    if len(s) == 0:
        return {f"{prefix}_pct": np.nan}
    return {f"{prefix}_pct": float(s.mean())}


def ordinal_summary(series, prefix, levels):
    s = series.dropna()
    result = {}
    result[f"{prefix}_median"] = s.median() if len(s) else np.nan
    result[f"{prefix}_mean"] = s.mean() if len(s) else np.nan
    for lv in levels:
        result[f"{prefix}_frac_{lv}"] = float((s == lv).mean()) if len(s) else np.nan
    return result


def main():
    log.info("=" * 60)
    log.info("P34v2 - Patient aggregation from P2v4 new JSON")
    log.info("=" * 60)
    t0 = time.time()

    # ── 读取 P2v4 CSV ──
    glom_csv = sorted(csv_dir.glob("*-P2v4_instance_glom_*.csv"))[-1]
    tub_csv = sorted(csv_dir.glob("*-P2v4_instance_tubule_*.csv"))[-1]
    art_csv = sorted(csv_dir.glob("*-P2v4_instance_artery_*.csv"))[-1]
    df_g = pd.read_csv(glom_csv)
    df_t = pd.read_csv(tub_csv)
    df_v = pd.read_csv(art_csv)
    log.info(f"Glom: {glom_csv.name}, shape={df_g.shape}")
    log.info(f"Tubule: {tub_csv.name}, shape={df_t.shape}")
    log.info(f"Artery: {art_csv.name}, shape={df_v.shape}")

    # ═══════════════════════════════════════════════
    # GLOM 聚合
    # ═══════════════════════════════════════════════
    # 连续特征
    glom_cont = [c for c in [
        "glom_dia_outer_um", "tuft_dia_inner_um",
        "glom_area_um2", "tuft_area_um2", "tuft_area_fraction", "bowman_area_um2",
        "perimeter_um", "circularity", "solidity", "eccentricity", "elongation",
        "major_axis_um", "minor_axis_um", "dist_nearest_glom_um",
    ] if c in df_g.columns]

    # 二值特征
    glom_bin = [c for c in [
        "has_GBM_thicken", "has_global_sclerosis",
        "has_segmental_sclerosis_FSGS",
        "has_crescent_strict", "has_necrosis", "is_effective_glom",
        "has_broken_capsule", "has_bowman_caps", "has_intraglom_inflam",
        "has_KW_nodule",
        "is_ischemic_sclerosis", "is_nonischemic_sclerosis",
        "is_global_sclerosis_scf", "is_segmental_sclerosis_scf",
        "has_spike_track", "has_wire_loop",
        "has_collapse", "has_mesangial_prolif",
        "g_has_thickening", "g_KW_nodule",
        "has_synechiae",
    ] if c in df_g.columns]

    # 计数特征
    glom_count = [c for c in [
        "n_GBM_patches", "n_FSGS_patches", "n_intraglom_inflam_cells",
        "n_cell_I0", "n_cell_I1", "n_cell_I2", "n_cell_I3",
        "n_cell_I7", "n_cell_I8",
        "n_cell_total_json", "n_inflam_cells_json",
    ] if c in df_g.columns]

    # ordinal 特征
    glom_ordinal = {}
    for c, levels in {
        "eg_class": [0, 1, 2],
        "fibrosis_grade": [0, 1, 2, 3, 4],
        "bowman_thickening": [0, 1, 2, 3],
        "crescent_type": [0, 1, 2, 3, 4],
        "gbm_type": [0, 1, 2, 3, 4],
        "sc_type": [0, 1, 2, 3],
    }.items():
        if c in df_g.columns:
            glom_ordinal[c] = levels

    # 聚合
    glom_rows = []
    for sample, sub in df_g.groupby("sample"):
        row = {"sample": sample, "n_glom": len(sub)}
        row["glom_area_total_um2"] = sub["glom_area_um2"].sum(skipna=True)

        for c in glom_cont:
            row.update(continuous_summary(sub[c], f"glom_{c}"))
        for c in glom_bin:
            row.update(binary_fraction(sub[c], f"glom_{c}"))
        for c in glom_count:
            row.update(continuous_summary(sub[c], f"glom_{c}"))
        for c, levels in glom_ordinal.items():
            row.update(ordinal_summary(sub[c], f"glom_{c}", levels))

        # W 分布
        if "mesangial_W_int" in sub.columns:
            for i in range(5):
                row[f"glom_W{i}_pct"] = float((sub["mesangial_W_int"] == i).mean())
            row["glom_W_unmapped_pct"] = float((sub["mesangial_W_int"] == -1).mean())

        glom_rows.append(row)

    df_pat_glom = pd.DataFrame(glom_rows)
    log.info(f"Glom patient features: {df_pat_glom.shape}")

    # ═══════════════════════════════════════════════
    # TUBULE 聚合
    # ═══════════════════════════════════════════════
    tub_rows = []
    for sample, sub in df_t.groupby("sample"):
        row = {"sample": sample, "n_tubule": len(sub)}
        row.update(continuous_summary(sub["tub_dia_um"], "tub_dia_um"))

        # W 分布
        if "tub_W_int" in sub.columns:
            for i in range(5):
                row[f"tub_W{i}_pct"] = float((sub["tub_W_int"] == i).mean())
            row["tub_W_unmapped_pct"] = float((sub["tub_W_int"] == -1).mean())

        # D type 分布
        if "tub_D_type" in sub.columns:
            for dtype in ["PT", "DT", "TAL", "CD", "异常"]:
                row[f"tub_{dtype}_pct"] = float((sub["tub_D_type"] == dtype).mean())

        tub_rows.append(row)

    df_pat_tub = pd.DataFrame(tub_rows)
    log.info(f"Tubule patient features: {df_pat_tub.shape}")

    # ═══════════════════════════════════════════════
    # ARTERY 聚合
    # ═══════════════════════════════════════════════
    art_rows = []
    for sample, sub in df_v.groupby("sample"):
        row = {"sample": sample, "n_artery": len(sub)}
        row.update(continuous_summary(sub["art_dia_outer_um"], "art_dia_outer_um"))
        row.update(continuous_summary(sub["art_dia_inner_um"], "art_dia_inner_um"))
        row.update(continuous_summary(sub["wall_thickness_um"], "art_wall_thickness_um"))
        row.update(continuous_summary(sub["wall_lumen_ratio"], "art_wall_lumen_ratio"))
        row["art_small_arteriole_pct"] = float(sub["is_small_arteriole"].mean())
        art_rows.append(row)

    df_pat_art = pd.DataFrame(art_rows)
    log.info(f"Artery patient features: {df_pat_art.shape}")

    # ═══════════════════════════════════════════════
    # 合并三轴
    # ═══════════════════════════════════════════════
    df_pat = df_pat_glom.merge(df_pat_tub, on="sample", how="outer")
    df_pat = df_pat.merge(df_pat_art, on="sample", how="outer")
    log.info(f"Combined shape: {df_pat.shape}")

    # ═══════════════════════════════════════════════
    # 合并临床数据
    # ═══════════════════════════════════════════════
    clin_files = sorted(csv_dir.glob("*-P4_merged_86例_*.csv"))
    if clin_files:
        df_clin = pd.read_csv(clin_files[-1])
        clin_cols = [c for c in ["sample", "eGFR_CKDEPI", "Class_All", "Class_All_ord",
                     "cohort", "50%eGFR", "age", "gender", "proteinuria_24h",
                     "serum_creatinine", "serum_albumin"] if c in df_clin.columns]
        df_merged = df_pat.merge(df_clin[clin_cols], on="sample", how="left")
        log.info(f"Merged clinical: +{len(clin_cols)-1} cols")
    else:
        df_merged = df_pat
        log.warning("No clinical data found!")

    # ═══════════════════════════════════════════════
    # 保存
    # ═══════════════════════════════════════════════
    out_csv = csv_dir / f"{nxt(csv_dir)}_{ts}-P3v4_patient_features_P2v4_V1.csv"
    df_merged.to_csv(out_csv, index=False)
    log.info(f"\nSaved: {out_csv.name}")
    log.info(f"Shape: {df_merged.shape}")
    log.info(f"Columns: {len(df_merged.columns)}")

    log.info(f"\nDONE in {time.time()-t0:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
