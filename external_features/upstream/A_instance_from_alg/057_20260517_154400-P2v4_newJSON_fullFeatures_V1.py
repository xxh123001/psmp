#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
057 - P2v4: 基于新版 JSON (20260517_153144) 重新提取全部 glom/tubule/artery instance 特征
  合并 P2 (几何+病理标志) + P33 (KG1 细粒度特征) 为一步

新JSON路径:
  part001: .../20260517_153144_86样本JSON_part001of002/007_json/{sample}/alg_results/hr0_alg.json/
  part002: .../007_json/{sample}/alg_results/hr0_alg.json/

输出:
  009_csv/{NNN}_{ts}-P2v4_instance_glom_V1.csv
  009_csv/{NNN}_{ts}-P2v4_instance_tubule_V1.csv
  009_csv/{NNN}_{ts}-P2v4_instance_artery_V1.csv
"""
from __future__ import annotations

import json, logging, math, multiprocessing as mp, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# § 0 路径
# ============================================================
开始时间 = datetime.now()
ts = 开始时间.strftime("%Y%m%d_%H%M%S")

proj = Path(os.environ.get("PSMP_PROJ_DIR", "./project") + "/"
            "01.20260517_060800_V5_NGM-DKD-PAS_replica")
csv_dir = proj / "009_csv"
log_dir = proj / "013_log"
for d in (csv_dir, log_dir):
    d.mkdir(parents=True, exist_ok=True)

# 新 JSON 两个 part
new_root = Path(os.environ.get("PSMP_HOME_DIR", ".") + "/Downloads/"
                "15.DigitalDNUPTx_全量推理_86样本_20260517_153144")
json_dir1 = new_root / "20260517_153144_86样本JSON_part001of002" / "007_json"
json_dir2 = new_root / "007_json"

def nxt(d):
    nums = [int(re.match(r"(\d+)_", p.name).group(1))
            for p in d.iterdir() if re.match(r"(\d+)_", p.name)]
    return f"{max(nums, default=0)+1:03d}"

seq = nxt(csv_dir)
out_glom = csv_dir / f"{seq}_{ts}-P2v4_instance_glom_V1.csv"
out_tubule = csv_dir / f"{seq}_{ts}-P2v4_instance_tubule_V1.csv"
out_artery = csv_dir / f"{seq}_{ts}-P2v4_instance_artery_V1.csv"
log_file = log_dir / f"{nxt(log_dir)}_{ts}-P2v4_newJSON_fullFeatures.log"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)


# ============================================================
# § 1 常量
# ============================================================
炎症标签_映射 = {
    "淋巴细胞": "lymph", "中性粒细胞": "neutro", "非典型中性粒细胞": "neutro",
    "浆细胞": "plasma", "嗜酸性-典型": "eosino", "嗜酸性-困难": "eosino",
    "嗜酸性-三核": "eosino",
}
W_10类 = ["W0", "W10", "W11", "W12", "W13", "W20", "W21", "W22", "W3", "W4"]
KT1_5类_映射 = {"KT1_D1": "PT", "KT1_D2": "DT", "KT1_D3": "TAL",
                "KT1_D4": "CD", "KT1_D5": "异常"}
KT1_5类_列 = ["PT", "DT", "TAL", "CD", "异常"]

KG_label_nodes = [
    "KG1_W", "KG1_W1", "KG1_W12",
    "KG1_GBM", "KG1_GBM1",
    "KG1_S", "KG1_SC", "KG1_SC2", "KG1_SCF",
    "KG1_FSGS", "KG1_CC", "KG1_CC0", "KG1_CC1",
    "KG1_CR", "KG1_CR2", "KG1_CRX",
    "KG1_N", "KG1_I", "KG1_I0",
    "KG1_F", "KG1_EG", "KG1_SY",
    "KG1_B", "KG1_BT", "KG1_BTX",
    "KG1_CI", "KG1_CS", "KG1_G",
]
KV_label_nodes = ["KV11_W", "KV11_W0", "KV11_W1"]


# ============================================================
# § 2 工具函数
# ============================================================
def bbx_center(bbx):
    if not bbx or len(bbx) < 4:
        return None
    return (bbx[0] + bbx[2] / 2.0, bbx[1] + bbx[3] / 2.0)


def trace_to_root(start_id, id_to_kId, target_ids, max_depth=5):
    current = start_id
    visited = set()
    for _ in range(max_depth):
        if current in target_ids:
            return current
        if current in visited:
            return None
        visited.add(current)
        nxt_id = id_to_kId.get(current)
        if not nxt_id:
            return None
        current = nxt_id
    return current if current in target_ids else None


def polygon_geometry(points, mpp):
    nan_result = {"area_um2": np.nan, "perimeter_um": np.nan,
                  "circularity": np.nan, "solidity": np.nan,
                  "eccentricity": np.nan, "elongation": np.nan,
                  "major_axis_um": np.nan, "minor_axis_um": np.nan}
    if points is None or len(points) < 3:
        return nan_result
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        coords = [(p[0], p[1]) for p in points if len(p) >= 2]
        if len(coords) < 3:
            return nan_result
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        area_px = poly.area; per_px = poly.length
        area_um2 = area_px * mpp * mpp; per_um = per_px * mpp
        circ = (4 * math.pi * area_px / (per_px ** 2)) if per_px > 0 else np.nan
        hull = poly.convex_hull
        sol = (area_px / hull.area) if hull.area > 0 else np.nan
    except Exception:
        return nan_result
    try:
        xy = np.asarray(coords)
        cov = np.cov(xy, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov)
        if eigvals[1] <= 0:
            ecc = elong = major = minor = np.nan
        else:
            minor = 2 * math.sqrt(max(eigvals[0], 0)) * mpp
            major = 2 * math.sqrt(eigvals[1]) * mpp
            ecc = math.sqrt(max(1 - eigvals[0] / eigvals[1], 0))
            elong = 1 - (minor / major) if major > 0 else np.nan
    except Exception:
        ecc = elong = major = minor = np.nan
    return {"area_um2": area_um2, "perimeter_um": per_um,
            "circularity": circ, "solidity": sol,
            "eccentricity": ecc, "elongation": elong,
            "major_axis_um": major, "minor_axis_um": minor}


def 粗W_to_int_KG(label):
    m = re.search(r"KG1_W(\d)$", label or "")
    return int(m.group(1)) if m else -1

def 粗W_to_int_KT(label):
    m = re.search(r"KT1_W(\d)$", label or "")
    return int(m.group(1)) if m else -1

def 细W_label_KT(label):
    m = re.search(r"KT1_W(\d+)$", label or "")
    return "W" + m.group(1) if m else ""


# ============================================================
# § 3 KG1 细粒度特征 (from P33)
# ============================================================
def extract_kg1_fine_features(gid, lbls):
    """对单个 KG1 提取全部细粒度标签特征"""
    feat = {}

    # EG
    eg_labels = lbls.get("KG1_EG", [])
    if "KG1_EG0" in eg_labels: feat["eg_class"] = 0
    elif "KG1_EG1" in eg_labels: feat["eg_class"] = 1
    elif "KG1_EG2" in eg_labels: feat["eg_class"] = 2
    else: feat["eg_class"] = np.nan

    # SC
    sc_labels = lbls.get("KG1_SC", [])
    if "KG1_SC0" in sc_labels: feat["sc_type"] = 0
    elif "KG1_SC1" in sc_labels: feat["sc_type"] = 1
    elif "KG1_SC2" in sc_labels: feat["sc_type"] = 2
    elif "KW" in sc_labels: feat["sc_type"] = 3
    else: feat["sc_type"] = np.nan
    feat["has_KW_nodule"] = 1 if "KW" in sc_labels else 0

    # SC2
    sc2_labels = lbls.get("KG1_SC2", [])
    feat["is_ischemic_sclerosis"] = 1 if "缺血性硬化" in sc2_labels else 0
    feat["is_nonischemic_sclerosis"] = 1 if "非缺血性硬化" in sc2_labels else 0

    # SCF
    scf_labels = lbls.get("KG1_SCF", [])
    feat["is_global_sclerosis_scf"] = 1 if "球性硬化" in scf_labels else 0
    feat["is_segmental_sclerosis_scf"] = 1 if "非球性硬化" in scf_labels else 0

    # F
    f_labels = lbls.get("KG1_F", [])
    f_grade = np.nan
    for fl in f_labels:
        if fl == "KG1_F0": f_grade = 0
        elif fl == "KG1_F1": f_grade = 1
        elif fl == "KG1_F2": f_grade = 2
        elif fl == "KG1_F3": f_grade = 3
        elif fl == "KG1_F4": f_grade = 4
    feat["fibrosis_grade"] = f_grade

    # GBM
    gbm_labels = lbls.get("KG1_GBM", [])
    gbm1_labels = lbls.get("KG1_GBM1", [])
    feat["has_spike_track"] = 1 if "钉突双轨" in gbm_labels else 0
    feat["has_wire_loop"] = 1 if "白金耳" in gbm_labels else 0
    if "白金耳" in gbm_labels: feat["gbm_type"] = 3
    elif "钉突双轨" in gbm_labels: feat["gbm_type"] = 2
    elif "淀粉样变" in gbm_labels: feat["gbm_type"] = 4
    elif "其他增厚" in gbm_labels: feat["gbm_type"] = 1
    else: feat["gbm_type"] = 0

    # CR
    cr_labels = lbls.get("KG1_CR", [])
    cr2_labels = lbls.get("KG1_CR2", [])
    crx_labels = lbls.get("KG1_CRX", [])
    if "KG1_CR2" in cr_labels or "细胞新月体" in cr2_labels: feat["crescent_type"] = 2
    elif "KG1_CR3" in cr_labels: feat["crescent_type"] = 3
    elif "KG1_CR4" in cr_labels: feat["crescent_type"] = 4
    elif "疑似新月体" in crx_labels: feat["crescent_type"] = 1
    else: feat["crescent_type"] = 0

    # B / BT
    bt_labels = lbls.get("KG1_BT", [])
    if "KG1_BT3" in bt_labels: feat["bowman_thickening"] = 3
    elif "KG1_BT2" in bt_labels: feat["bowman_thickening"] = 2
    elif "KG1_BT1" in bt_labels: feat["bowman_thickening"] = 1
    else: feat["bowman_thickening"] = 0

    # CS
    cs_labels = lbls.get("KG1_CS", [])
    feat["has_collapse"] = 1 if "皱缩塌陷" in cs_labels else 0

    # I0
    i0_labels = lbls.get("KG1_I0", [])
    feat["has_mesangial_prolif"] = 1 if "系膜增生" in i0_labels else 0

    # G
    g_labels = lbls.get("KG1_G", [])
    feat["g_has_thickening"] = 1 if "有增厚" in g_labels else 0
    feat["g_KW_nodule"] = 1 if "KW结节" in g_labels else 0

    # SY
    sy_labels = lbls.get("KG1_SY", [])
    feat["has_synechiae"] = 1 if sy_labels else 0

    # I (cell subtypes)
    i_labels = lbls.get("KG1_I", [])
    i_counter = Counter(i_labels)
    for sub in ["I0", "I1", "I2", "I3", "I7", "I8"]:
        feat[f"n_cell_{sub}"] = i_counter.get(f"KG1_{sub}", 0)
    feat["n_cell_total_json"] = sum(i_counter.values())

    # CI
    ci_labels = lbls.get("KG1_CI", [])
    feat["n_inflam_cells_json"] = len(ci_labels)

    return feat


# ============================================================
# § 4 单样本处理
# ============================================================
def find_sample_alg_dir(sample_id):
    for d in [json_dir1, json_dir2]:
        p = d / sample_id / "alg_results" / "hr0_alg.json"
        if p.exists():
            return p
    return None


def process_sample(sample_id):
    alg_dir = find_sample_alg_dir(sample_id)
    if alg_dir is None:
        return {"sample": sample_id, "status": "error", "err": "not found",
                "glom": [], "tubule": [], "artery": []}

    # 加载所有 JSON
    nodes_dict = {}
    id_to_kId = {}
    for p in sorted(alg_dir.glob("*.json")):
        try:
            items = json.load(open(p, encoding="utf-8"))
            if isinstance(items, dict):
                items = items.get("nodes", items.get("data", []))
            nodes_dict[p.stem] = items
            for it in items:
                iid = it.get("id", "")
                kid = it.get("kId", "")
                if iid and kid:
                    id_to_kId[iid] = kid
        except Exception:
            pass

    K_main = nodes_dict.get("K", [])
    if not K_main:
        return {"sample": sample_id, "status": "error", "err": "K.json empty",
                "glom": [], "tubule": [], "artery": []}

    mpp = 0.2518
    KG1_items = [k for k in K_main if k.get("label") == "KG1"]
    KT1_items = [k for k in K_main if k.get("label") == "KT1"]
    KV1_items = [k for k in K_main if k.get("label") == "KV1"]
    KG1_ids = {k["id"] for k in KG1_items}
    KV1_ids = {k["id"] for k in KV1_items}

    # ── 炎症 trees ──
    cell_id_to_C = {c["id"]: c for c in nodes_dict.get("KI1_C", [])}
    炎症_by_type = defaultdict(list)
    for cc in nodes_dict.get("KI1_CC", []):
        parent_id = cc.get("kId", "")
        c = cell_id_to_C.get(parent_id)
        if c is None:
            continue
        raw = cc.get("label", "")
        type5 = 炎症标签_映射.get(raw)
        if type5 is None:
            continue
        center = bbx_center(c.get("bbx"))
        if center is None:
            continue
        炎症_by_type[type5].append((center[0] * mpp, center[1] * mpp))
    inflam_trees = {t: cKDTree(np.array(lst)) if lst else None
                    for t, lst in 炎症_by_type.items()}

    # KG1 中心树
    KG1_xys = []
    for k in KG1_items:
        c = bbx_center(k.get("bbx"))
        if c is not None:
            KG1_xys.append((c[0] * mpp, c[1] * mpp))
    KG1_xy_tree = cKDTree(np.array(KG1_xys)) if len(KG1_xys) > 1 else None

    # KG1 子节点映射
    KG_映射 = {}
    for name in KG_label_nodes:
        mp_ = defaultdict(list)
        for it in nodes_dict.get(name, []):
            kid = it.get("kId", "")
            if not kid:
                continue
            target = trace_to_root(kid, id_to_kId, KG1_ids, max_depth=5)
            if target is None:
                continue
            mp_[target].append((it.get("id", ""), it.get("label", "")))
        KG_映射[name] = mp_

    # KG1 细粒度标签 (for P33-style features)
    kg1_labels = defaultdict(lambda: defaultdict(list))
    target_nodes_for_fine = [
        "KG1_EG", "KG1_SC", "KG1_SC2", "KG1_SCF", "KG1_F",
        "KG1_GBM", "KG1_GBM1", "KG1_CR", "KG1_CR2", "KG1_CRX",
        "KG1_FSGS", "KG1_W", "KG1_W1",
        "KG1_B", "KG1_BT", "KG1_BTX",
        "KG1_CS", "KG1_I0", "KG1_G",
        "KG1_SY", "KG1_N", "KG1_CC0",
        "KG1_S", "KG1_I", "KG1_CI",
    ]
    for node_name in target_nodes_for_fine:
        for it in nodes_dict.get(node_name, []):
            kid = it.get("kId", "")
            if not kid:
                continue
            target = trace_to_root(kid, id_to_kId, KG1_ids, max_depth=5)
            if target is None:
                continue
            lbl = it.get("label", "")
            if lbl:
                kg1_labels[target][node_name].append(lbl)

    glom_W = {}
    for it in nodes_dict.get("KG1_W", []):
        kid = it.get("kId", "")
        if not kid:
            continue
        target = trace_to_root(kid, id_to_kId, KG1_ids)
        if target is None:
            continue
        W = 粗W_to_int_KG(it.get("label", ""))
        if W >= 0:
            glom_W[target] = W

    GBM_set = set(KG_映射["KG1_GBM"].keys()) | set(KG_映射["KG1_GBM1"].keys())
    glob_scler_set = (set(KG_映射["KG1_S"].keys()) | set(KG_映射["KG1_SC"].keys())
                      | set(KG_映射["KG1_SC2"].keys()) | set(KG_映射["KG1_SCF"].keys()))
    fsgs_set = set(KG_映射["KG1_FSGS"].keys())
    crescent_strict = set(KG_映射["KG1_CC0"].keys()) | set(KG_映射["KG1_CC1"].keys())
    necrosis_set = set(KG_映射["KG1_N"].keys())
    effective_set = set(KG_映射["KG1_I"].keys()) | set(KG_映射["KG1_I0"].keys())
    broken_set = (set(KG_映射["KG1_CR"].keys()) | set(KG_映射["KG1_CR2"].keys())
                  | set(KG_映射["KG1_CRX"].keys()))
    bowman_caps_set = (set(KG_映射["KG1_B"].keys()) | set(KG_映射["KG1_BT"].keys())
                       | set(KG_映射["KG1_BTX"].keys()))
    inflam_in_glom_set = set(KG_映射["KG1_CI"].keys())

    # ── GLOM 特征 ──
    glom_rows = []
    for idx, k in enumerate(KG1_items):
        K_id = k.get("id", "")
        bbx = k.get("bbx", [0, 0, 0, 0])
        center_px = bbx_center(bbx)
        if center_px is None:
            continue
        cx_um, cy_um = center_px[0] * mpp, center_px[1] * mpp

        geom = polygon_geometry(k.get("points"), mpp)
        bbx_w_um = bbx[2] * mpp; bbx_h_um = bbx[3] * mpp
        bbx_aspect = max(bbx_w_um, bbx_h_um) / max(min(bbx_w_um, bbx_h_um), 1e-6)
        dia_outer_um = float(k.get("diameter_outer", 0)) * mpp
        dia_inner_um = float(k.get("diameter_inner", 0)) * mpp
        tuft_area_um2 = math.pi * (dia_inner_um / 2) ** 2 if dia_inner_um > 0 else np.nan
        glom_area_um2 = (geom["area_um2"] if not np.isnan(geom["area_um2"])
                         else (math.pi * (dia_outer_um / 2) ** 2 if dia_outer_um > 0 else np.nan))
        tuft_frac = (tuft_area_um2 / glom_area_um2) if (glom_area_um2 and not np.isnan(tuft_area_um2)) else np.nan
        bowman_area = (glom_area_um2 - tuft_area_um2) if (glom_area_um2 and not np.isnan(tuft_area_um2)) else np.nan

        W_int = glom_W.get(K_id, -1)
        W_oh = {f"glom_is_W{i}": int(W_int == i) for i in range(5)}

        if KG1_xy_tree is not None:
            dists, _ = KG1_xy_tree.query([cx_um, cy_um], k=2)
            dist_nearest = float(dists[1])
        else:
            dist_nearest = np.nan

        n_GBM = len(KG_映射["KG1_GBM"].get(K_id, [])) + len(KG_映射["KG1_GBM1"].get(K_id, []))
        n_FSGS = len(KG_映射["KG1_FSGS"].get(K_id, []))
        n_CI = len(KG_映射["KG1_CI"].get(K_id, []))

        row = {
            "sample": sample_id, "glom_id": K_id, "glom_idx": idx,
            "cx_um": cx_um, "cy_um": cy_um,
            "bbx_w_um": bbx_w_um, "bbx_h_um": bbx_h_um, "bbx_aspect": bbx_aspect,
            "glom_dia_outer_um": dia_outer_um, "tuft_dia_inner_um": dia_inner_um,
            "glom_area_um2": glom_area_um2, "tuft_area_um2": tuft_area_um2,
            "tuft_area_fraction": tuft_frac, "bowman_area_um2": bowman_area,
            "perimeter_um": geom["perimeter_um"],
            "circularity": geom["circularity"], "solidity": geom["solidity"],
            "eccentricity": geom["eccentricity"], "elongation": geom["elongation"],
            "major_axis_um": geom["major_axis_um"], "minor_axis_um": geom["minor_axis_um"],
            "mesangial_W_int": W_int, **W_oh,
            "has_GBM_thicken": int(K_id in GBM_set), "n_GBM_patches": n_GBM,
            "has_global_sclerosis": int(K_id in glob_scler_set),
            "has_segmental_sclerosis_FSGS": int(K_id in fsgs_set), "n_FSGS_patches": n_FSGS,
            "has_crescent_strict": int(K_id in crescent_strict),
            "has_necrosis": int(K_id in necrosis_set),
            "is_effective_glom": int(K_id in effective_set),
            "has_broken_capsule": int(K_id in broken_set),
            "has_bowman_caps": int(K_id in bowman_caps_set),
            "n_intraglom_inflam_cells": n_CI,
            "has_intraglom_inflam": int(K_id in inflam_in_glom_set),
            "dist_nearest_glom_um": dist_nearest, "mpp": mpp,
        }

        # P33 细粒度特征
        fine = extract_kg1_fine_features(K_id, kg1_labels.get(K_id, {}))
        row.update(fine)

        glom_rows.append(row)

    # ── TUBULE 特征 ──
    KT1_id_to_W粗 = {}
    for item in nodes_dict.get("KT1_W", []):
        kid = item.get("kId", "")
        if kid:
            粗 = 粗W_to_int_KT(item.get("label", ""))
            if 粗 >= 0:
                KT1_id_to_W粗[kid] = 粗

    KT1_D_dict = {}
    for item in nodes_dict.get("KT1_D", []):
        kid = item.get("kId", "")
        if kid:
            KT1_D_dict[kid] = KT1_5类_映射.get(item.get("label", ""), "")

    tubule_rows = []
    for idx, k in enumerate(KT1_items):
        K_id = k["id"]
        bbx = k.get("bbx", [0, 0, 0, 0])
        center_px = bbx_center(bbx)
        if center_px is None:
            continue
        cx_um, cy_um = center_px[0] * mpp, center_px[1] * mpp
        dia = float(k.get("diameter_outer", 0)) * mpp
        W_int = KT1_id_to_W粗.get(K_id, -1)
        D_type = KT1_D_dict.get(K_id, "")

        W_oh = {f"tub_is_W{w}": int(False) for w in W_10类}
        if W_int >= 0:
            粗_label = f"W{W_int}"
            if 粗_label in W_10类:
                W_oh[f"tub_is_{粗_label}"] = 1

        D_oh = {f"tub_is_{d}": int(D_type == d) for d in KT1_5类_列}

        tubule_rows.append({
            "sample": sample_id, "tub_id": K_id, "tub_idx": idx,
            "cx_um": cx_um, "cy_um": cy_um,
            "tub_dia_um": dia,
            "tub_W_int": W_int, **W_oh,
            "tub_D_type": D_type, **D_oh,
            "tub_is_tubulitis": 0,  # placeholder
            "mpp": mpp,
        })

    # ── ARTERY 特征 ──
    KV11_W_dict = {}
    for wname in KV_label_nodes:
        for item in nodes_dict.get(wname, []):
            kid = item.get("kId", "")
            if kid:
                KV11_W_dict[kid] = item.get("label", "")

    artery_rows = []
    for idx, k in enumerate(KV1_items):
        K_id = k["id"]
        bbx = k.get("bbx", [0, 0, 0, 0])
        center_px = bbx_center(bbx)
        if center_px is None:
            continue
        cx_um, cy_um = center_px[0] * mpp, center_px[1] * mpp
        dia_outer = float(k.get("diameter_outer", 0)) * mpp
        dia_inner = float(k.get("diameter_inner", 0)) * mpp
        wall_thick = (dia_outer - dia_inner) / 2 if dia_outer > dia_inner else 0
        wall_lumen = wall_thick / (dia_inner / 2) if dia_inner > 0 else np.nan
        is_small = int(dia_outer < 100)

        artery_rows.append({
            "sample": sample_id, "art_id": K_id, "art_idx": idx,
            "cx_um": cx_um, "cy_um": cy_um,
            "art_dia_outer_um": dia_outer, "art_dia_inner_um": dia_inner,
            "wall_thickness_um": wall_thick, "wall_lumen_ratio": wall_lumen,
            "is_small_arteriole": is_small,
            "KV11_subclass": KV11_W_dict.get(K_id, ""),
            "mpp": mpp,
        })

    return {
        "sample": sample_id, "status": "ok",
        "n_KG1": len(KG1_items), "n_KT1": len(KT1_items), "n_KV1": len(KV1_items),
        "glom": glom_rows, "tubule": tubule_rows, "artery": artery_rows,
    }


# ============================================================
# § 5 main
# ============================================================
def main():
    log.info("=" * 72)
    log.info("P2v4 - 新 JSON (20260517) 全特征提取 (glom+tubule+artery)")
    log.info(f"  JSON part1: {json_dir1}")
    log.info(f"  JSON part2: {json_dir2}")
    log.info("=" * 72)

    # 收集所有样本
    samples = set()
    for d in [json_dir1, json_dir2]:
        if d.exists():
            for p in d.iterdir():
                if p.is_dir() and p.name != ".DS_Store":
                    samples.add(p.name)
    samples = sorted(samples)
    log.info(f"发现 {len(samples)} 个样本")

    n_worker = min(8, mp.cpu_count())
    log.info(f"启动 {n_worker} workers")

    t0 = time.time()
    all_glom = []; all_tubule = []; all_artery = []
    diagnostics = []; errs = []

    with ProcessPoolExecutor(max_workers=n_worker) as ex:
        futures = {ex.submit(process_sample, s): s for s in samples}
        for i, fut in enumerate(as_completed(futures), 1):
            name = futures[fut]
            try:
                res = fut.result()
                if res["status"] == "error":
                    errs.append((res["sample"], res["err"]))
                    log.warning(f"  [{i}/{len(samples)}] {res['sample']}: ERROR {res['err']}")
                else:
                    all_glom.extend(res["glom"])
                    all_tubule.extend(res["tubule"])
                    all_artery.extend(res["artery"])
                    diagnostics.append({
                        "sample": res["sample"],
                        "n_KG1": res["n_KG1"], "n_KT1": res["n_KT1"], "n_KV1": res["n_KV1"],
                    })
                    if i % 20 == 0 or i == len(samples):
                        log.info(f"  进度 {i}/{len(samples)} | "
                                  f"glom={len(all_glom)} tub={len(all_tubule)} art={len(all_artery)}")
            except Exception as e:
                errs.append((name, str(e)))

    log.info(f"\n完成. 成功 {len(diagnostics)} / 失败 {len(errs)} / "
              f"耗时 {time.time()-t0:.1f}s")

    if not all_glom:
        log.error("无 glom 特征行, 退出")
        sys.exit(1)

    df_g = pd.DataFrame(all_glom)
    df_t = pd.DataFrame(all_tubule) if all_tubule else pd.DataFrame()
    df_v = pd.DataFrame(all_artery) if all_artery else pd.DataFrame()
    log.info(f"glom shape   = {df_g.shape}")
    log.info(f"tubule shape = {df_t.shape}")
    log.info(f"artery shape = {df_v.shape}")

    df_g.to_csv(out_glom, index=False)
    df_t.to_csv(out_tubule, index=False) if len(df_t) else None
    df_v.to_csv(out_artery, index=False) if len(df_v) else None
    log.info(f"\n落盘 glom   → {out_glom.name}")
    log.info(f"落盘 tubule → {out_tubule.name}")
    log.info(f"落盘 artery → {out_artery.name}")

    # sanity check
    log.info("\n=== sanity check ===")
    log.info(f"per-sample glom count:")
    gc = df_g.groupby("sample").size()
    log.info(f"  median={gc.median():.0f}  min={gc.min()}  max={gc.max()}")
    log.info(f"  total glom = {len(df_g)}")

    # 直径分布
    dia = df_g["glom_dia_outer_um"]
    log.info(f"\nglom_dia_outer_um: <100={int((dia<100).sum())} | "
             f"100-500={int(((dia>=100)&(dia<=500)).sum())} | >500={int((dia>500).sum())}")

    log.info(f"\n耗时 {(datetime.now()-开始时间).total_seconds():.1f}s")
    log.info("=" * 72)
    log.info("DONE")


if __name__ == "__main__":
    main()
