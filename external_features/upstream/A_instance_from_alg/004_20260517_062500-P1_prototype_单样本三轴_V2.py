#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本: 004_20260517_062500-P1_prototype_单样本三轴_V2.py
功能: V5 NGM-DKD-PAS P1 prototype V2 - 修复 B 类 KG1_* 节点多层 kId 追溯

V2 修复 (vs V1):
  V1 直接用 kId→KG1.id 一层 join, 但 DNUPTx chain4 的 B 类节点
  (KG1_GBM/FSGS/CC0/CRX/I0/G/CS/BT) 实际是两层桥:
    KG1_GBM.kId → 袢内血管极剔除.id (polygroup 中转)
    袢内血管极剔除.kId → KG1_CC.id  
    KG1_CC.kId → KG1.id

  V2 引入 trace_to_KG1(id, max_depth=5) 通用追溯函数, 一次性解决所有 B 类节点。

输出:
  009_csv/{NNN}_{ts}-P1_proto_单样本三轴_V2_glom_V1.csv
  009_csv/{NNN}_{ts}-P1_proto_单样本三轴_V2_tubule_V1.csv  (同 V1)
  009_csv/{NNN}_{ts}-P1_proto_单样本三轴_V2_artery_V1.csv  (同 V1)
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# § 0 路径与序号
# ============================================================
开始时间 = datetime.now()
时间戳 = 开始时间.strftime("%Y%m%d_%H%M%S")

项目根 = Path(
    os.environ.get("PSMP_PROJ_DIR", "./project") + "/"
    "01.20260517_060800_V5_NGM-DKD-PAS_replica"
)
csv目录 = 项目根 / "009_csv"
meta目录 = 项目根 / "008_meta"
log目录 = 项目根 / "013_log"
for d in (csv目录, meta目录, log目录):
    d.mkdir(parents=True, exist_ok=True)

样本ID = "1_S_5"
样本目录 = 项目根 / "011_data" / "raw" / "alg_results_86样本" / f"{样本ID}_alg_result"
alg_root = 样本目录 / "hr0_alg.json"


def 取下一个序号(目标目录: Path) -> str:
    已有 = []
    for p in 目标目录.iterdir():
        m = re.match(r"(\d+)_", p.name)
        if m:
            已有.append(int(m.group(1)))
    return f"{max(已有, default=0) + 1:03d}"


csv序号 = 取下一个序号(csv目录)
meta序号 = 取下一个序号(meta目录)
log序号 = 取下一个序号(log目录)
中文 = "P1_proto_V2_单样本三轴"
版本 = "V1"
out_glom = csv目录 / f"{csv序号}_{时间戳}-{中文}_glom_{版本}.csv"
out_tubule = csv目录 / f"{csv序号}_{时间戳}-{中文}_tubule_{版本}.csv"
out_artery = csv目录 / f"{csv序号}_{时间戳}-{中文}_artery_{版本}.csv"
out_meta = meta目录 / f"{meta序号}_{时间戳}-{中文}_{版本}.meta.json"
log文件 = log目录 / f"{log序号}_{时间戳}-{中文}_{版本}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log文件, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
日志 = logging.getLogger(__name__)


# ============================================================
# § 1 常量
# ============================================================
炎症标签_映射 = {
    "淋巴细胞": "lymph",
    "中性粒细胞": "neutro",
    "非典型中性粒细胞": "neutro",
    "浆细胞": "plasma",
    "嗜酸性-典型": "eosino",
    "嗜酸性-困难": "eosino",
    "嗜酸性-三核": "eosino",
}

W_10类 = ["W0", "W10", "W11", "W12", "W13", "W20", "W21", "W22", "W3", "W4"]
W粗_to_int = {"W0": 0, "W1": 1, "W2": 2, "W3": 3, "W4": 4}

KT1_5类_映射 = {
    "KT1_D1": "PT", "KT1_D2": "DT", "KT1_D3": "TAL",
    "KT1_D4": "CD", "KT1_D5": "异常",
}
KT1_5类_列 = ["PT", "DT", "TAL", "CD", "异常"]


def bbx_center(bbx):
    if not bbx or len(bbx) < 4:
        return None
    return (bbx[0] + bbx[2] / 2.0, bbx[1] + bbx[3] / 2.0)


def 粗W_to_int_KT(label: str) -> int:
    if not label:
        return -1
    m = re.search(r"KT1_W(\d)$", label)
    if m:
        return int(m.group(1))
    return -1


def 细W_label_KT(label: str) -> str:
    if not label:
        return ""
    m = re.search(r"KT1_W(\d+)$", label)
    if not m:
        return ""
    return "W" + m.group(1)


def 粗W_to_int_KG(label: str) -> int:
    """KG1_W0/1/2/3/4 → 0/1/2/3/4"""
    if not label:
        return -1
    m = re.search(r"KG1_W(\d)$", label)
    if m:
        return int(m.group(1))
    return -1


# ============================================================
# § 2 polygon 几何
# ============================================================
def polygon_geometry(points, mpp: float):
    if points is None or len(points) < 3:
        return {"area_um2": np.nan, "perimeter_um": np.nan,
                "circularity": np.nan, "solidity": np.nan,
                "eccentricity": np.nan, "elongation": np.nan,
                "major_axis_um": np.nan, "minor_axis_um": np.nan}
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
    except Exception:
        return {"area_um2": np.nan, "perimeter_um": np.nan,
                "circularity": np.nan, "solidity": np.nan,
                "eccentricity": np.nan, "elongation": np.nan,
                "major_axis_um": np.nan, "minor_axis_um": np.nan}

    try:
        coords = [(p[0], p[1]) for p in points if len(p) >= 2]
        if len(coords) < 3:
            raise ValueError("少于 3 点")
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = make_valid(poly)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        area_px = poly.area
        per_px = poly.length
        area_um2 = area_px * mpp * mpp
        per_um = per_px * mpp
        circ = (4 * math.pi * area_px / (per_px ** 2)) if per_px > 0 else np.nan
        hull = poly.convex_hull
        sol = (area_px / hull.area) if hull.area > 0 else np.nan
    except Exception:
        return {"area_um2": np.nan, "perimeter_um": np.nan,
                "circularity": np.nan, "solidity": np.nan,
                "eccentricity": np.nan, "elongation": np.nan,
                "major_axis_um": np.nan, "minor_axis_um": np.nan}

    try:
        xy = np.asarray(coords)
        cov = np.cov(xy, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov)
        if eigvals[1] <= 0:
            ecc = np.nan; elong = np.nan; major = np.nan; minor = np.nan
        else:
            minor = 2 * math.sqrt(max(eigvals[0], 0)) * mpp
            major = 2 * math.sqrt(eigvals[1]) * mpp
            ratio = eigvals[0] / eigvals[1] if eigvals[1] > 0 else 1.0
            ecc = math.sqrt(max(1 - ratio, 0))
            elong = 1 - (minor / major) if major > 0 else np.nan
    except Exception:
        ecc = np.nan; elong = np.nan; major = np.nan; minor = np.nan

    return {
        "area_um2": area_um2, "perimeter_um": per_um,
        "circularity": circ, "solidity": sol,
        "eccentricity": ecc, "elongation": elong,
        "major_axis_um": major, "minor_axis_um": minor,
    }


# ============================================================
# § 3 加载所有 JSON + 建全局 id 索引
# ============================================================
def 读json(name):
    p = alg_root / name
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)


def 加载全部():
    """返回 (K_main, nodes_dict, id_to_kId_global, id_to_node_global, id_to_label_global)"""
    json_files = sorted([p for p in alg_root.iterdir() if p.suffix == ".json"])
    日志.info(f"扫描 {len(json_files)} 个 JSON 文件")

    nodes_dict = {}
    id_to_kId = {}
    id_to_node = {}
    id_to_label = {}
    total = 0
    for p in json_files:
        name = p.stem
        try:
            d = json.load(open(p))
        except Exception as e:
            日志.warning(f"  跳过 {p.name}: {e}")
            continue
        nodes_dict[name] = d
        if not d:
            continue
        for it in d:
            iid = it.get("id", "")
            if not iid:
                continue
            id_to_node[iid] = name
            id_to_label[iid] = it.get("label", "")
            kid = it.get("kId", "")
            if kid:
                id_to_kId[iid] = kid
            total += 1
    日志.info(f"  扫描 {total} items, 唯一 id = {len(id_to_node)}")

    K_main = nodes_dict.get("K", [])
    return K_main, nodes_dict, id_to_kId, id_to_node, id_to_label


# ============================================================
# § 4 通用 kId 多层追溯
# ============================================================
def trace_to_root(start_id: str, id_to_kId: dict, target_ids: set,
                  max_depth: int = 5) -> str:
    """从 start_id 一路 kId 追溯, 找到 target_ids 中的第一个匹配, 返回那个 id (或 None)"""
    current = start_id
    visited = set()
    for _ in range(max_depth):
        if current in target_ids:
            return current
        if current in visited:
            return None
        visited.add(current)
        nxt = id_to_kId.get(current)
        if not nxt:
            return None
        current = nxt
    return current if current in target_ids else None


def 构建_label_node_到_KG1_映射(node_name: str, nodes_dict: dict,
                                  id_to_kId: dict, KG1_ids: set):
    """对一个 label-only 节点, 返回 {KG1.id → set of (item_id, item_label)} 及命中率统计"""
    result = defaultdict(list)
    items = nodes_dict.get(node_name, [])
    n_total = len(items)
    n_hit = 0
    for it in items:
        iid = it.get("id", "")
        kid_start = it.get("kId", "")
        if not kid_start:
            continue
        # 多层追溯
        target = trace_to_root(kid_start, id_to_kId, KG1_ids, max_depth=5)
        if target is None:
            continue
        result[target].append((iid, it.get("label", "")))
        n_hit += 1
    return result, n_hit, n_total


# ============================================================
# § 5 AXIS-G  Glomerulus 特征 (~25 维, V2 多层 join)
# ============================================================
def 抽取_glom(K_main, nodes_dict, id_to_kId, mpp: float, KG1_xy_tree):
    KG1_items = [k for k in K_main if k.get("label") == "KG1"]
    KG1_ids = {k["id"] for k in KG1_items}
    日志.info(f"  KG1 数: {len(KG1_items)}")

    # § 5.1 各特征 label-node 多层 join 到 KG1.id
    KG_映射 = {}  # node_name → {KG1.id → [(item_id, item_label), ...]}
    for name in ["KG1_W", "KG1_W1", "KG1_W12",
                 "KG1_GBM", "KG1_GBM1",
                 "KG1_S", "KG1_SC", "KG1_SC2", "KG1_SCF",
                 "KG1_FSGS",
                 "KG1_CC", "KG1_CC0", "KG1_CC1",
                 "KG1_CR", "KG1_CR2", "KG1_CRX",
                 "KG1_N", "KG1_I", "KG1_I0",
                 "KG1_F", "KG1_EG", "KG1_SY",
                 "KG1_B", "KG1_BT", "KG1_BTX",
                 "KG1_CI", "KG1_CS", "KG1_G"]:
        mapping, n_hit, n_total = 构建_label_node_到_KG1_映射(
            name, nodes_dict, id_to_kId, KG1_ids)
        KG_映射[name] = mapping
        日志.info(f"  {name:12s}: {n_hit}/{n_total} hit ({n_hit*100/max(n_total,1):.0f}%) → 覆盖 {len(mapping)}/{len(KG1_items)} KG1")

    # § 5.2 mesangial W 0-4 映射 (优先 W12 > W1 > W细 > W粗)
    glom_W = {}  # KG1.id → W_int (0-4)
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

    # § 5.3 build feature rows per KG1 instance
    GBM_set = set(KG_映射["KG1_GBM"].keys()) | set(KG_映射["KG1_GBM1"].keys())
    glob_scler_set = (set(KG_映射["KG1_S"].keys()) | set(KG_映射["KG1_SC"].keys())
                      | set(KG_映射["KG1_SC2"].keys()) | set(KG_映射["KG1_SCF"].keys()))
    fsgs_set = set(KG_映射["KG1_FSGS"].keys())
    crescent_set = (set(KG_映射["KG1_CC"].keys()) | set(KG_映射["KG1_CC0"].keys())
                    | set(KG_映射["KG1_CC1"].keys()))
    crescent_strict = set(KG_映射["KG1_CC0"].keys()) | set(KG_映射["KG1_CC1"].keys())
    necrosis_set = set(KG_映射["KG1_N"].keys())
    effective_set = set(KG_映射["KG1_I"].keys()) | set(KG_映射["KG1_I0"].keys())
    broken_set = (set(KG_映射["KG1_CR"].keys()) | set(KG_映射["KG1_CR2"].keys())
                  | set(KG_映射["KG1_CRX"].keys()))
    bowman_caps_set = (set(KG_映射["KG1_B"].keys()) | set(KG_映射["KG1_BT"].keys())
                       | set(KG_映射["KG1_BTX"].keys()))
    inflam_in_glom_set = set(KG_映射["KG1_CI"].keys())  # 球内炎症细胞 (有就标)

    rows = []
    for idx, k in enumerate(KG1_items):
        K_id = k.get("id", "")
        bbx = k.get("bbx", [0, 0, 0, 0])
        center_px = bbx_center(bbx)
        if center_px is None:
            continue
        cx_um, cy_um = center_px[0] * mpp, center_px[1] * mpp

        geom = polygon_geometry(k.get("points"), mpp)
        bbx_w_um = bbx[2] * mpp
        bbx_h_um = bbx[3] * mpp
        bbx_aspect = (max(bbx_w_um, bbx_h_um) /
                      max(min(bbx_w_um, bbx_h_um), 1e-6))
        dia_outer_um = float(k.get("diameter_outer", 0)) * mpp
        dia_inner_um = float(k.get("diameter_inner", 0)) * mpp

        tuft_area_um2 = math.pi * (dia_inner_um / 2) ** 2 if dia_inner_um > 0 else np.nan
        glom_area_um2 = geom["area_um2"] if not np.isnan(geom["area_um2"]) else (math.pi * (dia_outer_um / 2) ** 2 if dia_outer_um > 0 else np.nan)
        tuft_area_fraction = (tuft_area_um2 / glom_area_um2) if (glom_area_um2 and not np.isnan(tuft_area_um2)) else np.nan
        bowman_area_um2 = (glom_area_um2 - tuft_area_um2) if (glom_area_um2 and not np.isnan(tuft_area_um2)) else np.nan

        W_int = glom_W.get(K_id, -1)
        W_top = f"W{W_int}" if W_int >= 0 else ""
        W_oh = {f"glom_is_W{i}": int(W_int == i) for i in range(5)}

        if KG1_xy_tree is not None and KG1_xy_tree.n > 1:
            dists, _ = KG1_xy_tree.query([cx_um, cy_um], k=2)
            dist_nearest = float(dists[1])
        else:
            dist_nearest = np.nan

        # GBM 区域计数 (有可能一个球有 multi GBM patches)
        n_GBM = len(KG_映射["KG1_GBM"].get(K_id, [])) + len(KG_映射["KG1_GBM1"].get(K_id, []))
        n_FSGS = len(KG_映射["KG1_FSGS"].get(K_id, []))
        n_CC0 = len(KG_映射["KG1_CC0"].get(K_id, []))
        n_CI = len(KG_映射["KG1_CI"].get(K_id, []))  # 球内炎症细胞 count

        row = {
            "sample": 样本ID,
            "glom_id": K_id,
            "glom_idx_in_sample": idx,
            "cx_um": cx_um, "cy_um": cy_um,
            "bbx_w_um": bbx_w_um, "bbx_h_um": bbx_h_um, "bbx_aspect": bbx_aspect,
            "glom_dia_outer_um": dia_outer_um,
            "tuft_dia_inner_um": dia_inner_um,
            "glom_area_um2": glom_area_um2,
            "tuft_area_um2": tuft_area_um2,
            "tuft_area_fraction": tuft_area_fraction,
            "bowman_area_um2": bowman_area_um2,
            "perimeter_um": geom["perimeter_um"],
            "circularity": geom["circularity"],
            "solidity": geom["solidity"],
            "eccentricity": geom["eccentricity"],
            "elongation": geom["elongation"],
            "major_axis_um": geom["major_axis_um"],
            "minor_axis_um": geom["minor_axis_um"],
            # mesangial (Tervaert key)
            "mesangial_W_top": W_top,
            "mesangial_W_int": W_int,
            **W_oh,
            # DKD 特异
            "has_GBM_thicken": int(K_id in GBM_set),
            "n_GBM_patches": n_GBM,
            "has_global_sclerosis": int(K_id in glob_scler_set),
            "has_segmental_sclerosis_FSGS": int(K_id in fsgs_set),
            "n_FSGS_patches": n_FSGS,
            "has_crescent": int(K_id in crescent_set),
            "has_crescent_strict": int(K_id in crescent_strict),
            "n_crescent0_patches": n_CC0,
            "has_necrosis": int(K_id in necrosis_set),
            "is_effective_glom": int(K_id in effective_set),
            "has_broken_capsule": int(K_id in broken_set),
            "has_bowman_caps": int(K_id in bowman_caps_set),
            "n_intraglom_inflam_cells": n_CI,
            "has_intraglom_inflam": int(K_id in inflam_in_glom_set),
            "dist_nearest_glom_um": dist_nearest,
            "mpp": mpp,
        }
        rows.append(row)

    return rows


# ============================================================
# § 6 AXIS-TI  Tubule 特征 (~22 维, 复用 V1 逻辑)
# ============================================================
def 抽取_tubule(K_main, nodes_dict, mpp: float,
                inflam_trees, KT1_W_int_arr, KT1_xy_arr,
                KG1_xy_tree):
    KT1_items = [k for k in K_main if k.get("label") == "KT1"]
    日志.info(f"  KT1 数: {len(KT1_items)}")

    K_id_to_W粗 = {}
    K_id_to_W细 = {}
    for item in nodes_dict.get("KT1_W", []):
        kid = item.get("kId", "")
        if not kid:
            continue
        lbl = item.get("label", "")
        粗 = 粗W_to_int_KT(lbl)
        if 粗 >= 0:
            K_id_to_W粗[kid] = f"W{粗}"
            K_id_to_W细[kid] = f"W{粗}"
    for item in nodes_dict.get("KT1_W1", []) + nodes_dict.get("KT1_W2", []):
        kid = item.get("kId", "")
        if not kid:
            continue
        细 = 细W_label_KT(item.get("label", ""))
        if 细 in W_10类:
            K_id_to_W细[kid] = 细

    K_id_to_KT1class = {}
    for item in nodes_dict.get("KT1_D", []):
        kid = item.get("kId", "")
        if not kid:
            continue
        K_id_to_KT1class[kid] = KT1_5类_映射.get(item.get("label", ""), "异常")

    K_id_to_tub_n = Counter()
    for item in nodes_dict.get("KT1_CI", []):
        kid = item.get("kId", "")
        if kid:
            K_id_to_tub_n[kid] += 1

    tree_KT1 = cKDTree(KT1_xy_arr) if len(KT1_xy_arr) > 0 else None
    radii = [25.0, 50.0, 100.0]
    inflam_types = ["lymph", "neutro", "plasma", "eosino"]

    rows = []
    for idx, k in enumerate(KT1_items):
        K_id = k.get("id", "")
        center_px = bbx_center(k.get("bbx"))
        if center_px is None:
            continue
        cx_um, cy_um = center_px[0] * mpp, center_px[1] * mpp

        bbx = k.get("bbx", [0, 0, 0, 0])
        bbx_w_um = bbx[2] * mpp
        bbx_h_um = bbx[3] * mpp
        bbx_aspect = max(bbx_w_um, bbx_h_um) / max(min(bbx_w_um, bbx_h_um), 1e-6)
        area_um2 = float(k.get("area", 0)) * mpp * mpp
        per_um = float(k.get("perimeter", 0)) * mpp
        circ = (4 * math.pi * area_um2 / (per_um ** 2)) if per_um > 0 else np.nan
        dia_outer_um = float(k.get("diameter_outer", 0)) * mpp
        dia_inner_um = float(k.get("diameter_inner", 0)) * mpp
        lumen_area = math.pi * (dia_inner_um / 2) ** 2 if dia_inner_um > 0 else np.nan
        lumen_frac = (lumen_area / area_um2) if (area_um2 > 0 and not np.isnan(lumen_area)) else np.nan

        W_top = K_id_to_W细.get(K_id, "")
        粗_lbl = K_id_to_W粗.get(K_id, "")
        W_int = W粗_to_int.get(粗_lbl, -1)
        W_oh = {f"tub_is_{w}": int(W_top == w) for w in W_10类}

        KT1_class = K_id_to_KT1class.get(K_id, "未知")
        KT1_oh = {f"tub_is_{c}": int(KT1_class == c) for c in KT1_5类_列}

        n_tubitis = K_id_to_tub_n.get(K_id, 0)
        is_tubulitis = int(n_tubitis > 0)

        center = np.array([cx_um, cy_um])
        neigh = {}
        for r in radii:
            total = 0
            for t in inflam_types:
                tree = inflam_trees.get(t)
                n = len(tree.query_ball_point(center, r=r)) if tree is not None else 0
                neigh[f"n_{t}_{int(r)}"] = n
                total += n
            neigh[f"n_inflam_total_{int(r)}"] = total

        if tree_KT1 is not None and tree_KT1.n > 1:
            dists, _ = tree_KT1.query(center, k=2)
            dist_nearest_KT = float(dists[1])
            ids50 = [i for i in tree_KT1.query_ball_point(center, r=50.0) if i != idx]
            n_neigh_50 = len(ids50)
            if ids50:
                w = KT1_W_int_arr[ids50]
                v = w[w >= 0]
                avg_W = float(v.mean()) if len(v) > 0 else np.nan
            else:
                avg_W = np.nan
        else:
            dist_nearest_KT = np.nan; n_neigh_50 = 0; avg_W = np.nan

        if KG1_xy_tree is not None and KG1_xy_tree.n > 0:
            d, _ = KG1_xy_tree.query(center, k=1)
            dist_nearest_glom = float(d)
        else:
            dist_nearest_glom = np.nan

        rows.append({
            "sample": 样本ID, "tubule_id": K_id, "tub_idx_in_sample": idx,
            "cx_um": cx_um, "cy_um": cy_um,
            "bbx_w_um": bbx_w_um, "bbx_h_um": bbx_h_um, "bbx_aspect": bbx_aspect,
            "tub_area_um2": area_um2, "tub_perimeter_um": per_um,
            "tub_circularity": circ,
            "tub_dia_outer_um": dia_outer_um, "tub_dia_inner_um": dia_inner_um,
            "tub_lumen_area_um2": lumen_area, "tub_lumen_area_fraction": lumen_frac,
            "tub_W_top": W_top, "tub_W_int": W_int, **W_oh,
            "tub_KT1_class": KT1_class, **KT1_oh,
            "tub_is_tubulitis": is_tubulitis, "tub_n_tubulitis_cells": int(n_tubitis),
            **neigh,
            "dist_nearest_tubule_um": dist_nearest_KT,
            "n_neighbor_tubule_50": int(n_neigh_50),
            "avg_W_int_neighbor_50": avg_W,
            "dist_nearest_glom_um": dist_nearest_glom,
            "mpp": mpp,
        })
    return rows


# ============================================================
# § 7 AXIS-V  Artery 特征 (~10 维)
# ============================================================
def 抽取_artery(K_main, nodes_dict, id_to_kId, mpp: float, KG1_xy_tree):
    KV1_items = [k for k in K_main if k.get("label") == "KV1"]
    KV1_ids = {k["id"] for k in KV1_items}
    日志.info(f"  KV1 数: {len(KV1_items)}")

    KV_映射 = {}
    for name in ["KV11_W", "KV11_W0", "KV11_W1"]:
        mapping = defaultdict(list)
        n_hit = 0; n_total = 0
        for it in nodes_dict.get(name, []):
            n_total += 1
            kid = it.get("kId", "")
            if not kid:
                continue
            target = trace_to_root(kid, id_to_kId, KV1_ids, max_depth=5)
            if target is None:
                continue
            mapping[target].append((it.get("id", ""), it.get("label", "")))
            n_hit += 1
        KV_映射[name] = mapping
        日志.info(f"  {name:12s}: {n_hit}/{n_total} hit → 覆盖 {len(mapping)}/{len(KV1_items)} KV1")

    rows = []
    for idx, k in enumerate(KV1_items):
        K_id = k.get("id", "")
        center_px = bbx_center(k.get("bbx"))
        if center_px is None:
            continue
        cx_um, cy_um = center_px[0] * mpp, center_px[1] * mpp
        bbx = k.get("bbx", [0, 0, 0, 0])

        area_um2 = float(k.get("area", 0)) * mpp * mpp
        per_um = float(k.get("perimeter", 0)) * mpp
        circ = (4 * math.pi * area_um2 / (per_um ** 2)) if per_um > 0 else np.nan
        dia_outer_um = float(k.get("diameter_outer", 0)) * mpp
        dia_inner_um = float(k.get("diameter_inner", 0)) * mpp
        wall_dia = (dia_outer_um - dia_inner_um) / 2.0 if dia_outer_um > dia_inner_um else np.nan
        wall_lumen_ratio = (wall_dia / dia_inner_um) if dia_inner_um > 0 else np.nan
        is_small = int(dia_outer_um < 50)

        KV11_W = KV_映射["KV11_W"].get(K_id, [])
        KV11_W_label = KV11_W[0][1] if KV11_W else ""

        if KG1_xy_tree is not None and KG1_xy_tree.n > 0:
            d, _ = KG1_xy_tree.query([cx_um, cy_um], k=1)
            dist_glom = float(d)
        else:
            dist_glom = np.nan

        rows.append({
            "sample": 样本ID, "artery_id": K_id, "art_idx_in_sample": idx,
            "cx_um": cx_um, "cy_um": cy_um,
            "art_area_um2": area_um2, "art_perimeter_um": per_um,
            "art_circularity": circ,
            "art_dia_outer_um": dia_outer_um, "art_dia_inner_um": dia_inner_um,
            "wall_diameter_um": wall_dia,
            "wall_lumen_ratio": wall_lumen_ratio,
            "is_small_arteriole": is_small,
            "KV11_subclass": KV11_W_label,
            "dist_nearest_glom_um": dist_glom,
            "mpp": mpp,
        })
    return rows


# ============================================================
# § 8 main
# ============================================================
def main():
    日志.info("=" * 72)
    日志.info(f"V5 P1 prototype V2 - 单样本 {样本ID} 三轴特征提取 (多层 kId 追溯)")
    日志.info(f"  alg 根: {alg_root}")
    日志.info("=" * 72)

    t0 = time.time()
    K_main, nodes_dict, id_to_kId, id_to_node, id_to_label = 加载全部()
    cnt = Counter(k.get("label", "?") for k in K_main)
    日志.info(f"K.json: {dict(cnt)} | 总 items = {len(K_main)}")

    mpp = 0.2518  # default
    日志.info(f"mpp = {mpp} (默认 Aperio scan)")

    # 炎症细胞 trees
    cell_id_to_C = {c["id"]: c for c in nodes_dict.get("KI1_C", [])}
    炎症_by_type = defaultdict(list)
    for cc in nodes_dict.get("KI1_CC", []):
        parent_id = cc.get("kId", "")
        c = cell_id_to_C.get(parent_id)
        if c is None:
            continue
        raw = cc.get("label", "")
        if raw == "非炎症细胞":
            continue
        type5 = 炎症标签_映射.get(raw)
        if type5 is None:
            continue
        center = bbx_center(c.get("bbx"))
        if center is None:
            continue
        炎症_by_type[type5].append((center[0] * mpp, center[1] * mpp))
    inflam_trees = {t: cKDTree(np.array(lst)) if lst else None
                    for t, lst in 炎症_by_type.items()}
    for t in ["lymph", "neutro", "plasma", "eosino"]:
        日志.info(f"  inflam {t:7s} count = {len(炎症_by_type.get(t, []))}")

    # KT1 / KG1 中心
    KT1_items = [k for k in K_main if k.get("label") == "KT1"]
    KT1_id_to_W粗 = {}
    for item in nodes_dict.get("KT1_W", []):
        kid = item.get("kId", "")
        if kid:
            粗 = 粗W_to_int_KT(item.get("label", ""))
            if 粗 >= 0:
                KT1_id_to_W粗[kid] = 粗
    KT1_xys = []
    KT1_W_ints = []
    for k in KT1_items:
        c = bbx_center(k.get("bbx"))
        if c is None: continue
        KT1_xys.append((c[0] * mpp, c[1] * mpp))
        KT1_W_ints.append(KT1_id_to_W粗.get(k["id"], -1))
    KT1_xy_arr = np.array(KT1_xys) if KT1_xys else np.zeros((0, 2))
    KT1_W_int_arr = np.array(KT1_W_ints) if KT1_W_ints else np.array([], dtype=int)

    KG1_items = [k for k in K_main if k.get("label") == "KG1"]
    KG1_xys = []
    for k in KG1_items:
        c = bbx_center(k.get("bbx"))
        if c is not None:
            KG1_xys.append((c[0] * mpp, c[1] * mpp))
    KG1_xy_arr = np.array(KG1_xys) if KG1_xys else np.zeros((0, 2))
    KG1_xy_tree = cKDTree(KG1_xy_arr) if len(KG1_xy_arr) > 0 else None

    日志.info("\n[AXIS-G] V2 多层 kId 追溯 + glom 特征 ...")
    rows_g = 抽取_glom(K_main, nodes_dict, id_to_kId, mpp, KG1_xy_tree)
    df_g = pd.DataFrame(rows_g)
    日志.info(f"  glom DF shape = {df_g.shape}")

    日志.info("\n[AXIS-TI] tubule 特征 ...")
    rows_t = 抽取_tubule(K_main, nodes_dict, mpp, inflam_trees,
                         KT1_W_int_arr, KT1_xy_arr, KG1_xy_tree)
    df_t = pd.DataFrame(rows_t)
    日志.info(f"  tubule DF shape = {df_t.shape}")

    日志.info("\n[AXIS-V] artery 特征 (V2 多层 join) ...")
    rows_v = 抽取_artery(K_main, nodes_dict, id_to_kId, mpp, KG1_xy_tree)
    df_v = pd.DataFrame(rows_v)
    日志.info(f"  artery DF shape = {df_v.shape}")

    df_g.to_csv(out_glom, index=False, encoding="utf-8")
    df_t.to_csv(out_tubule, index=False, encoding="utf-8")
    df_v.to_csv(out_artery, index=False, encoding="utf-8")
    日志.info(f"\n落盘 glom   → {out_glom}")
    日志.info(f"落盘 tubule → {out_tubule}")
    日志.info(f"落盘 artery → {out_artery}")

    日志.info("\n========= V2 sanity check (重点对比 V1) =========")
    日志.info(f"glom: n={len(df_g)} | area median={df_g['glom_area_um2'].median():.0f} μm² | "
              f"circularity median={df_g['circularity'].median():.3f}")
    日志.info(f"  W (mesangial) 0-4 分布: " +
              ", ".join(f"W{i}={int((df_g['mesangial_W_int']==i).sum())}" for i in range(5)) +
              f", 未映射={int((df_g['mesangial_W_int']==-1).sum())}")
    日志.info(f"  特征命中率 (V2 期望提升):")
    for feat in ["has_GBM_thicken", "has_global_sclerosis", "has_segmental_sclerosis_FSGS",
                 "has_crescent", "has_crescent_strict", "has_necrosis", "is_effective_glom",
                 "has_broken_capsule", "has_bowman_caps", "has_intraglom_inflam"]:
        c = int(df_g[feat].sum())
        日志.info(f"    {feat:30s}: {c}/{len(df_g)}  ({c*100/len(df_g):.0f}%)")
    日志.info(f"  其他 patch 计数:")
    for col in ["n_GBM_patches", "n_FSGS_patches", "n_crescent0_patches",
                "n_intraglom_inflam_cells"]:
        v = df_g[col]
        日志.info(f"    {col:30s}: median={v.median():.1f}  max={v.max():.0f}")

    日志.info(f"\ntubule: n={len(df_t)} | tubulitis pos={int(df_t['tub_is_tubulitis'].sum())}")
    日志.info(f"artery: n={len(df_v)} | wall_lumen_ratio median={df_v['wall_lumen_ratio'].median():.2f}")

    meta = {
        "脚本": str(Path(__file__).name),
        "运行时间戳": 时间戳,
        "样本": 样本ID,
        "mpp": mpp,
        "V2修复说明": "多层 kId 追溯, 修复 B 类 KG1_* 节点 join 0 问题",
        "K_label_分布": dict(cnt),
        "glom_shape": list(df_g.shape),
        "tubule_shape": list(df_t.shape),
        "artery_shape": list(df_v.shape),
        "运行时间_s": time.time() - t0,
        "host": __import__("socket").gethostname(),
        "python_version": sys.version.split()[0],
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    日志.info(f"\nmeta → {out_meta}")
    日志.info(f"DONE. 总耗时 {time.time()-t0:.1f}s")
    日志.info("=" * 72)


if __name__ == "__main__":
    main()
