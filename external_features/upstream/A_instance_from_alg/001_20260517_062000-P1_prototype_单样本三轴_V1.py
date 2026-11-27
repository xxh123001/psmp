#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本: 001_20260517_062000-P1_prototype_单样本三轴_V1.py
功能: V5 NGM-DKD-PAS P1 prototype - 在 Mac 单样本 (1_S_5) 验证三轴特征提取链路

三轴:
  AXIS-G  Glomerulus  (KG1 in K.json + KG1_W/W1/W12/GBM/SC/FSGS/CC join)  ~25 维
  AXIS-TI Tubule       (KT1 in K.json + KT1_W/W1/W2/D/CI join, V4 复用思路)  ~22 维
  AXIS-V  Vessel       (KV1 in K.json + KV11_/KV13_ join)                     ~10 维

输出:
  009_csv/{NNN}_{ts}-P1_proto_1_S_5_glom_V1.csv       (~44 行)
  009_csv/{NNN}_{ts}-P1_proto_1_S_5_tubule_V1.csv     (~1534 行)
  009_csv/{NNN}_{ts}-P1_proto_1_S_5_artery_V1.csv     (~807 行)
  008_meta/{NNN}_{ts}-P1_proto_*.meta.json
  013_log/{NNN}_{ts}-P1_proto_*.log

不并发, 单进程, 估时 < 30 s
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

样本根 = 项目根 / "011_data" / "raw" / "alg_results_86样本"
样本ID = "1_S_5"  # prototype 单例
样本目录 = 样本根 / f"{样本ID}_alg_result"


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
中文 = "P1_proto_单样本三轴"
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

# tubule W 10 类 (V4 V2 修正版)
W_10类 = ["W0", "W10", "W11", "W12", "W13", "W20", "W21", "W22", "W3", "W4"]
W粗_to_int = {"W0": 0, "W1": 1, "W2": 2, "W3": 3, "W4": 4}

# tubule KT1 5 类
KT1_5类_映射 = {
    "KT1_D1": "PT", "KT1_D2": "DT", "KT1_D3": "TAL",
    "KT1_D4": "CD", "KT1_D5": "异常",
}
KT1_5类_列 = ["PT", "DT", "TAL", "CD", "异常"]


def bbx_center(bbx):
    if not bbx or len(bbx) < 4:
        return None
    return (bbx[0] + bbx[2] / 2.0, bbx[1] + bbx[3] / 2.0)


def 粗W_to_int(label: str) -> int:
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


# ============================================================
# § 2 polygon 几何 / circularity / eccentricity / solidity
# ============================================================
def polygon_geometry(points, mpp: float):
    """
    输入: chain4 polygon points = [[x,y],[x,y],...] (px)
    输出: dict(area_um2, perimeter_um, circularity, solidity, eccentricity, elongation)
    若 points 无效 → 返回全 NaN
    """
    if points is None or len(points) < 3:
        return {"area_um2": np.nan, "perimeter_um": np.nan,
                "circularity": np.nan, "solidity": np.nan,
                "eccentricity": np.nan, "elongation": np.nan,
                "major_axis_um": np.nan, "minor_axis_um": np.nan}
    try:
        from shapely.geometry import Polygon, MultiPoint
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
            # 取最大
            poly = max(poly.geoms, key=lambda g: g.area)
        area_px = poly.area
        per_px = poly.length
        area_um2 = area_px * mpp * mpp
        per_um = per_px * mpp
        circ = (4 * math.pi * area_px / (per_px ** 2)) if per_px > 0 else np.nan
        hull = poly.convex_hull
        sol = (area_px / hull.area) if hull.area > 0 else np.nan
    except Exception as e:
        return {"area_um2": np.nan, "perimeter_um": np.nan,
                "circularity": np.nan, "solidity": np.nan,
                "eccentricity": np.nan, "elongation": np.nan,
                "major_axis_um": np.nan, "minor_axis_um": np.nan}

    # 椭圆拟合 → eccentricity & elongation
    try:
        xy = np.asarray(coords)
        cov = np.cov(xy, rowvar=False)
        eigvals, _ = np.linalg.eigh(cov)  # 升序
        if eigvals[1] <= 0:
            ecc = np.nan
            elong = np.nan
            major = np.nan
            minor = np.nan
        else:
            # major / minor 半轴长 (px) ≈ 2 * sqrt(eigval)
            minor = 2 * math.sqrt(max(eigvals[0], 0)) * mpp
            major = 2 * math.sqrt(eigvals[1]) * mpp
            ratio = eigvals[0] / eigvals[1] if eigvals[1] > 0 else 1.0
            ecc = math.sqrt(max(1 - ratio, 0))   # 0=圆 1=线
            elong = 1 - (minor / major) if major > 0 else np.nan
    except Exception:
        ecc = np.nan
        elong = np.nan
        major = np.nan
        minor = np.nan

    return {
        "area_um2": area_um2,
        "perimeter_um": per_um,
        "circularity": circ,
        "solidity": sol,
        "eccentricity": ecc,
        "elongation": elong,
        "major_axis_um": major,
        "minor_axis_um": minor,
    }


# ============================================================
# § 3 加载 chain4 JSON
# ============================================================
def 加载样本(样本目录: Path):
    """返回 (info, K_main, nodes)

    Mac zip 结构：
      样本目录 / hr0_alg.json / *.json           (无 info.json)
    Linux V4 结构：
      样本目录 / info.json
      样本目录 / alg_results / hr0_alg.json / *.json

    若 info.json 不存在 → 用默认 mpp=0.2518
    """
    info_path = 样本目录 / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        info = {"mpp": 0.2518, "size": "unknown", "_source": "default (no info.json)"}
        日志.warning(f"info.json 不存在, 使用默认 mpp=0.2518")

    # 尝试两种 alg 根目录
    alg_root_v4 = 样本目录 / "alg_results" / "hr0_alg.json"
    alg_root_mac = 样本目录 / "hr0_alg.json"
    if alg_root_v4.exists():
        alg_root = alg_root_v4
    elif alg_root_mac.exists():
        alg_root = alg_root_mac
    else:
        raise FileNotFoundError(f"alg_results 不存在: 试过 {alg_root_v4} 和 {alg_root_mac}")
    日志.info(f"  alg 根: {alg_root}")

    def 读json(name):
        p = alg_root / name
        if not p.exists():
            return []
        with open(p) as f:
            return json.load(f)

    K_main = 读json("K.json")
    nodes = {
        "K": K_main,
        # tubule
        "KT1_W": 读json("KT1_W.json"),
        "KT1_W1": 读json("KT1_W1.json"),
        "KT1_W2": 读json("KT1_W2.json"),
        "KT1_D": 读json("KT1_D.json"),
        "KT1_CI": 读json("KT1_CI.json"),
        # glom
        "KG1_W": 读json("KG1_W.json"),
        "KG1_W1": 读json("KG1_W1.json"),
        "KG1_W12": 读json("KG1_W12.json"),
        "KG1_GBM": 读json("KG1_GBM.json"),
        "KG1_GBM1": 读json("KG1_GBM1.json"),
        "KG1_S": 读json("KG1_S.json"),
        "KG1_SC": 读json("KG1_SC.json"),
        "KG1_SC2": 读json("KG1_SC2.json"),
        "KG1_SCF": 读json("KG1_SCF.json"),
        "KG1_FSGS": 读json("KG1_FSGS.json"),
        "KG1_CC": 读json("KG1_CC.json"),
        "KG1_CC0": 读json("KG1_CC0.json"),
        "KG1_CC1": 读json("KG1_CC1.json"),
        "KG1_N": 读json("KG1_N.json"),
        "KG1_I": 读json("KG1_I.json"),
        "KG1_I0": 读json("KG1_I0.json"),
        "KG1_F": 读json("KG1_F.json"),
        # vessel
        "KV1": 读json("KV1.json"),
        "KV11_W": 读json("KV11_W.json"),
        "KV11_W0": 读json("KV11_W0.json"),
        "KV11_W1": 读json("KV11_W1.json"),
        # inflam cells
        "KI1_C": 读json("KI1_C.json"),
        "KI1_CC": 读json("KI1_CC.json"),
    }
    return info, K_main, nodes


# ============================================================
# § 4 AXIS-G  Glomerulus 特征 (~25 维)
# ============================================================
def 抽取_glom(K_main, nodes, mpp: float, KG1_xy_tree):
    KG1_items = [k for k in K_main if k.get("label") == "KG1"]
    日志.info(f"  KG1 数: {len(KG1_items)}")

    def kId_to_label(node_name):
        return {item.get("kId", ""): item.get("label", "") for item in nodes.get(node_name, [])}

    def kId_present(node_name):
        return {item.get("kId", "") for item in nodes.get(node_name, []) if item.get("kId")}

    KG1_id_to_W = {}
    for item in nodes.get("KG1_W", []):
        kid = item.get("kId", "")
        lbl = item.get("label", "")
        m = re.search(r"KG1_W(\d)$", lbl)
        if kid and m:
            KG1_id_to_W[kid] = int(m.group(1))

    GBM_set = kId_present("KG1_GBM") | kId_present("KG1_GBM1")
    glob_scler_set = kId_present("KG1_S") | kId_present("KG1_SC") | kId_present("KG1_SC2") | kId_present("KG1_SCF")
    fsgs_set = kId_present("KG1_FSGS")
    crescent_set = kId_present("KG1_CC") | kId_present("KG1_CC0") | kId_present("KG1_CC1")
    necrosis_set = kId_present("KG1_N")
    effective_set = kId_present("KG1_I") | kId_present("KG1_I0")

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
        dia_outer_um = float(k.get("diameter_outer", 0)) * mpp  # full glom
        dia_inner_um = float(k.get("diameter_inner", 0)) * mpp  # tuft (chain4)

        # tuft / bowman
        tuft_area_um2 = math.pi * (dia_inner_um / 2) ** 2 if dia_inner_um > 0 else np.nan
        glom_area_um2 = geom["area_um2"] if not np.isnan(geom["area_um2"]) else (math.pi * (dia_outer_um / 2) ** 2)
        tuft_area_fraction = (tuft_area_um2 / glom_area_um2) if (glom_area_um2 and not np.isnan(tuft_area_um2)) else np.nan
        bowman_area_um2 = (glom_area_um2 - tuft_area_um2) if (glom_area_um2 and not np.isnan(tuft_area_um2)) else np.nan

        W_int = KG1_id_to_W.get(K_id, -1)
        W_top = f"W{W_int}" if W_int >= 0 else ""
        W_oh = {f"glom_is_W{i}": int(W_int == i) for i in range(5)}

        # 最近邻
        if KG1_xy_tree is not None and KG1_xy_tree.n > 1:
            dists, _ = KG1_xy_tree.query([cx_um, cy_um], k=2)
            dist_nearest = float(dists[1])
        else:
            dist_nearest = np.nan

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
            "mesangial_W_top": W_top,
            "mesangial_W_int": W_int,
            **W_oh,
            "has_GBM_thicken": int(K_id in GBM_set),
            "has_global_sclerosis": int(K_id in glob_scler_set),
            "has_segmental_sclerosis_FSGS": int(K_id in fsgs_set),
            "has_crescent": int(K_id in crescent_set),
            "has_necrosis": int(K_id in necrosis_set),
            "is_effective_glom": int(K_id in effective_set),
            "dist_nearest_glom_um": dist_nearest,
            "mpp": mpp,
        }
        rows.append(row)

    return rows


# ============================================================
# § 5 AXIS-TI  Tubule 特征 (~22 维, 简化版 V4 step6 V2)
# ============================================================
def 抽取_tubule(K_main, nodes, mpp: float,
                inflam_trees, KT1_W_int_arr, KT1_xy_arr,
                KG1_xy_tree):
    KT1_items = [k for k in K_main if k.get("label") == "KT1"]
    日志.info(f"  KT1 数: {len(KT1_items)}")

    K_id_to_W粗 = {}
    K_id_to_W细 = {}
    for item in nodes.get("KT1_W", []):
        kid = item.get("kId", "")
        if not kid:
            continue
        lbl = item.get("label", "")
        粗 = 粗W_to_int(lbl)
        if 粗 >= 0:
            K_id_to_W粗[kid] = f"W{粗}"
            K_id_to_W细[kid] = f"W{粗}"
    for item in nodes.get("KT1_W1", []) + nodes.get("KT1_W2", []):
        kid = item.get("kId", "")
        if not kid:
            continue
        细 = 细W_label_KT(item.get("label", ""))
        if 细 in W_10类:
            K_id_to_W细[kid] = 细

    K_id_to_KT1class = {}
    for item in nodes.get("KT1_D", []):
        kid = item.get("kId", "")
        if not kid:
            continue
        K_id_to_KT1class[kid] = KT1_5类_映射.get(item.get("label", ""), "异常")

    K_id_to_tub_n = Counter()
    for item in nodes.get("KT1_CI", []):
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

        # 最近邻 tubule
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

        # 最近邻 glom
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
# § 6 AXIS-V  Vessel/Artery 特征 (~10 维, V5 全新)
# ============================================================
def 抽取_artery(K_main, nodes, mpp: float, KG1_xy_tree):
    KV1_items = [k for k in K_main if k.get("label") == "KV1"]
    日志.info(f"  KV1 数: {len(KV1_items)}")

    KV_id_to_W = {}
    for item in nodes.get("KV11_W", []):
        kid = item.get("kId", "")
        lbl = item.get("label", "")
        if kid and lbl:
            KV_id_to_W[kid] = lbl

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
        dia_outer_um = float(k.get("diameter_outer", 0)) * mpp  # full artery
        dia_inner_um = float(k.get("diameter_inner", 0)) * mpp  # lumen
        wall_dia = (dia_outer_um - dia_inner_um) / 2.0 if dia_outer_um > dia_inner_um else np.nan
        wall_lumen_ratio = (wall_dia / dia_inner_um) if dia_inner_um > 0 else np.nan
        is_small = int(dia_outer_um < 50)  # NGM 阈值

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
            "KV11_subclass": KV_id_to_W.get(K_id, ""),
            "dist_nearest_glom_um": dist_glom,
            "mpp": mpp,
        })
    return rows


# ============================================================
# § 7 主入口
# ============================================================
def main():
    日志.info("=" * 72)
    日志.info(f"V5 P1 prototype - 单样本 {样本ID} 三轴特征提取")
    日志.info(f"  样本目录: {样本目录}")
    日志.info(f"  输出 glom   : {out_glom.name}")
    日志.info(f"  输出 tubule : {out_tubule.name}")
    日志.info(f"  输出 artery : {out_artery.name}")
    日志.info("=" * 72)

    t0 = time.time()
    info, K_main, nodes = 加载样本(样本目录)
    mpp = float(info.get("mpp", 0.2518))
    日志.info(f"info.json: mpp = {mpp}, WSI 尺寸 = {info.get('size','?')}")
    日志.info(f"K.json 总 items = {len(K_main)}")
    cnt = Counter(k.get("label", "?") for k in K_main)
    日志.info(f"K.json label 分布 = {dict(cnt)}")

    # 预计算 inflam trees & KT1 数组
    cell_id_to_C = {c["id"]: c for c in nodes.get("KI1_C", [])}
    炎症_by_type = defaultdict(list)
    for cc in nodes.get("KI1_CC", []):
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

    # KT1 / KG1 / KV1 中心 & W_int arr
    KT1_items = [k for k in K_main if k.get("label") == "KT1"]
    KT1_id_to_W粗 = {}
    for item in nodes.get("KT1_W", []):
        kid = item.get("kId", "")
        if kid:
            粗 = 粗W_to_int(item.get("label", ""))
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

    # ============== 三轴 ==============
    日志.info("\n[AXIS-G] 抽取 glom 特征 ...")
    rows_g = 抽取_glom(K_main, nodes, mpp, KG1_xy_tree)
    df_g = pd.DataFrame(rows_g)
    日志.info(f"  glom DF shape = {df_g.shape}")

    日志.info("\n[AXIS-TI] 抽取 tubule 特征 ...")
    rows_t = 抽取_tubule(K_main, nodes, mpp, inflam_trees,
                         KT1_W_int_arr, KT1_xy_arr, KG1_xy_tree)
    df_t = pd.DataFrame(rows_t)
    日志.info(f"  tubule DF shape = {df_t.shape}")

    日志.info("\n[AXIS-V] 抽取 artery 特征 ...")
    rows_v = 抽取_artery(K_main, nodes, mpp, KG1_xy_tree)
    df_v = pd.DataFrame(rows_v)
    日志.info(f"  artery DF shape = {df_v.shape}")

    # ============== 落盘 ==============
    df_g.to_csv(out_glom, index=False, encoding="utf-8")
    df_t.to_csv(out_tubule, index=False, encoding="utf-8")
    df_v.to_csv(out_artery, index=False, encoding="utf-8")
    日志.info(f"\n落盘 glom   → {out_glom}")
    日志.info(f"落盘 tubule → {out_tubule}")
    日志.info(f"落盘 artery → {out_artery}")

    # ============== sanity check ==============
    日志.info("\n========= sanity check =========")
    日志.info(f"glom: n={len(df_g)} | area median={df_g['glom_area_um2'].median():.0f} μm² | "
              f"circularity median={df_g['circularity'].median():.3f}")
    日志.info(f"  W (mesangial) 0-4 分布: " +
              ", ".join(f"W{i}={int((df_g['mesangial_W_int']==i).sum())}" for i in range(5)) +
              f", 未映射={int((df_g['mesangial_W_int']==-1).sum())}")
    日志.info(f"  特征命中率: GBM={df_g['has_GBM_thicken'].sum()}, "
              f"global_scler={df_g['has_global_sclerosis'].sum()}, "
              f"FSGS={df_g['has_segmental_sclerosis_FSGS'].sum()}, "
              f"crescent={df_g['has_crescent'].sum()}, "
              f"necrosis={df_g['has_necrosis'].sum()}")
    日志.info(f"tubule: n={len(df_t)} | area median={df_t['tub_area_um2'].median():.0f} μm² | "
              f"tubulitis pos={int(df_t['tub_is_tubulitis'].sum())}")
    日志.info(f"  W 10 类分布:")
    for w in W_10类:
        c = int(df_t[f'tub_is_{w}'].sum())
        日志.info(f"    {w}: {c}")
    日志.info(f"  KT1 5 类分布:")
    for cls in KT1_5类_列:
        c = int(df_t[f'tub_is_{cls}'].sum())
        日志.info(f"    {cls}: {c}")
    日志.info(f"artery: n={len(df_v)} | wall_lumen_ratio median="
              f"{df_v['wall_lumen_ratio'].median():.2f} | small_arteriole={int(df_v['is_small_arteriole'].sum())}")

    # ============== meta ==============
    meta = {
        "脚本": str(Path(__file__).name),
        "运行时间戳": 时间戳,
        "样本": 样本ID,
        "mpp": mpp,
        "info_json": str(info),
        "K_label_分布": dict(cnt),
        "glom_shape": list(df_g.shape),
        "tubule_shape": list(df_t.shape),
        "artery_shape": list(df_v.shape),
        "输出_glom": str(out_glom),
        "输出_tubule": str(out_tubule),
        "输出_artery": str(out_artery),
        "运行时间_s": time.time() - t0,
        "host": __import__("socket").gethostname(),
        "python_version": sys.version.split()[0],
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    日志.info(f"\nmeta → {out_meta}")

    日志.info("=" * 72)
    日志.info(f"DONE. 总耗时 {time.time()-t0:.1f}s")
    日志.info("=" * 72)


if __name__ == "__main__":
    main()
