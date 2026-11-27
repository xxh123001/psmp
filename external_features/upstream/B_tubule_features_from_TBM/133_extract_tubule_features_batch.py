#!/usr/bin/env python3
"""
小管特征提取 V1 — 全部样本批量提取
Step 1: 小管级 CSV (每行=一个小管, ~119 列)
Step 2: 患者级 CSV (每行=一个患者, mean/std/skew/kurt 聚合)
过滤: 皮质(K_2) + 无破损 (W0/W1/W2)
"""
import json, math, warnings, os
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.spatial import cKDTree, ConvexHull
from scipy.stats import skew, kurtosis
from shapely.geometry import Polygon, Point
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════
# 工具函数 (与单样本版相同)
# ══════════════════════════════════════════════════════════

def load_json(fp):
    with open(fp) as f:
        return json.load(f)

def pts_to_np(item):
    """返回外轮廓点 (最大子 contour)。
    支持两种格式:
      1) [[x,y], [x,y], ...]  (小管外/内轮廓)
      2) flat 多 contour 格式: [n_contours, n_pts_1, x,y,..., n_pts_2, x,y,...]  (TBM)
    """
    pts = item.get("points", [])
    if not pts:
        return None
    if isinstance(pts[0], list):
        return np.array(pts, dtype=float)
    # flat 多 contour 格式
    contours = decode_multi_contour(pts)
    if not contours:
        return None
    # 返回点数最多 (即最长边界, 通常是外环)
    return max(contours, key=lambda c: len(c))

def decode_multi_contour(flat):
    """解码 [n_contours, c1_n, x,y..., c2_n, x,y...] 多边形多轮廓格式"""
    contours = []
    n_cont = int(flat[0])
    i = 1
    for _ in range(n_cont):
        if i >= len(flat):
            break
        n = int(flat[i]); i += 1
        if n < 3 or i + 2*n > len(flat):
            break
        pts = np.array(flat[i:i+2*n], dtype=float).reshape(-1, 2)
        contours.append(pts)
        i += 2*n
    return contours

def get_tbm_inner_outer(item):
    """对 TBM (ring 形) 返回 (outer_contour, inner_contour) 两条边界"""
    pts = item.get("points", [])
    if not pts or isinstance(pts[0], list):
        return None, None
    cs = decode_multi_contour(pts)
    cs = [c for c in cs if len(c) >= 3]
    if not cs:
        return None, None
    if len(cs) == 1:
        return cs[0], None
    # 按面积排序: 最大=outer, 第二大=inner_hole
    cs_sorted = sorted(cs, key=lambda c: polygon_area(c), reverse=True)
    return cs_sorted[0], cs_sorted[1]

def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_perimeter(pts):
    d = np.diff(pts, axis=0, append=pts[:1])
    return np.sqrt((d**2).sum(axis=1)).sum()

def polygon_centroid(pts):
    return pts.mean(axis=0)

def circularity_val(area, perimeter):
    if perimeter < 1e-9:
        return 0.0
    return 4 * math.pi * area / (perimeter ** 2)

def eccentricity_from_pts(pts):
    if len(pts) < 5:
        return np.nan
    c = pts.mean(axis=0)
    centered = pts - c
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    if eigvals[0] < 1e-9:
        return 0.0
    return np.sqrt(1 - eigvals[1] / eigvals[0])

def solidity_val(area, pts):
    if len(pts) < 3:
        return np.nan
    try:
        hull = ConvexHull(pts)
        return area / hull.volume if hull.volume > 0 else np.nan
    except:
        return np.nan

def convexity_val(perimeter, pts):
    if len(pts) < 3:
        return np.nan
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        hp = polygon_perimeter(hull_pts)
        return hp / perimeter if perimeter > 0 else np.nan
    except:
        return np.nan

def smoothness(pts, step):
    if pts is None:
        return np.nan
    n = len(pts)
    if n < step * 2:
        return np.nan
    angles = []
    for i in range(n):
        p0 = pts[(i - step) % n]
        p1 = pts[i]
        p2 = pts[(i + step) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        cos_a = np.clip(cos_a, -1, 1)
        angles.append(np.arccos(cos_a))
    return np.std(angles)

def fourier_descriptors(pts, n_harmonics=10):
    if pts is None or len(pts) < 10:
        return [np.nan] * n_harmonics
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - z.mean()
    fft_vals = np.fft.fft(z)
    magnitudes = np.abs(fft_vals)
    if magnitudes[1] < 1e-9:
        return [np.nan] * n_harmonics
    norm = magnitudes[1]
    result = []
    for k in range(1, n_harmonics + 1):
        if k < len(magnitudes):
            result.append(magnitudes[k] / norm)
        else:
            result.append(0.0)
    return result

def safe_ratio(a, b):
    if b is None or np.isnan(b) or abs(b) < 1e-9:
        return np.nan
    if a is None or np.isnan(a):
        return np.nan
    return a / b

W_MAP = {"KT1_W0": 0, "KT1_W1": 1, "KT1_W2": 2, "KT1_W3": 3, "KT1_W4": 4}
D_MAP = {"KT1_D1": "PT", "KT1_D2": "DCT", "KT1_D3": "TAL", "KT1_D4": "CD", "KT1_D5": "ABN"}
L_MAP = {"KT1_L0": 0, "KT1_L1": 1, "KT1_L2": 2}
BD_MAP = {"正常": 0, "脱落-轻微": 1, "脱落-严重": 2, "远端干扰": -1}
ED_MAP = {"KT1_ED0": 0, "KT1_ED1": 1, "KT1_ED2": 2, "KT1_ED3": 3, "KT1_ED4": 4, "滴状变性": 5}

# ══════════════════════════════════════════════════════════
# 单样本提取函数
# ══════════════════════════════════════════════════════════

def extract_one_sample(sample_dir, sample_name):
    """提取一个样本的所有小管特征, 返回 list of dict"""
    alg_dir = sample_dir / "alg_results" / "hr0_alg.json"
    
    required = ["K.json", "小管扩张.json", "小管外轮廓.json", "小管内轮廓.json",
                 "TBM.json", "KT1_W.json", "K_.json"]
    for fn in required:
        if not (alg_dir / fn).exists():
            print(f"    [SKIP] {fn} missing")
            return []
    
    k_all = load_json(alg_dir / "K.json")
    dilated = load_json(alg_dir / "小管扩张.json")
    outer_all = load_json(alg_dir / "小管外轮廓.json")
    inner_all = load_json(alg_dir / "小管内轮廓.json")
    tbm_all = load_json(alg_dir / "TBM.json")
    kt1_w = load_json(alg_dir / "KT1_W.json")
    k_ = load_json(alg_dir / "K_.json")
    
    # 真正的管腔标注 (KT1_TT.json), kId 链接到 KT1.id
    kt1_tt = load_json(alg_dir / "KT1_TT.json") if (alg_dir / "KT1_TT.json").exists() else []
    
    # 可选 JSON
    kt1_d = load_json(alg_dir / "KT1_D.json") if (alg_dir / "KT1_D.json").exists() else []
    kt1_l = load_json(alg_dir / "KT1_L.json") if (alg_dir / "KT1_L.json").exists() else []
    kt1_bd = load_json(alg_dir / "KT1_BD.json") if (alg_dir / "KT1_BD.json").exists() else []
    kt1_ed = load_json(alg_dir / "KT1_ED.json") if (alg_dir / "KT1_ED.json").exists() else []
    tbm_cls = load_json(alg_dir / "TBM_CLS.json") if (alg_dir / "TBM_CLS.json").exists() else []
    
    # 皮质多边形
    cortex_polys = []
    for item in k_:
        if item.get("label") == "K_2":
            pts = pts_to_np(item)
            if pts is not None and len(pts) >= 3:
                try:
                    cortex_polys.append(Polygon(pts))
                except:
                    pass
    
    def is_cortex(cx, cy):
        p = Point(cx, cy)
        return any(poly.contains(p) for poly in cortex_polys)
    
    # 索引
    kt1_items = [d for d in k_all if d.get("label") == "KT1"]
    k_by_id = {d["id"]: d for d in kt1_items}
    
    w_by_kid = {d["kId"]: d["label"] for d in kt1_w}
    d_by_kid = {d["kId"]: d["label"] for d in kt1_d}
    l_by_kid = {d["kId"]: d["label"] for d in kt1_l}
    bd_by_kid = {d["kId"]: d["label"] for d in kt1_bd}
    ed_by_kid = {d["kId"]: d["label"] for d in kt1_ed}
    tbmcls_by_kid = {d["kId"]: d["label"] for d in tbm_cls}
    
    dil_by_id = {d["id"]: d for d in dilated}
    
    outer_by_dilid = defaultdict(list)
    for d in outer_all:
        outer_by_dilid[d["kId"]].append(d)
    inner_by_dilid = defaultdict(list)
    for d in inner_all:
        inner_by_dilid[d["kId"]].append(d)
    tbm_by_outerid = {d["kId"]: d for d in tbm_all}
    
    # KT1_TT.kId 链接 KT1.id (= kid in main loop)
    tt_by_kid = defaultdict(list)
    for d in kt1_tt:
        tt_by_kid[d["kId"]].append(d)
    
    # 桥接
    if len(kt1_items) == 0 or len(dilated) == 0:
        return []
    
    # bbx 格式: [x, y, w, h] → 中心 = (x+w/2, y+h/2)
    kt1_centers = []
    kt1_id_list = []
    for d in kt1_items:
        b = d.get("bbx", [0, 0, 0, 0])
        kt1_centers.append([b[0] + b[2]/2, b[1] + b[3]/2])
        kt1_id_list.append(d["id"])
    kt1_centers = np.array(kt1_centers)
    
    dil_centers = []
    dil_id_list = []
    for d in dilated:
        b = d.get("bbx", [0, 0, 0, 0])
        dil_centers.append([b[0] + b[2]/2, b[1] + b[3]/2])
        dil_id_list.append(d["id"])
    dil_centers = np.array(dil_centers)
    
    tree_dil = cKDTree(dil_centers)
    dists, indices = tree_dil.query(kt1_centers)
    k_to_dil = {}
    for i, kid in enumerate(kt1_id_list):
        if dists[i] < 50:
            k_to_dil[kid] = dil_id_list[indices[i]]
    
    # mpp
    info_path = sample_dir / "info.json"
    mpp = 0.25
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        mpp = float(info.get("mpp", info.get("MPP", 0.25)))
    
    px2um2 = mpp * mpp
    rows = []
    
    # 预检: 如果皮质过滤后0个小管通过, 则跳过皮质过滤
    use_cortex_filter = bool(cortex_polys)
    if cortex_polys:
        n_cortex_pass = 0
        for kid2, k2 in k_by_id.items():
            w2 = W_MAP.get(w_by_kid.get(kid2, ""), -1)
            if w2 not in [0, 1, 2]:
                continue
            b2 = k2.get("bbx", [0, 0, 0, 0])
            if is_cortex(b2[0] + b2[2]/2, b2[1] + b2[3]/2):
                n_cortex_pass += 1
                break
        if n_cortex_pass == 0:
            use_cortex_filter = False
            print(f"[WARN] cortex filter=0 pass, disabled for {sample_name}")
    
    for kid, k_item in k_by_id.items():
        w_label = w_by_kid.get(kid, "")
        w_grade = W_MAP.get(w_label, -1)
        if w_grade not in [0, 1, 2]:
            continue
        
        b = k_item.get("bbx", [0, 0, 0, 0])
        cx_px, cy_px = b[0] + b[2]/2, b[1] + b[3]/2
        if use_cortex_filter and not is_cortex(cx_px, cy_px):
            continue
        
        dil_id = k_to_dil.get(kid)
        if dil_id is None:
            continue
        
        outers = outer_by_dilid.get(dil_id, [])
        if not outers:
            continue
        outer_item = max(outers, key=lambda x: x.get("area", 0))
        outer_pts = pts_to_np(outer_item)
        
        inners = inner_by_dilid.get(dil_id, [])
        inner_item = max(inners, key=lambda x: x.get("area", 0)) if inners else None
        inner_pts = pts_to_np(inner_item) if inner_item else None
        
        tbm_item = tbm_by_outerid.get(outer_item["id"])
        tbm_pts = pts_to_np(tbm_item) if tbm_item else None  # largest contour (BM outer ring)
        # 解码 TBM ring 的 outer/inner 两条边界 (用于 BM 厚度计算)
        tbm_outer_ring, tbm_inner_ring = (None, None)
        if tbm_item is not None:
            tbm_outer_ring, tbm_inner_ring = get_tbm_inner_outer(tbm_item)
        
        # 真管腔: KT1_TT.kId == KT1.id (= kid)
        tt_list = tt_by_kid.get(kid, [])
        lumen_item = max(tt_list, key=lambda x: x.get("area", 0)) if tt_list else None
        lumen_pts = pts_to_np(lumen_item) if lumen_item else None
        
        if outer_pts is None or len(outer_pts) < 3:
            continue
        
        row = {"sample": sample_name, "tubule_id": kid}
        
        # 分类
        row["W_grade"] = w_grade
        row["tubule_type"] = D_MAP.get(d_by_kid.get(kid, ""), "UNK")
        row["lumen_level"] = L_MAP.get(l_by_kid.get(kid, ""), -1)
        row["brush_border"] = BD_MAP.get(bd_by_kid.get(kid, ""), -1)
        row["epithelial_damage"] = ED_MAP.get(ed_by_kid.get(kid, ""), -1)
        row["TBM_class"] = tbmcls_by_kid.get(kid, "")
        
        # 面积
        # 命名: TUBULE = 外缘内总区域; TBM = 基底膜环; TE_LMEN = 上皮+管腔总区域 (= inner_pts polygon);
        #       LMEN = 真管腔 (KT1_TT 多边形); TE = 真上皮 = TE_LMEN - LMEN
        outer_area_px = polygon_area(outer_pts)
        inside_bm_area_px = polygon_area(inner_pts) if inner_pts is not None and len(inner_pts) >= 3 else 0
        tbm_area_px = tbm_item.get("area", 0) if tbm_item else 0
        has_lumen = lumen_item is not None
        lumen_area_px = (lumen_item.get("area", 0) if has_lumen else np.nan)
        
        row["TBM_AREA"] = tbm_area_px * px2um2
        row["TE_LMEN_AREA"] = max(inside_bm_area_px * px2um2, 0)
        row["TUBULE_AREA"] = outer_area_px * px2um2
        # 没有 KT1_TT 标注: LMEN_AREA / TE_AREA = NaN (不可计算)
        if has_lumen:
            row["LMEN_AREA"] = lumen_area_px * px2um2
            row["TE_AREA"] = max(inside_bm_area_px - lumen_area_px, 0) * px2um2
        else:
            row["LMEN_AREA"] = np.nan
            row["TE_AREA"] = np.nan
        
        # 直径
        row["TUBULE_DAMT"] = outer_item.get("diameter_outer", 0) * mpp
        # LMEN_DAMT 来自 KT1_TT.diameter_inner (真管腔直径)
        row["LMEN_DAMT"] = (lumen_item.get("diameter_inner", 0) * mpp) if lumen_item else 0
        row["TE_LMEN_DAMT"] = outer_item.get("diameter_inner", 0) * mpp
        
        # TBM 厚度 = TBM ring 外缘点到内缘点的距离
        if (tbm_outer_ring is not None and tbm_inner_ring is not None
                and len(tbm_outer_ring) >= 3 and len(tbm_inner_ring) >= 3):
            tree_inner = cKDTree(tbm_inner_ring)
            td, _ = tree_inner.query(tbm_outer_ring)
            td_um = td * mpp
            row["TBM_THICK_MAX"] = td_um.max()
            row["TBM_THICK_MIN"] = td_um.min()
            row["TBM_THICK_AVE"] = td_um.mean()
            row["TBM_THICK_STD"] = td_um.std()
        elif tbm_item:
            half = (tbm_item.get("diameter_outer", 0) - tbm_item.get("diameter_inner", 0)) / 2 * mpp
            row["TBM_THICK_MAX"] = half
            row["TBM_THICK_MIN"] = np.nan
            row["TBM_THICK_AVE"] = half
            row["TBM_THICK_STD"] = np.nan
        else:
            row["TBM_THICK_MAX"] = row["TBM_THICK_MIN"] = row["TBM_THICK_AVE"] = row["TBM_THICK_STD"] = np.nan
        
        # 上皮厚度 = (BM 内缘 / inner_pts → 真管腔表面 / lumen_pts) 距离
        if (inner_pts is not None and len(inner_pts) >= 3
                and lumen_pts is not None and len(lumen_pts) >= 3):
            tree_lmn = cKDTree(lumen_pts)
            te_d, _ = tree_lmn.query(inner_pts)
            te_um = te_d * mpp
            row["TE_THICK_MAX"] = te_um.max()
            row["TE_THICK_MIN"] = te_um.min()
            row["TE_THICK_AVE"] = te_um.mean()
            row["TE_THICK_STD"] = te_um.std()
        else:
            row["TE_THICK_MAX"] = row["TE_THICK_MIN"] = row["TE_THICK_AVE"] = row["TE_THICK_STD"] = np.nan
        
        # 小管壁厚 = TBM外缘 → 真管腔表面 距离 (BM + 上皮 总厚度)
        od = outer_item.get("diameter_outer", 0)
        if (outer_pts is not None and len(outer_pts) >= 3
                and lumen_pts is not None and len(lumen_pts) >= 3):
            tree_lmn2 = cKDTree(lumen_pts)
            wd, _ = tree_lmn2.query(outer_pts)
            wd_um = wd * mpp
            row["TUBULE_THICK_MAX"] = wd_um.max()
            row["TUBULE_THICK_MIN"] = wd_um.min()
            row["TUBULE_THICK_AVE"] = wd_um.mean()
            row["TUBULE_THICK_STD"] = wd_um.std()
        else:
            ind = (lumen_item.get("diameter_inner", 0) if lumen_item
                   else (inner_item.get("diameter_inner", 0) if inner_item else 0))
            wt = (od - ind) / 2 * mpp
            row["TUBULE_THICK_MAX"] = wt
            row["TUBULE_THICK_MIN"] = np.nan
            row["TUBULE_THICK_AVE"] = wt
            row["TUBULE_THICK_STD"] = np.nan
        
        # 管腔厚度 = 真管腔多边形 centroid 到边界距离 (radius 统计)
        if lumen_pts is not None and len(lumen_pts) >= 3:
            ic = polygon_centroid(lumen_pts)
            idists = np.sqrt(((lumen_pts - ic)**2).sum(axis=1)) * mpp
            row["LMEN_THICK_MAX"] = idists.max()
            row["LMEN_THICK_MIN"] = idists.min()
            row["LMEN_THICK_AVE"] = idists.mean()
            row["LMEN_THICK_STD"] = idists.std()
        else:
            row["LMEN_THICK_MAX"] = row["LMEN_THICK_MIN"] = row["LMEN_THICK_AVE"] = row["LMEN_THICK_STD"] = np.nan
        
        # TE+LMEN 厚度
        tel_d = outer_item.get("diameter_inner", 0)
        if tel_d > 0:
            row["TE_LMEN_THICK_MAX"] = tel_d * mpp / 2
            row["TE_LMEN_THICK_MIN"] = np.nan
            row["TE_LMEN_THICK_AVE"] = tel_d * mpp / 2
            row["TE_LMEN_THICK_STD"] = np.nan
        else:
            row["TE_LMEN_THICK_MAX"] = row["TE_LMEN_THICK_MIN"] = row["TE_LMEN_THICK_AVE"] = row["TE_LMEN_THICK_STD"] = np.nan
        
        # 平滑度: LMEN_SMOOTH 用真管腔 (lumen_pts); TE_LMEN_SMOOTH 用 BM 内表面 (inner_pts)
        for step in [10, 15, 20, 30]:
            row[f"LMEN_SMOOTH_{step}"] = smoothness(lumen_pts, step) if lumen_pts is not None else np.nan
            row[f"TBM_SMOOTH_{step}"] = smoothness(tbm_outer_ring, step) if tbm_outer_ring is not None else np.nan
            row[f"TE_LMEN_SMOOTH_{step}"] = smoothness(inner_pts, step) if inner_pts is not None else np.nan
        
        # 面积比值
        row["TE_LMEN_TBM_AREA_R"] = safe_ratio(row["TE_LMEN_AREA"], row["TBM_AREA"])
        row["TE_TBM_AREA_R"] = safe_ratio(row["TE_AREA"], row["TBM_AREA"])
        row["LMEN_TBM_AREA_R"] = safe_ratio(row["LMEN_AREA"], row["TBM_AREA"])
        row["TUBULE_TBM_AREA_R"] = safe_ratio(row["TUBULE_AREA"], row["TBM_AREA"])
        row["TE_LMEN_TUBULE_AREA_R"] = safe_ratio(row["TE_LMEN_AREA"], row["TUBULE_AREA"])
        row["LMEN_TE_LMEN_AREA_R"] = safe_ratio(row["LMEN_AREA"], row["TE_LMEN_AREA"])
        row["TE_TE_LMEN_AREA_R"] = safe_ratio(row["TE_AREA"], row["TE_LMEN_AREA"])
        row["LMEN_TUBULE_AREA_R"] = safe_ratio(row["LMEN_AREA"], row["TUBULE_AREA"])
        row["TE_TUBULE_AREA_R"] = safe_ratio(row["TE_AREA"], row["TUBULE_AREA"])
        row["LMEN_TE_AREA_R"] = safe_ratio(row["LMEN_AREA"], row["TE_AREA"])
        
        # 直径/厚度比值
        ld = row["LMEN_DAMT"]; td2 = row["TUBULE_DAMT"]; teld = row["TE_LMEN_DAMT"]
        tbm_tm = row.get("TBM_THICK_MAX", np.nan)
        te_tm = row.get("TE_THICK_MAX", np.nan)
        tbm_dm = (tbm_item.get("diameter_outer", 0) * mpp) if tbm_item else np.nan
        te_dm = teld
        
        row["LMEN_DAMT_TBM_THICK_R"] = safe_ratio(ld, tbm_tm)
        row["TUBULE_DAMT_TBM_THICK_R"] = safe_ratio(td2, tbm_tm)
        row["TE_LMEN_DAMT_TBM_THICK_R"] = safe_ratio(teld, tbm_tm)
        row["TE_TBM_THICK_R"] = safe_ratio(te_tm, tbm_tm)
        row["LMEN_DAMT_TUBULE_DAMT_R"] = safe_ratio(ld, td2)
        row["LMEN_DAMT_TE_LMEN_DAMT"] = safe_ratio(ld, teld)
        row["LMEN_DAMT_TE_THICK_R"] = safe_ratio(ld, te_tm)
        row["TUBULE_DAMT_TE_LMEN_DAMT_R"] = safe_ratio(td2, teld)
        row["TE_THICK_TUBULE_DAMT_R"] = safe_ratio(te_tm, td2)
        row["TE_TE_LMEN_DAMT_THICK_R"] = safe_ratio(te_tm, teld)
        row["TBM_DAMT_TUBULE_DAMT_R"] = safe_ratio(tbm_dm, td2)
        row["LMEN_DAMT_TBM_DAMT_R"] = safe_ratio(ld, tbm_dm)
        row["TBM_DAMT_TE_LMEN_DAMT_R"] = safe_ratio(tbm_dm, teld)
        row["TE_THICK_TBM_DAMT_R"] = safe_ratio(te_tm, tbm_dm)
        row["TE_DAMT_TE_LMEN_DAMT_R"] = safe_ratio(te_dm, teld)
        row["TE_DAMT_TUBULE_DAMT_R"] = safe_ratio(te_dm, td2)
        row["LMEN_DAMT_TE_DAMT_R"] = safe_ratio(ld, te_dm)
        row["TE_DAMT_TBM_DAMT_R"] = safe_ratio(te_dm, tbm_dm)
        row["TE_DAMT_TBM_THICK_R"] = safe_ratio(te_dm, tbm_tm)
        
        # 形状
        op = polygon_perimeter(outer_pts)
        row["outer_circularity"] = circularity_val(outer_area_px, op)
        row["outer_eccentricity"] = eccentricity_from_pts(outer_pts)
        row["outer_solidity"] = solidity_val(outer_area_px, outer_pts)
        row["outer_convexity"] = convexity_val(op, outer_pts)
        row["outer_aspect_ratio"] = od / max(outer_item.get("diameter_inner", 1), 1)
        
        # lumen 圆形度/离心率 = 真管腔 (lumen_pts)
        if lumen_pts is not None and len(lumen_pts) >= 3:
            la = polygon_area(lumen_pts)
            lp = polygon_perimeter(lumen_pts)
            row["lumen_circularity"] = circularity_val(la, lp)
            row["lumen_eccentricity"] = eccentricity_from_pts(lumen_pts)
        else:
            row["lumen_circularity"] = row["lumen_eccentricity"] = np.nan
        
        # TBM 圆形度 = BM 外环
        if tbm_outer_ring is not None and len(tbm_outer_ring) >= 3:
            row["TBM_circularity"] = circularity_val(polygon_area(tbm_outer_ring),
                                                     polygon_perimeter(tbm_outer_ring))
        else:
            row["TBM_circularity"] = np.nan
        
        # lumen 偏心 = 外缘 vs 真管腔的 centroid 距离
        if lumen_pts is not None and len(lumen_pts) >= 3:
            oc = polygon_centroid(outer_pts)
            lc = polygon_centroid(lumen_pts)
            row["lumen_centroid_offset"] = np.sqrt(((oc - lc)**2).sum()) * mpp
        else:
            row["lumen_centroid_offset"] = np.nan
        
        # Fourier
        for i, v in enumerate(fourier_descriptors(outer_pts), 1):
            row[f"fourier_outer_{i}"] = v
        fd_in = fourier_descriptors(lumen_pts) if lumen_pts is not None else [np.nan]*10
        for i, v in enumerate(fd_in, 1):
            row[f"fourier_lumen_{i}"] = v
        fd_t = fourier_descriptors(tbm_pts) if tbm_pts is not None else [np.nan]*10
        for i, v in enumerate(fd_t, 1):
            row[f"fourier_TBM_{i}"] = v
        
        # 坐标
        row["cx_um"] = cx_px * mpp
        row["cy_um"] = cy_px * mpp
        row["is_cortex"] = 1
        
        rows.append(row)
    
    return rows

# ══════════════════════════════════════════════════════════
# 批量提取
# ══════════════════════════════════════════════════════════

def _worker(sd_str):
    """子进程: 提取一个样本"""
    sd = Path(sd_str)
    try:
        rows = extract_one_sample(sd, sd.name)
        return sd.name, rows, None
    except Exception as e:
        return sd.name, [], str(e)


if __name__ == "__main__":
    base = Path(os.environ.get("PSMP_PROJ_DIR", "./project") + "/DigitalDNUPTxPaper_MaterialToShare/基底膜")
    out_dir = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data") + "/009_csv")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 找所有样本目录
    sample_dirs = sorted([d for d in base.iterdir()
                          if d.is_dir() and d.name != "007_json" and not d.name.startswith(".")])
    print(f"共发现 {len(sample_dirs)} 个样本目录")
    
    N_WORKERS = 24
    print(f"并发 worker 数: {N_WORKERS}")
    
    all_rows = []
    done_cnt = 0
    total = len(sample_dirs)
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_name = {ex.submit(_worker, str(sd)): sd.name for sd in sample_dirs}
        for fut in as_completed(future_to_name):
            sname = future_to_name[fut]
            try:
                name, rows, err = fut.result()
                done_cnt += 1
                if err:
                    print(f"[{done_cnt}/{total}] {name}... ERROR: {err}")
                else:
                    all_rows.extend(rows)
                    print(f"[{done_cnt}/{total}] {name}... n={len(rows)}")
            except Exception as e:
                done_cnt += 1
                print(f"[{done_cnt}/{total}] {sname}... FUTURE_ERROR: {e}")
    
    # ── Step 1: 小管级 CSV ──
    df_tub = pd.DataFrame(all_rows)
    tub_file = out_dir / f"tubule_features_all_{ts}.csv"
    df_tub.to_csv(tub_file, index=False)
    print(f"\n小管级 CSV: {tub_file}")
    print(f"  shape: {df_tub.shape}")
    print(f"  样本数: {df_tub['sample'].nunique()}")
    
    # ── Step 2: 患者级聚合 ──
    cat_cols = ["sample", "tubule_id", "tubule_type", "TBM_class", "cx_um", "cy_um", "is_cortex"]
    num_cols = [c for c in df_tub.columns if c not in cat_cols]
    
    # 患者级 6 种聚合 (NEPTUNE 论文标准): mean, max, min, std, skew, kurt
    agg_rows = []
    for sname, grp in df_tub.groupby("sample"):
        row = {"sample": sname, "n_tubules": len(grp)}
        for col in num_cols:
            vals = grp[col].dropna().values.astype(float)
            if len(vals) >= 3:
                row[f"mean_{col}"] = np.mean(vals)
                row[f"max_{col}"] = np.max(vals)
                row[f"min_{col}"] = np.min(vals)
                row[f"std_{col}"] = np.std(vals)
                row[f"skew_{col}"] = skew(vals)
                row[f"kurt_{col}"] = kurtosis(vals)
            elif len(vals) >= 1:
                row[f"mean_{col}"] = np.mean(vals)
                row[f"max_{col}"] = np.max(vals)
                row[f"min_{col}"] = np.min(vals)
                row[f"std_{col}"] = np.nan
                row[f"skew_{col}"] = np.nan
                row[f"kurt_{col}"] = np.nan
            else:
                row[f"mean_{col}"] = row[f"max_{col}"] = row[f"min_{col}"] = np.nan
                row[f"std_{col}"] = row[f"skew_{col}"] = row[f"kurt_{col}"] = np.nan
        agg_rows.append(row)
    
    df_pat = pd.DataFrame(agg_rows)
    pat_file = out_dir / f"patient_tubule_features_{ts}.csv"
    df_pat.to_csv(pat_file, index=False)
    print(f"\n患者级 CSV: {pat_file}")
    print(f"  shape: {df_pat.shape}")
    print(f"  样本数: {len(df_pat)}")
    print(f"\n✅ Done")
