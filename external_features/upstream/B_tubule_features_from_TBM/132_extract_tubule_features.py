#!/usr/bin/env python3
"""
小管特征提取 V1 — 从基底膜JSON提取99+扩展特征
Step 1: 单样本试运行 (10_S_3)
过滤: 皮质 + 无破损 (W0/W1/W2)
"""
import json, math
import numpy as np
import os
from pathlib import Path
from scipy.spatial import cKDTree, ConvexHull
from shapely.geometry import Polygon, Point
from collections import defaultdict

# ══════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════

def load_json(fp):
    with open(fp) as f:
        return json.load(f)

def pts_to_np(item):
    """从 JSON item 的 points 提取 Nx2 numpy array"""
    pts = item.get("points", [])
    if not pts:
        return None
    if isinstance(pts[0], list):
        return np.array(pts, dtype=float)
    else:
        arr = np.array(pts, dtype=float)
        if len(arr) % 2 != 0:
            return None
        return arr.reshape(-1, 2)

def polygon_area(pts):
    """Shoelace formula"""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygon_perimeter(pts):
    d = np.diff(pts, axis=0, append=pts[:1])
    return np.sqrt((d**2).sum(axis=1)).sum()

def polygon_centroid(pts):
    return pts.mean(axis=0)

def circularity(area, perimeter):
    if perimeter < 1e-9:
        return 0.0
    return 4 * math.pi * area / (perimeter ** 2)

def eccentricity_from_pts(pts):
    """拟合椭圆, 返回离心率"""
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

def solidity(area, pts):
    """area / convex_hull_area"""
    if len(pts) < 3:
        return np.nan
    try:
        hull = ConvexHull(pts)
        return area / hull.volume if hull.volume > 0 else np.nan
    except:
        return np.nan

def convexity(perimeter, pts):
    """convex_perimeter / perimeter"""
    if len(pts) < 3:
        return np.nan
    try:
        hull = ConvexHull(pts)
        # hull perimeter
        hull_pts = pts[hull.vertices]
        hp = polygon_perimeter(hull_pts)
        return hp / perimeter if perimeter > 0 else np.nan
    except:
        return np.nan

def smoothness(pts, step):
    """轮廓平滑度: 相邻step个点的角度变化"""
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

def contour_thickness_stats(inner_pts, outer_pts, n_sample=360):
    """从内轮廓到外轮廓的最短距离, 返回 max/min/ave/std"""
    if inner_pts is None or outer_pts is None:
        return np.nan, np.nan, np.nan, np.nan
    if len(inner_pts) < 3 or len(outer_pts) < 3:
        return np.nan, np.nan, np.nan, np.nan
    tree = cKDTree(outer_pts)
    dists, _ = tree.query(inner_pts)
    return dists.max(), dists.min(), dists.mean(), dists.std()

def tbm_thickness_stats(tbm_pts, outer_pts, n_sample=None):
    """TBM厚度: TBM轮廓到外轮廓的距离"""
    if tbm_pts is None or outer_pts is None:
        return np.nan, np.nan, np.nan, np.nan
    if len(tbm_pts) < 3 or len(outer_pts) < 3:
        return np.nan, np.nan, np.nan, np.nan
    tree = cKDTree(outer_pts)
    dists, _ = tree.query(tbm_pts)
    # TBM 厚度 = TBM区域的宽度, 用 item 的 diameter_outer - diameter_inner 更准确
    # 这里用逐点距离作为补充
    return dists.max(), dists.min(), dists.mean(), dists.std()

def fourier_descriptors(pts, n_harmonics=10):
    """椭圆Fourier描述子前N阶"""
    if len(pts) < 10:
        return [np.nan] * n_harmonics
    # 转复数
    z = pts[:, 0] + 1j * pts[:, 1]
    z = z - z.mean()
    fft_vals = np.fft.fft(z)
    # 归一化: 除以第一阶的幅度
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

# ══════════════════════════════════════════════════════════
# 主逻辑: 单样本提取
# ══════════════════════════════════════════════════════════

base = Path(os.environ.get("PSMP_PROJ_DIR", "./project") + "/DigitalDNUPTxPaper_MaterialToShare/基底膜")
sample_name = "10_S_3"
alg_dir = base / sample_name / "alg_results" / "hr0_alg.json"

print(f"Processing: {sample_name}")

# ── 加载所有 JSON ──
k_all = load_json(alg_dir / "K.json")
dilated = load_json(alg_dir / "小管扩张.json")
outer_all = load_json(alg_dir / "小管外轮廓.json")
inner_all = load_json(alg_dir / "小管内轮廓.json")
tbm_all = load_json(alg_dir / "TBM.json")
kt1_w = load_json(alg_dir / "KT1_W.json")
kt1_d = load_json(alg_dir / "KT1_D.json")
kt1_l = load_json(alg_dir / "KT1_L.json")
kt1_bd = load_json(alg_dir / "KT1_BD.json")
kt1_ed = load_json(alg_dir / "KT1_ED.json")
tbm_cls = load_json(alg_dir / "TBM_CLS.json")
k_ = load_json(alg_dir / "K_.json")

# ── 皮质多边形 (K_2) ──
cortex_polys = []
for item in k_:
    if item.get("label") == "K_2":
        pts = pts_to_np(item)
        if pts is not None and len(pts) >= 3:
            cortex_polys.append(Polygon(pts))
print(f"  皮质多边形: {len(cortex_polys)}")

def is_cortex(cx, cy):
    p = Point(cx, cy)
    return any(poly.contains(p) for poly in cortex_polys)

# ── 构建索引 ──
# K(KT1): id → item
kt1_items = [d for d in k_all if d.get("label") == "KT1"]
k_by_id = {d["id"]: d for d in kt1_items}
print(f"  K(KT1): {len(kt1_items)}")

# 分类层: kId → label (kId 指向 K.id)
w_by_kid = {d["kId"]: d["label"] for d in kt1_w}
d_by_kid = {d["kId"]: d["label"] for d in kt1_d}
l_by_kid = {d["kId"]: d["label"] for d in kt1_l}
bd_by_kid = {d["kId"]: d["label"] for d in kt1_bd}
ed_by_kid = {d["kId"]: d["label"] for d in kt1_ed}
tbmcls_by_kid = {d["kId"]: d["label"] for d in tbm_cls}

# 小管扩张: id → item
dil_by_id = {d["id"]: d for d in dilated}

# 外轮廓: kId → 小管扩张.id, 建立 kId(=dil_id) → [outer_items]
outer_by_dilid = defaultdict(list)
for d in outer_all:
    outer_by_dilid[d["kId"]].append(d)

# 内轮廓: kId → 小管扩张.id
inner_by_dilid = defaultdict(list)
for d in inner_all:
    inner_by_dilid[d["kId"]].append(d)

# TBM: kId → 小管外轮廓.id
tbm_by_outerid = {d["kId"]: d for d in tbm_all}

# ── 桥接: K(KT1) ↔ 小管扩张 (bbx 中心最近邻) ──
kt1_centers = []
kt1_id_list = []
for d in kt1_items:
    b = d.get("bbx", [0, 0, 0, 0])
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    kt1_centers.append([cx, cy])
    kt1_id_list.append(d["id"])
kt1_centers = np.array(kt1_centers)

dil_centers = []
dil_id_list = []
for d in dilated:
    b = d.get("bbx", [0, 0, 0, 0])
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    dil_centers.append([cx, cy])
    dil_id_list.append(d["id"])
dil_centers = np.array(dil_centers)

tree_dil = cKDTree(dil_centers)
dists, indices = tree_dil.query(kt1_centers)
# 桥接: K(KT1).id → 小管扩张.id
k_to_dil = {}
for i, kid in enumerate(kt1_id_list):
    if dists[i] < 50:  # 50px 阈值
        k_to_dil[kid] = dil_id_list[indices[i]]
print(f"  桥接成功: {len(k_to_dil)}/{len(kt1_items)} (dist<50px)")

# ── 获取 mpp (像素→微米转换) ──
info_path = base / sample_name / "info.json"
with open(info_path) as f:
    info = json.load(f)
mpp = info.get("mpp", info.get("MPP", 0.25))
if isinstance(mpp, str):
    mpp = float(mpp)
print(f"  mpp: {mpp}")

# ══════════════════════════════════════════════════════════
# 特征提取: 遍历每个 K(KT1) 小管
# ══════════════════════════════════════════════════════════

W_MAP = {"KT1_W0": 0, "KT1_W1": 1, "KT1_W2": 2, "KT1_W3": 3, "KT1_W4": 4}
D_MAP = {"KT1_D1": "PT", "KT1_D2": "DCT", "KT1_D3": "TAL", "KT1_D4": "CD", "KT1_D5": "ABN"}
L_MAP = {"KT1_L0": 0, "KT1_L1": 1, "KT1_L2": 2}
BD_MAP = {"正常": 0, "脱落-轻微": 1, "脱落-严重": 2, "远端干扰": -1}
ED_MAP = {"KT1_ED0": 0, "KT1_ED1": 1, "KT1_ED2": 2, "KT1_ED3": 3, "KT1_ED4": 4, "滴状变性": 5}

rows = []
n_skip_cortex = 0
n_skip_w = 0
n_ok = 0

for kid, k_item in k_by_id.items():
    # ── 分类特征 ──
    w_label = w_by_kid.get(kid, "")
    w_grade = W_MAP.get(w_label, -1)
    
    # 过滤: 只保留 W0/W1/W2
    if w_grade not in [0, 1, 2]:
        n_skip_w += 1
        continue
    
    # 皮质过滤
    b = k_item.get("bbx", [0, 0, 0, 0])
    cx_px, cy_px = (b[0]+b[2])/2, (b[1]+b[3])/2
    if not is_cortex(cx_px, cy_px):
        n_skip_cortex += 1
        continue
    
    # ── 获取几何数据 ──
    dil_id = k_to_dil.get(kid)
    if dil_id is None:
        continue
    
    # 外轮廓
    outers = outer_by_dilid.get(dil_id, [])
    if not outers:
        continue
    outer_item = max(outers, key=lambda x: x.get("area", 0))  # 取最大的
    outer_pts = pts_to_np(outer_item)
    
    # 内轮廓 (管腔)
    inners = inner_by_dilid.get(dil_id, [])
    inner_item = max(inners, key=lambda x: x.get("area", 0)) if inners else None
    inner_pts = pts_to_np(inner_item) if inner_item else None
    
    # TBM
    tbm_item = tbm_by_outerid.get(outer_item["id"])
    tbm_pts = pts_to_np(tbm_item) if tbm_item else None
    
    if outer_pts is None:
        continue
    
    # ══════════════════════════════════════════════════════
    # 计算特征 (像素单位, 最后统一转 um)
    # ══════════════════════════════════════════════════════
    row = {"sample": sample_name, "tubule_id": kid}
    
    # ── 分类特征 ──
    row["W_grade"] = w_grade
    d_label = d_by_kid.get(kid, "")
    row["tubule_type"] = D_MAP.get(d_label, "UNK")
    l_label = l_by_kid.get(kid, "")
    row["lumen_level"] = L_MAP.get(l_label, -1)
    bd_label = bd_by_kid.get(kid, "")
    row["brush_border"] = BD_MAP.get(bd_label, -1)
    ed_label = ed_by_kid.get(kid, "")
    row["epithelial_damage"] = ED_MAP.get(ed_label, -1)
    tbmcls_label = tbmcls_by_kid.get(kid, "")
    row["TBM_class"] = tbmcls_label
    
    # ── A. 面积 (PDF #1-6, 不含 NCLEI) ──
    outer_area_px = polygon_area(outer_pts) if len(outer_pts) >= 3 else outer_item.get("area", 0)
    inner_area_px = polygon_area(inner_pts) if inner_pts is not None and len(inner_pts) >= 3 else 0
    tbm_area_px = tbm_item.get("area", 0) if tbm_item else 0
    
    te_lmen_area_px = outer_area_px - tbm_area_px  # 上皮+管腔
    te_area_px = te_lmen_area_px - inner_area_px    # 上皮 = (外-TBM) - 管腔
    tubule_area_px = outer_area_px                   # 小管总面积
    lmen_area_px = inner_area_px
    
    px2um2 = mpp * mpp
    row["TBM_AREA"] = tbm_area_px * px2um2
    row["TE_LMEN_AREA"] = max(te_lmen_area_px * px2um2, 0)
    row["TUBULE_AREA"] = tubule_area_px * px2um2
    row["LMEN_AREA"] = lmen_area_px * px2um2
    row["TE_AREA"] = max(te_area_px * px2um2, 0)
    
    # ── B. 直径 (PDF #7-9) ──
    row["TUBULE_DAMT"] = outer_item.get("diameter_outer", 0) * mpp
    row["LMEN_DAMT"] = (inner_item.get("diameter_inner", 0) * mpp) if inner_item else 0
    row["TE_LMEN_DAMT"] = (outer_item.get("diameter_inner", 0) * mpp)
    
    # ── C. 厚度 (PDF #10-29) ──
    # TBM 厚度: TBM外边界到TBM内边界
    if tbm_item:
        tbm_di = tbm_item.get("diameter_inner", 0) * mpp
        tbm_do = tbm_item.get("diameter_outer", 0) * mpp
        # 逐点厚度
        if tbm_pts is not None and outer_pts is not None and len(tbm_pts) >= 3:
            tree_outer = cKDTree(outer_pts)
            tbm_dists_to_outer, _ = tree_outer.query(tbm_pts)
            tbm_dists_um = tbm_dists_to_outer * mpp
            row["TBM_THICK_MAX"] = tbm_dists_um.max()
            row["TBM_THICK_MIN"] = tbm_dists_um.min()
            row["TBM_THICK_AVE"] = tbm_dists_um.mean()
            row["TBM_THICK_STD"] = tbm_dists_um.std()
        else:
            row["TBM_THICK_MAX"] = (tbm_do - tbm_di) / 2
            row["TBM_THICK_MIN"] = np.nan
            row["TBM_THICK_AVE"] = (tbm_do - tbm_di) / 2
            row["TBM_THICK_STD"] = np.nan
    else:
        for k in ["TBM_THICK_MAX","TBM_THICK_MIN","TBM_THICK_AVE","TBM_THICK_STD"]:
            row[k] = np.nan
    
    # 上皮厚度: 内轮廓到外轮廓 (减去TBM)
    if inner_pts is not None and len(inner_pts) >= 3:
        tree_o = cKDTree(outer_pts)
        te_dists, _ = tree_o.query(inner_pts)
        te_dists_um = te_dists * mpp
        # 扣除 TBM 厚度
        tbm_avg = row.get("TBM_THICK_AVE", 0) or 0
        te_dists_um_adj = np.maximum(te_dists_um - tbm_avg / mpp * mpp, 0)
        row["TE_THICK_MAX"] = te_dists_um_adj.max()
        row["TE_THICK_MIN"] = te_dists_um_adj.min()
        row["TE_THICK_AVE"] = te_dists_um_adj.mean()
        row["TE_THICK_STD"] = te_dists_um_adj.std()
    else:
        for k in ["TE_THICK_MAX","TE_THICK_MIN","TE_THICK_AVE","TE_THICK_STD"]:
            row[k] = np.nan
    
    # 小管壁厚 (外轮廓直径方向)
    outer_perim = polygon_perimeter(outer_pts)
    outer_diam_px = outer_item.get("diameter_outer", 0)
    inner_diam_px = inner_item.get("diameter_inner", 0) if inner_item else 0
    wall_thick_px = (outer_diam_px - inner_diam_px) / 2
    row["TUBULE_THICK_MAX"] = wall_thick_px * mpp  # 近似
    row["TUBULE_THICK_MIN"] = np.nan  # 需要逐点
    row["TUBULE_THICK_AVE"] = wall_thick_px * mpp
    row["TUBULE_THICK_STD"] = np.nan
    
    # 管腔厚度 (内轮廓直径方向)
    if inner_pts is not None and len(inner_pts) >= 3:
        inner_cent = polygon_centroid(inner_pts)
        inner_dists = np.sqrt(((inner_pts - inner_cent)**2).sum(axis=1))
        lmen_dists_um = inner_dists * mpp
        row["LMEN_THICK_MAX"] = lmen_dists_um.max()
        row["LMEN_THICK_MIN"] = lmen_dists_um.min()
        row["LMEN_THICK_AVE"] = lmen_dists_um.mean()
        row["LMEN_THICK_STD"] = lmen_dists_um.std()
    else:
        for k in ["LMEN_THICK_MAX","LMEN_THICK_MIN","LMEN_THICK_AVE","LMEN_THICK_STD"]:
            row[k] = np.nan
    
    # TE+LMEN 厚度
    if inner_pts is not None and len(inner_pts) >= 3:
        te_lmen_d = outer_item.get("diameter_inner", 0)
        row["TE_LMEN_THICK_MAX"] = te_lmen_d * mpp / 2  # 近似半径
        row["TE_LMEN_THICK_MIN"] = np.nan
        row["TE_LMEN_THICK_AVE"] = te_lmen_d * mpp / 2
        row["TE_LMEN_THICK_STD"] = np.nan
    else:
        for k in ["TE_LMEN_THICK_MAX","TE_LMEN_THICK_MIN","TE_LMEN_THICK_AVE","TE_LMEN_THICK_STD"]:
            row[k] = np.nan
    
    # ── D. 平滑度 (PDF #30-41) ──
    for step in [10, 15, 20, 30]:
        row[f"LMEN_SMOOTH_{step}"] = smoothness(inner_pts, step) if inner_pts is not None else np.nan
        row[f"TBM_SMOOTH_{step}"] = smoothness(tbm_pts, step) if tbm_pts is not None else np.nan
        row[f"TE_LMEN_SMOOTH_{step}"] = smoothness(outer_pts, step)
    
    # ── E. 面积比值 (PDF #42-51) ──
    def safe_ratio(a, b):
        return a / b if b > 1e-9 else np.nan
    
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
    
    # ── F. 直径/厚度比值 (PDF #52-70) ──
    lmen_damt = row["LMEN_DAMT"]
    tubule_damt = row["TUBULE_DAMT"]
    te_lmen_damt = row["TE_LMEN_DAMT"]
    tbm_thick_max = row.get("TBM_THICK_MAX", np.nan)
    te_thick_max = row.get("TE_THICK_MAX", np.nan)
    tbm_damt = (tbm_item.get("diameter_outer", 0) * mpp) if tbm_item else np.nan
    te_damt = te_lmen_damt  # 近似
    
    row["LMEN_DAMT_TBM_THICK_R"] = safe_ratio(lmen_damt, tbm_thick_max)
    row["TUBULE_DAMT_TBM_THICK_R"] = safe_ratio(tubule_damt, tbm_thick_max)
    row["TE_LMEN_DAMT_TBM_THICK_R"] = safe_ratio(te_lmen_damt, tbm_thick_max)
    row["TE_TBM_THICK_R"] = safe_ratio(te_thick_max, tbm_thick_max)
    row["LMEN_DAMT_TUBULE_DAMT_R"] = safe_ratio(lmen_damt, tubule_damt)
    row["LMEN_DAMT_TE_LMEN_DAMT"] = safe_ratio(lmen_damt, te_lmen_damt)
    row["LMEN_DAMT_TE_THICK_R"] = safe_ratio(lmen_damt, te_thick_max)
    row["TUBULE_DAMT_TE_LMEN_DAMT_R"] = safe_ratio(tubule_damt, te_lmen_damt)
    row["TE_THICK_TUBULE_DAMT_R"] = safe_ratio(te_thick_max, tubule_damt)
    row["TE_TE_LMEN_DAMT_THICK_R"] = safe_ratio(te_thick_max, te_lmen_damt)
    row["TBM_DAMT_TUBULE_DAMT_R"] = safe_ratio(tbm_damt, tubule_damt)
    row["LMEN_DAMT_TBM_DAMT_R"] = safe_ratio(lmen_damt, tbm_damt)
    row["TBM_DAMT_TE_LMEN_DAMT_R"] = safe_ratio(tbm_damt, te_lmen_damt)
    row["TE_THICK_TBM_DAMT_R"] = safe_ratio(te_thick_max, tbm_damt)
    row["TE_DAMT_TE_LMEN_DAMT_R"] = safe_ratio(te_damt, te_lmen_damt)
    row["TE_DAMT_TUBULE_DAMT_R"] = safe_ratio(te_damt, tubule_damt)
    row["LMEN_DAMT_TE_DAMT_R"] = safe_ratio(lmen_damt, te_damt)
    row["TE_DAMT_TBM_DAMT_R"] = safe_ratio(te_damt, tbm_damt)
    row["TE_DAMT_TBM_THICK_R"] = safe_ratio(te_damt, tbm_thick_max)
    
    # ── G. 形状描述子 (扩展) ──
    outer_area_um2 = outer_area_px * px2um2
    outer_perim_um = outer_perim * mpp
    row["outer_circularity"] = circularity(outer_area_px, outer_perim)
    row["outer_eccentricity"] = eccentricity_from_pts(outer_pts)
    row["outer_solidity"] = solidity(outer_area_px, outer_pts)
    row["outer_convexity"] = convexity(outer_perim, outer_pts)
    row["outer_aspect_ratio"] = outer_item.get("diameter_outer", 1) / max(outer_item.get("diameter_inner", 1), 1)
    
    if inner_pts is not None and len(inner_pts) >= 3:
        inner_area_calc = polygon_area(inner_pts)
        inner_perim_calc = polygon_perimeter(inner_pts)
        row["lumen_circularity"] = circularity(inner_area_calc, inner_perim_calc)
        row["lumen_eccentricity"] = eccentricity_from_pts(inner_pts)
    else:
        row["lumen_circularity"] = np.nan
        row["lumen_eccentricity"] = np.nan
    
    if tbm_pts is not None and len(tbm_pts) >= 3:
        tbm_area_calc = polygon_area(tbm_pts)
        tbm_perim_calc = polygon_perimeter(tbm_pts)
        row["TBM_circularity"] = circularity(tbm_area_calc, tbm_perim_calc)
    else:
        row["TBM_circularity"] = np.nan
    
    # 管腔中心偏移
    if inner_pts is not None and len(inner_pts) >= 3:
        outer_cent = polygon_centroid(outer_pts)
        inner_cent = polygon_centroid(inner_pts)
        row["lumen_centroid_offset"] = np.sqrt(((outer_cent - inner_cent)**2).sum()) * mpp
    else:
        row["lumen_centroid_offset"] = np.nan
    
    # ── H. Fourier 描述子 (扩展) ──
    fd_outer = fourier_descriptors(outer_pts)
    for i, v in enumerate(fd_outer, 1):
        row[f"fourier_outer_{i}"] = v
    
    if inner_pts is not None:
        fd_inner = fourier_descriptors(inner_pts)
    else:
        fd_inner = [np.nan] * 10
    for i, v in enumerate(fd_inner, 1):
        row[f"fourier_lumen_{i}"] = v
    
    if tbm_pts is not None:
        fd_tbm = fourier_descriptors(tbm_pts)
    else:
        fd_tbm = [np.nan] * 10
    for i, v in enumerate(fd_tbm, 1):
        row[f"fourier_TBM_{i}"] = v
    
    # ── I. 空间特征 ──
    row["cx_um"] = cx_px * mpp
    row["cy_um"] = cy_px * mpp
    row["is_cortex"] = 1
    
    rows.append(row)
    n_ok += 1

print(f"\n  皮质外排除: {n_skip_cortex}")
print(f"  W3/W4排除: {n_skip_w}")
print(f"  最终保留: {n_ok}")
print(f"  特征列数: {len(rows[0]) if rows else 0}")

# ── 保存 ──
import pandas as pd
df = pd.DataFrame(rows)
out_path = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data") + "/009_csv")
out_file = out_path / "tubule_features_test_10_S_3.csv"
df.to_csv(out_file, index=False)
print(f"\n已保存: {out_file}")
print(f"  shape: {df.shape}")
print(f"\n特征列:")
for c in df.columns:
    print(f"  {c}")
