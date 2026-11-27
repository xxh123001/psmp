"""列出所有提取的小管特征, 按类别分组, 生成 Markdown 报告"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

csv_dir = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data") + "/009_csv")
out_dir = Path(os.environ.get("PSMP_DATA_ROOT", "./discovery_data") + "/小管差异分析")
out_dir.mkdir(exist_ok=True)

feat_f = sorted(csv_dir.glob("tubule_features_all_*.csv"))[-1]
print(f"Reading: {feat_f.name}")
df = pd.read_csv(feat_f)
print(f"Shape: {df.shape}")

# 分类特征
categories = {
    "标识符 / Identifiers": [
        "sample", "tubule_id"
    ],
    "分类标签 / Classification": [
        ("W_grade", "小管损伤等级 (0=正常, 1=轻度, 2=中度; W3/W4 已过滤)"),
        ("tubule_type", "小管类型 (PT=近曲, DCT=远曲, TAL=髓袢升支粗段, CD=集合管, ABN=异常)"),
        ("lumen_level", "管腔程度 (0=正常, 1=扩张, 2=塌陷)"),
        ("brush_border", "刷状缘 (0=正常, 1=轻度脱落, 2=严重脱落)"),
        ("epithelial_damage", "上皮损伤 (0=正常, 1-4=渐进, 5=滴状变性)"),
        ("TBM_class", "TBM 分类标签"),
    ],
    "面积特征 / Area features (单位 μm²)": [
        ("TUBULE_AREA", "小管总面积 = 外缘内全部区域 (BM+上皮+管腔)"),
        ("TBM_AREA", "基底膜环面积 (TBM ring area)"),
        ("TE_LMEN_AREA", "BM 内全部区域 = 上皮+管腔 (= 小管内轮廓多边形面积)"),
        ("LMEN_AREA", "真管腔面积 (KT1_TT 多边形面积)"),
        ("TE_AREA", "上皮面积 = TE_LMEN_AREA - LMEN_AREA"),
    ],
    "直径特征 / Diameter features (单位 μm)": [
        ("TUBULE_DAMT", "小管外径 (TUBULE 外缘外接圆直径)"),
        ("TE_LMEN_DAMT", "BM 内表面直径 (= 上皮外表面直径)"),
        ("LMEN_DAMT", "真管腔直径 (KT1_TT.diameter_inner)"),
    ],
    "厚度特征 / Thickness features (单位 μm, 每个 5 个统计量)": [
        ("TBM_THICK_{MAX,MIN,AVE,STD}", "基底膜厚度 = TBM 外环到内环逐点距离"),
        ("TE_THICK_{MAX,MIN,AVE,STD}", "上皮厚度 = BM 内表面到真管腔逐点距离"),
        ("TUBULE_THICK_{MAX,MIN,AVE,STD}", "小管壁厚 = TBM 外缘到真管腔逐点距离 (BM+上皮 总厚度)"),
        ("LMEN_THICK_{MAX,MIN,AVE,STD}", "管腔半径统计 (真管腔 centroid 到边界距离)"),
        ("TE_LMEN_THICK_{MAX,MIN,AVE,STD}", "上皮+管腔总半径 (= outer.diameter_inner / 2)"),
    ],
    "平滑度特征 / Smoothness features (4 种步长 step=10/15/20/30)": [
        ("LMEN_SMOOTH_{10,15,20,30}", "真管腔边界平滑度 (步长 step 处局部角度标准差)"),
        ("TBM_SMOOTH_{10,15,20,30}", "TBM 外环边界平滑度"),
        ("TE_LMEN_SMOOTH_{10,15,20,30}", "BM 内表面边界平滑度"),
    ],
    "面积比值 / Area ratios (10 个)": [
        ("TE_LMEN_TBM_AREA_R", "(上皮+管腔) / 基底膜"),
        ("TE_TBM_AREA_R", "上皮 / 基底膜"),
        ("LMEN_TBM_AREA_R", "管腔 / 基底膜"),
        ("TUBULE_TBM_AREA_R", "小管 / 基底膜"),
        ("TE_LMEN_TUBULE_AREA_R", "(上皮+管腔) / 小管"),
        ("LMEN_TE_LMEN_AREA_R", "管腔 / (上皮+管腔)"),
        ("TE_TE_LMEN_AREA_R", "上皮 / (上皮+管腔)"),
        ("LMEN_TUBULE_AREA_R", "管腔 / 小管"),
        ("TE_TUBULE_AREA_R", "上皮 / 小管"),
        ("LMEN_TE_AREA_R", "管腔 / 上皮"),
    ],
    "直径/厚度比值 / Diameter & Thickness ratios (19 个)": [
        ("LMEN_DAMT_TBM_THICK_R", "管腔直径 / TBM 最大厚度"),
        ("TUBULE_DAMT_TBM_THICK_R", "小管直径 / TBM 最大厚度"),
        ("TE_LMEN_DAMT_TBM_THICK_R", "(上皮+管腔)直径 / TBM 最大厚度"),
        ("TE_TBM_THICK_R", "上皮厚度 / TBM 厚度"),
        ("LMEN_DAMT_TUBULE_DAMT_R", "管腔直径 / 小管直径"),
        ("LMEN_DAMT_TE_LMEN_DAMT", "管腔直径 / (上皮+管腔)直径"),
        ("LMEN_DAMT_TE_THICK_R", "管腔直径 / 上皮厚度"),
        ("TUBULE_DAMT_TE_LMEN_DAMT_R", "小管直径 / (上皮+管腔)直径"),
        ("TE_THICK_TUBULE_DAMT_R", "上皮厚度 / 小管直径"),
        ("TE_TE_LMEN_DAMT_THICK_R", "上皮厚度 / (上皮+管腔)直径"),
        ("TBM_DAMT_TUBULE_DAMT_R", "TBM 外径 / 小管直径"),
        ("LMEN_DAMT_TBM_DAMT_R", "管腔直径 / TBM 外径"),
        ("TBM_DAMT_TE_LMEN_DAMT_R", "TBM 外径 / (上皮+管腔)直径"),
        ("TE_THICK_TBM_DAMT_R", "上皮厚度 / TBM 外径"),
        ("TE_DAMT_TE_LMEN_DAMT_R", "上皮直径 / (上皮+管腔)直径"),
        ("TE_DAMT_TUBULE_DAMT_R", "上皮直径 / 小管直径"),
        ("LMEN_DAMT_TE_DAMT_R", "管腔直径 / 上皮直径"),
        ("TE_DAMT_TBM_DAMT_R", "上皮直径 / TBM 外径"),
        ("TE_DAMT_TBM_THICK_R", "上皮直径 / TBM 最大厚度"),
    ],
    "外缘形状描述子 / Outer shape descriptors": [
        ("outer_circularity", "外缘圆形度 = 4πA/P² (1=完美圆形)"),
        ("outer_eccentricity", "外缘离心率 (0=圆, 1=直线)"),
        ("outer_solidity", "外缘致密度 = area / convex_hull_area (1=凸形)"),
        ("outer_convexity", "外缘凸度 = hull_perimeter / perimeter"),
        ("outer_aspect_ratio", "外缘长宽比 = diameter_outer / diameter_inner"),
    ],
    "管腔/TBM 形状 / Lumen & TBM shape": [
        ("lumen_circularity", "真管腔圆形度"),
        ("lumen_eccentricity", "真管腔离心率"),
        ("lumen_centroid_offset", "管腔偏心距 = 外缘 vs 真管腔 centroid 距离 (μm)"),
        ("TBM_circularity", "TBM 外环圆形度"),
    ],
    "Fourier 描述子 / Fourier descriptors (前 10 阶谐波)": [
        ("fourier_outer_{1..10}", "外缘形状 Fourier 谱"),
        ("fourier_lumen_{1..10}", "真管腔形状 Fourier 谱"),
        ("fourier_TBM_{1..10}", "TBM 外环形状 Fourier 谱"),
    ],
    "空间位置 / Spatial": [
        ("cx_um", "中心 x 坐标 (μm)"),
        ("cy_um", "中心 y 坐标 (μm)"),
        ("is_cortex", "是否位于皮质 (1=是)"),
    ],
}

# 统计有效性
def stat_summary(col):
    if col not in df.columns:
        return "MISSING"
    s = df[col]
    if s.dtype == object:
        return f"n_unique={s.nunique()}, top={s.value_counts().head(2).to_dict()}"
    s_num = pd.to_numeric(s, errors='coerce')
    n_valid = s_num.notna().sum()
    return (f"n_valid={n_valid:>6} ({n_valid/len(df)*100:5.1f}%), "
            f"median={s_num.median():.3f}, range=[{s_num.min():.2f}, {s_num.max():.2f}]")

# 生成 Markdown
lines = []
lines.append("# 小管特征总览\n")
lines.append(f"**数据**: `{feat_f.name}`  ")
lines.append(f"**样本数**: {df['sample'].nunique()}  ")
lines.append(f"**小管总数**: {len(df):,}  ")
lines.append(f"**特征列数**: {df.shape[1]}\n")
lines.append("---\n")

total_listed = 0
all_listed_cols = set()
for cat, items in categories.items():
    lines.append(f"\n## {cat}\n")
    lines.append("| 字段 | 描述 | 统计 |")
    lines.append("|---|---|---|")
    for it in items:
        if isinstance(it, tuple):
            name, desc = it
        else:
            name = it; desc = ""
        # 展开占位符 {MAX,MIN,AVE,STD} 等
        if "{" in name:
            import re
            m = re.search(r"\{([^}]+)\}", name)
            if m:
                vars_str = m.group(1)
                if ".." in vars_str:  # 1..10
                    a, b = vars_str.split("..")
                    vars_list = [str(i) for i in range(int(a), int(b)+1)]
                else:
                    vars_list = [v.strip() for v in vars_str.split(",")]
                for v in vars_list:
                    cn = name.replace(m.group(0), v)
                    stat = stat_summary(cn)
                    lines.append(f"| `{cn}` | {desc} | {stat} |")
                    all_listed_cols.add(cn)
                    total_listed += 1
                continue
        stat = stat_summary(name) if name in df.columns else "-"
        lines.append(f"| `{name}` | {desc} | {stat} |")
        all_listed_cols.add(name)
        total_listed += 1

# 未列入的字段
unlisted = set(df.columns) - all_listed_cols
if unlisted:
    lines.append(f"\n## ⚠ 未分类字段 ({len(unlisted)})\n")
    for c in sorted(unlisted):
        lines.append(f"- `{c}`: {stat_summary(c)}")

lines.append(f"\n\n**已列出特征总数**: {total_listed}")

# 关键修复说明
lines.append("\n\n---\n## 关键修复历史\n")
lines.append("""
1. **bbx 格式修复**: JSON 中 `bbx = [x, y, w, h]` (不是 `[x1,y1,x2,y2]`)，中心计算修正为 `(x+w/2, y+h/2)`。修复后皮质过滤对所有样本生效。
2. **TBM points 多 contour 解码**: TBM.json 的 points 格式为 `[n_contours, c1_n, x,y,..., c2_n, x,y,...]`，包含外环和内环。修复后能正确得到 BM ring 边界。
3. **TBM 厚度计算**: 改为 TBM 外环 → 内环逐点距离。修复后 TBM_THICK_AVE 从 397 μm（错误）变为 5.5 μm（生理合理）。
4. **真管腔标注**: 使用 `KT1_TT.json` 作为真管腔（而非"小管内轮廓"，后者实际是 BM 内表面）。修复后 TE_AREA 从 85% 为 0 变为 6.9% 为 0（中位数 802 μm²）。
""")

# 输出
md_file = out_dir / "01_features_overview.md"
md_file.write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved: {md_file}")
print(f"Total features listed: {total_listed}")
print(f"Unlisted: {len(unlisted)}")
if unlisted:
    print(f"  {sorted(unlisted)[:10]}...")
