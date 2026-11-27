#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
068 - 过滤 P2v4: 保留 ≥100μm 肾小球, 输出过滤后 CSV + 图集
"""
import pandas as pd, numpy as np, re
import os
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang HK", "Heiti TC",
                                           "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

proj = Path(os.environ.get("PSMP_PROJ_DIR", "./project") + "/"
            "01.20260517_060800_V5_NGM-DKD-PAS_replica")
csv_dir = proj / "009_csv"
img_dir = proj / "010_image"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
def nxt(d):
    n = [int(re.match(r"(\d+)_", p.name).group(1))
         for p in d.iterdir() if re.match(r"(\d+)_", p.name)]
    return f"{max(n, default=0)+1:03d}"

# ── 读数据 ──
p2v4 = sorted(csv_dir.glob("*-P2v4_instance_glom_*.csv"))[-1]
df = pd.read_csv(p2v4)
df["dia_um"] = 2 * np.sqrt(df["glom_area_um2"].clip(lower=1) / np.pi)
print(f"原始 P2v4: {len(df)} gloms")

# ── 过滤: ≥100μm ──
df_filt = df[df["dia_um"] >= 100].copy().reset_index(drop=True)
print(f"过滤后 (≥100μm): {len(df_filt)} gloms")
print(f"剔除: {len(df) - len(df_filt)} ({(len(df)-len(df_filt))/len(df)*100:.1f}%)")

# ── 保存过滤后 CSV ──
out_csv_name = f"{nxt(csv_dir)}_{ts}-P2v4f_instance_glom_filtered100um_V1.csv"
out_csv = csv_dir / out_csv_name
df_filt.to_csv(out_csv, index=False)
print(f"→ 保存: {out_csv.name}")

# ══════════════════════════════════════════════════════════════
# 图集: 过滤前后对比
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

df_rm = df[df["dia_um"] < 100]

# ── (a) 过滤前后 size 分布 ──
ax = fig.add_subplot(gs[0, 0])
bins = np.arange(0, 400, 10)
ax.hist(df["dia_um"].clip(upper=400), bins=bins, alpha=0.5,
        color="#E74C3C", label=f"过滤前 (n={len(df)})", density=True)
ax.hist(df_filt["dia_um"].clip(upper=400), bins=bins, alpha=0.6,
        color="#2ECC71", label=f"过滤后 (n={len(df_filt)})", density=True)
ax.axvline(100, color="red", ls="--", lw=2, label="100μm 阈值")
ax.set_xlabel("等效直径 (μm)")
ax.set_ylabel("密度")
ax.set_title("(a) 过滤前后大小分布", fontsize=13, weight="bold")
ax.legend(fontsize=9)

# ── (b) 每样本 glom 数 ──
ax = fig.add_subplot(gs[0, 1])
n_pre = df.groupby("sample").size().reset_index(name="n_pre")
n_post = df_filt.groupby("sample").size().reset_index(name="n_post")
n_comp = n_pre.merge(n_post, on="sample", how="left").fillna(0)
n_comp = n_comp.sort_values("n_pre", ascending=False)
x = range(len(n_comp))
ax.bar(x, n_comp["n_pre"], alpha=0.4, color="#E74C3C", label="过滤前")
ax.bar(x, n_comp["n_post"], alpha=0.7, color="#2ECC71", label="过滤后")
ax.set_xlabel("样本 (按球数排序)")
ax.set_ylabel("Glom 数")
ax.set_title(f"(b) 每样本 Glom 数 ({len(df)}→{len(df_filt)})",
             fontsize=13, weight="bold")
ax.legend(fontsize=9)
ax.set_xticks([])

# ── (c) 保留球 circularity 分布 ──
ax = fig.add_subplot(gs[0, 2])
ax.hist(df_filt["circularity"], bins=30, color="#3498DB", alpha=0.7, edgecolor="white")
ax.axvline(df_filt["circularity"].median(), color="navy", ls="--",
           label=f"中位数={df_filt['circularity'].median():.3f}")
ax.set_xlabel("Circularity")
ax.set_ylabel("频数")
ax.set_title("(c) 保留球圆度分布", fontsize=13, weight="bold")
ax.legend(fontsize=10)

# ── (d) 保留球 面积 vs 圆度 散点 ──
ax = fig.add_subplot(gs[1, 0])
sc = ax.scatter(df_filt["dia_um"], df_filt["circularity"],
                c=df_filt["dia_um"], cmap="RdYlGn", s=12, alpha=0.5,
                vmin=100, vmax=300)
plt.colorbar(sc, ax=ax, label="直径 (μm)")
ax.set_xlabel("直径 (μm)")
ax.set_ylabel("Circularity")
ax.set_title("(d) 保留球: 直径 vs 圆度", fontsize=13, weight="bold")

# ── (e) 病理分类统计 (过滤后) ──
ax = fig.add_subplot(gs[1, 1])
cats = {
    "全球硬化": df_filt["has_global_sclerosis"].sum(),
    "节段硬化": df_filt["has_segmental_sclerosis_FSGS"].sum(),
    "新月体": df_filt["has_crescent_strict"].sum(),
    "坏死": df_filt["has_necrosis"].sum(),
    "破损囊壁": df_filt["has_broken_capsule"].sum(),
    "GBM增厚": df_filt["has_GBM_thicken"].sum(),
}
colors_bar = ["#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C", "#34495E", "#3498DB"]
bars = ax.barh(list(cats.keys()), list(cats.values()), color=colors_bar, alpha=0.8)
for bar, v in zip(bars, cats.values()):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f"{v} ({v/len(df_filt)*100:.0f}%)", va="center", fontsize=10)
ax.set_xlabel("数量")
ax.set_title("(e) 保留球病理分类", fontsize=13, weight="bold")
ax.set_xlim(0, max(cats.values()) * 1.3)

# ── (f) 总结文本 ──
ax = fig.add_subplot(gs[1, 2])
ax.axis("off")
summary = (
    f"过滤标准: 等效直径 ≥ 100μm\n"
    f"{'─'*35}\n"
    f"过滤前: {len(df)} gloms\n"
    f"过滤后: {len(df_filt)} gloms\n"
    f"剔除:   {len(df)-len(df_filt)} ({(len(df)-len(df_filt))/len(df)*100:.1f}%)\n"
    f"{'─'*35}\n"
    f"保留球特征:\n"
    f"  样本数:     {df_filt['sample'].nunique()}\n"
    f"  每样本中位: {df_filt.groupby('sample').size().median():.0f} 个\n"
    f"  中位直径:   {df_filt['dia_um'].median():.0f} μm\n"
    f"  中位面积:   {df_filt['glom_area_um2'].median():.0f} μm²\n"
    f"  中位圆度:   {df_filt['circularity'].median():.3f}\n"
    f"{'─'*35}\n"
    f"输出文件:\n"
    f"  {out_csv_name}\n"
)
ax.text(0.05, 0.95, summary, fontsize=12, va="top", family="monospace",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

fig.suptitle("P2v4 过滤图集: ≥100μm 保留 1742 个高质量肾小球",
             fontsize=16, weight="bold", y=0.98)

out_png = img_dir / f"{nxt(img_dir)}_{ts}-P2v4f_filter_atlas.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
print(f"→ 图集: {out_png.name}")
print("\n完成!")
