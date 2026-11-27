#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""067 - 检查 ≥100μm + 无破损囊壁 过滤后剩余肾小球数"""
import pandas as pd, numpy as np
import os
from pathlib import Path

csv_dir = Path(os.environ.get("PSMP_PROJ_DIR", "./project") + "/"
               "01.20260517_060800_V5_NGM-DKD-PAS_replica/009_csv")
p2v4 = sorted(csv_dir.glob("*-P2v4_instance_glom_*.csv"))[-1]
df = pd.read_csv(p2v4)
df["dia_um"] = 2 * np.sqrt(df["glom_area_um2"].clip(lower=1) / np.pi)

n_total = len(df)
print(f"P2v4 总数: {n_total}")
print()

# 各过滤条件
mask_size = df["dia_um"] >= 100
mask_capsule = df["has_broken_capsule"] == 0

print("=== 单独过滤 ===")
print(f"  ≥100μm:         {mask_size.sum()} ({mask_size.sum()/n_total*100:.1f}%)")
print(f"  无破损囊壁:      {mask_capsule.sum()} ({mask_capsule.sum()/n_total*100:.1f}%)")
print()

# 组合
mask_both = mask_size & mask_capsule
print(f"=== 组合: ≥100μm + 无破损囊壁 ===")
print(f"  剩余: {mask_both.sum()} / {n_total} ({mask_both.sum()/n_total*100:.1f}%)")
print(f"  剔除: {n_total - mask_both.sum()} ({(n_total-mask_both.sum())/n_total*100:.1f}%)")
print()

# 保留球特征
kept = df[mask_both]
print(f"保留球的基本特征:")
print(f"  中位直径: {kept['dia_um'].median():.0f} μm")
print(f"  中位面积: {kept['glom_area_um2'].median():.0f} μm²")
print(f"  中位圆度: {kept['circularity'].median():.3f}")
print(f"  涉及样本: {kept['sample'].nunique()} 个")
print(f"  每样本中位数: {kept.groupby('sample').size().median():.0f} 个")
print()

# 被剔除的分类
r_only_size = (~mask_size & mask_capsule).sum()
r_only_caps = (mask_size & ~mask_capsule).sum()
r_both_bad = (~mask_size & ~mask_capsule).sum()
print(f"=== 被剔除原因分析 ({n_total - mask_both.sum()} 个) ===")
print(f"  仅因 <100μm:       {r_only_size}")
print(f"  仅因 破损囊壁:     {r_only_caps}")
print(f"  两者皆有:          {r_both_bad}")
