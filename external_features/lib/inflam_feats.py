#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-tubule peri-tubular inflammation features (from discovery 235/373)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

MPP = 0.2518
INFLAM_LABELS = {
    "淋巴细胞",
    "中性粒细胞",
    "非典型中性粒细胞",
    "浆细胞",
    "嗜酸性-典型",
    "嗜酸性-困难",
    "嗜酸性-三核",
}
DBSCAN_EPS, DBSCAN_MIN = 80.0, 8
INFLAM_FEATS = [
    "dens_50",
    "inflam_frac_50",
    "nearest_um",
    "hotspot_ge5",
    "in_cluster",
    "peri_glom_dens",
    "peri_art_dens",
    "atrophy_inflam_f",
    "local_disp",
    "ring_grad",
]


def bbx_center_um(bbx):
    if not bbx or len(bbx) < 4:
        return None
    x0, y0, x1, y1 = bbx[:4]
    return ((x0 + x1) / 2.0 * MPP, (y0 + y1) / 2.0 * MPP)


def load_cells(alg_dir: Path):
    hr = alg_dir / "hr0_alg.json"
    if not hr.is_dir():
        hr = alg_dir
    c_path, cc_path = hr / "KI1_C.json", hr / "KI1_CC.json"
    if not c_path.exists() or not cc_path.exists():
        return np.zeros((0, 2)), np.zeros((0, 2))
    cells = {c["id"]: c for c in json.load(open(c_path)) if isinstance(c, dict) and "id" in c}
    inflam, non = [], []
    for cc in json.load(open(cc_path)):
        if not isinstance(cc, dict):
            continue
        lab = cc.get("label", "")
        parent = cells.get(cc.get("kId", ""))
        if parent is None:
            continue
        ctr = bbx_center_um(parent.get("bbx"))
        if ctr is None:
            continue
        if lab == "非炎症细胞":
            non.append(ctr)
        elif lab in INFLAM_LABELS:
            inflam.append(ctr)
    return (
        np.asarray(inflam, float) if inflam else np.zeros((0, 2)),
        np.asarray(non, float) if non else np.zeros((0, 2)),
    )


def features_one(sample, tub_xy, tub_ids, tub_W, glom_xy, art_xy, alg_dir: Path):
    inflam, non = load_cells(alg_dir)
    n = len(tub_xy)
    out = {k: np.zeros(n, float) for k in INFLAM_FEATS}
    out["tubule_id"] = tub_ids
    out["sample"] = np.array([sample] * n)
    if n == 0:
        return pd.DataFrame(out)

    A25, A50, A100 = np.pi * 25**2, np.pi * 50**2, np.pi * 100**2
    A_far = A100 - A50
    out["nearest_um"][:] = 250.0
    if len(inflam) == 0:
        return pd.DataFrame(out)

    tree_i = cKDTree(inflam)
    tree_n = cKDTree(non) if len(non) else None
    n25 = np.asarray(tree_i.query_ball_point(tub_xy, 25, return_length=True), float)
    n50 = np.asarray(tree_i.query_ball_point(tub_xy, 50, return_length=True), float)
    n100 = np.asarray(tree_i.query_ball_point(tub_xy, 100, return_length=True), float)
    dens25, dens50 = n25 / A25, n50 / A50
    dens_far = (n100 - n50).clip(min=0) / A_far
    non50 = (
        np.asarray(tree_n.query_ball_point(tub_xy, 50, return_length=True), float)
        if tree_n is not None
        else np.zeros(n)
    )
    nearest, nn_idx = tree_i.query(tub_xy, k=1)
    nearest = np.asarray(nearest, float)
    nn_idx = np.asarray(nn_idx, int)

    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(inflam)
    in_cl = np.zeros(n)
    for i, j in enumerate(nn_idx):
        L = int(lab[j])
        if L >= 0 and nearest[i] <= DBSCAN_EPS:
            in_cl[i] = 1.0

    peri_g = np.zeros(n)
    peri_a = np.zeros(n)
    if len(glom_xy):
        _, gi = cKDTree(glom_xy).query(tub_xy, k=1)
        gi = np.atleast_1d(np.asarray(gi, int))
        g_n50 = np.asarray(tree_i.query_ball_point(glom_xy, 50, return_length=True), float)
        peri_g = (g_n50 / A50)[gi]
    if len(art_xy):
        _, ai = cKDTree(art_xy).query(tub_xy, k=1)
        ai = np.atleast_1d(np.asarray(ai, int))
        a_n50 = np.asarray(tree_i.query_ball_point(art_xy, 50, return_length=True), float)
        peri_a = (a_n50 / A50)[ai]

    tree_tub = cKDTree(tub_xy)
    _, i_all = tree_tub.query(inflam, k=1)
    inflam_near_atrophy = (tub_W[np.asarray(i_all, int)] >= 2).astype(float)
    atrophy_f = np.zeros(n)
    balls = tree_i.query_ball_point(tub_xy, 100)
    local_disp = np.zeros(n)
    for i, idxs in enumerate(balls):
        if not idxs:
            continue
        atrophy_f[i] = float(inflam_near_atrophy[idxs].mean())
        if len(idxs) >= 3:
            d = np.linalg.norm(inflam[idxs] - tub_xy[i], axis=1)
            mu = d.mean()
            local_disp[i] = (d.std() / mu) if mu > 1e-6 else 0.0

    out["dens_50"] = dens50
    out["inflam_frac_50"] = n50 / (n50 + non50 + 1e-9)
    out["nearest_um"] = nearest
    out["hotspot_ge5"] = (n50 >= 5).astype(float)
    out["in_cluster"] = in_cl
    out["peri_glom_dens"] = peri_g
    out["peri_art_dens"] = peri_a
    out["atrophy_inflam_f"] = atrophy_f
    out["local_disp"] = local_disp
    out["ring_grad"] = dens25 - dens_far
    return pd.DataFrame(out)
