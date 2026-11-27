#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segmentation evaluation for Table 1: tissue DSC, structure iDSC, F1, PPV.

Metrics follow Methods 4.3 / Table 1 footnote:
  * tissue classes       -> pixel-level Dice similarity coefficient (DSC)
  * structure instances  -> instance DSC (iDSC) over matched pairs, plus
                            F1 and PPV; a predicted and a reference
                            instance are matched when instance Dice > 0.5
  * instance-level classification labels (e.g. glomerular lesion labels)
                         -> F1 and PPV only (iDSC not applicable)

Input schema
------------
--pred_dir / --ref_dir: directories of per-slide ``<slide>.npz`` files,
each containing
    inst  : (H, W) int32  instance id per pixel (0 = background)
    cls   : (H, W) uint8  class code per pixel
An instance's class = majority class code over its pixels.
Class codes are mapped to names with --class_map, a JSON file
``{"1": "tubule", "2": "glomerulus", ...}``.

--tissue_classes:   names evaluated by pixel DSC
--instance_classes: names evaluated by iDSC/F1/PPV
--labels: optional CSV with per-instance classification labels for
          lesion-label F1/PPV. Columns:
          slide, inst, source(pred|ref), <label columns 0/1...>
          Labels are compared on matched instances of --label_class.

Output: metrics CSV (per class: mean DSC / iDSC / F1 / PPV) plus a
per-slide breakdown.

Usage:
  python eval_segmentation.py --pred_dir pred_npz/ --ref_dir ref_npz/ \
      --class_map class_map.json \
      --tissue_classes tissue,cortex,medulla,artifact \
      --instance_classes tubule,glomerulus,artery,inflam_cell \
      [--labels glom_lesion_labels.csv --label_class glomerulus] \
      --out table1_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_DICE = 0.5


def load_maps(d: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for f in sorted(Path(d).glob("*.npz")):
        z = np.load(f)
        out[f.stem] = (z["inst"], z["cls"])
    return out


def inst_class_ids(inst: np.ndarray, cls: np.ndarray) -> dict[int, int]:
    """instance id -> majority class code."""
    m = inst > 0
    pairs = {}
    ids, counts = np.unique(inst[m], return_counts=True)
    # majority vote per instance
    order = np.argsort(inst[m], kind="stable")
    lab_i, lab_c = inst[m][order], cls[m][order]
    out = {}
    starts = np.r_[0, np.nonzero(np.diff(lab_i) > 0)[0] + 1, len(lab_i)]
    for a, b in zip(starts[:-1], starts[1:]):
        iid = int(lab_i[a])
        codes, cnts = np.unique(lab_c[a:b], return_counts=True)
        out[iid] = int(codes[np.argmax(cnts)])
    return out


def pixel_dice(pred_c: np.ndarray, ref_c: np.ndarray) -> float:
    den = pred_c.sum() + ref_c.sum()
    return float(2.0 * (pred_c & ref_c).sum() / den) if den else float("nan")


def class_instances(inst: np.ndarray, inst_cls: dict[int, int], code: int):
    """pixel count per instance id restricted to class code."""
    ids = [i for i, c in inst_cls.items() if c == code]
    if not ids:
        return {}, np.zeros(inst.shape, bool)
    mask = np.isin(inst, ids)
    return ids, mask


def instance_dice_pairs(pi, pm, ri, rm):
    """dict[(pred_id, ref_id)] -> dice, plus instance sizes."""
    joint = pm & rm
    if not joint.any():
        return {}, {}, {}
    both = np.stack([pi[joint], ri[joint]], axis=1)
    keys, cnts = np.unique(both, axis=0, return_counts=True)
    p_sizes = dict(zip(*np.unique(pi[pm], return_counts=True)))
    r_sizes = dict(zip(*np.unique(ri[rm], return_counts=True)))
    dice = {}
    for (p, r), inter in zip(map(tuple, keys), cnts):
        dice[(int(p), int(r))] = 2.0 * inter / (p_sizes[int(p)] + r_sizes[int(r)])
    return dice, p_sizes, r_sizes


def greedy_match(dice: dict, thr: float):
    """best-first matching restricted to dice > thr."""
    used_p, used_r, pairs = set(), set(), []
    for (p, r), d in sorted(dice.items(), key=lambda kv: -kv[1]):
        if d <= thr:
            break
        if p in used_p or r in used_r:
            continue
        used_p.add(p)
        used_r.add(r)
        pairs.append((p, r, d))
    return pairs


def eval_slide(pred, ref, code, name, class_map_rev):
    pi, pc = pred
    ri, rc = ref
    pi_cls, ri_cls = inst_class_ids(pi, pc), inst_class_ids(ri, rc)
    if name in TISSUE:
        pm, rm = pc == code, rc == code
        return {"slide": slide_name, "class": name, "dsc": pixel_dice(pm, rm)}
    pids, pm = class_instances(pi, pi_cls, code)
    rids, rm = class_instances(ri, ri_cls, code)
    dice, _, _ = instance_dice_pairs(pi, pm, ri, rm)
    pairs = greedy_match(dice, MATCH_DICE)
    tp = len(pairs)
    fp, fn = len(pids) - tp, len(rids) - tp
    row = {
        "slide": slide_name,
        "class": name,
        "n_pred": len(pids),
        "n_ref": len(rids),
        "n_match": tp,
        "idsc": float(np.mean([d for _, _, d in pairs])) if pairs else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
        "ppv": tp / (tp + fp) if (tp + fp) else float("nan"),
        "matches": pairs,
    }
    return row


TISSUE: set = set()


def main():
    ap = argparse.ArgumentParser(description="Table 1 segmentation evaluation")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--class_map", required=True, help='JSON {"code": "name"}')
    ap.add_argument("--tissue_classes", default="tissue,cortex,medulla,artifact")
    ap.add_argument("--instance_classes", default="tubule,glomerulus,artery,inflam_cell")
    ap.add_argument("--labels", default="", help="optional lesion-label CSV")
    ap.add_argument("--label_class", default="glomerulus")
    ap.add_argument("--out", default="table1_metrics.csv")
    args = ap.parse_args()

    global TISSUE
    TISSUE = set(args.tissue_classes.split(","))
    inst_names = args.instance_classes.split(",")
    cmap = {int(k): v for k, v in json.load(open(args.class_map)).items()}

    pred = load_maps(Path(args.pred_dir))
    ref = load_maps(Path(args.ref_dir))
    slides = sorted(set(pred) & set(ref))
    if not slides:
        sys.exit("no shared slide ids between pred_dir and ref_dir")

    rows, all_pairs = [], {}
    for s in slides:
        global slide_name
        slide_name = s
        for code, name in cmap.items():
            if name in TISSUE | set(inst_names):
                row = eval_slide(pred[s], ref[s], code, name, None)
                pairs = row.pop("matches", None)
                if pairs is not None and name == args.label_class:
                    all_pairs[s] = pairs
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # cohort summary: mean per-slide DSC; pooled counts for F1/PPV
    agg = []
    for name, g in df.groupby("class"):
        if name in TISSUE:
            agg.append({"class": name, "DSC": g.dsc.mean()})
        else:
            tp, fp, fn = g.n_match.sum(), 0, 0
            fp = int(g.n_pred.sum() - tp)
            fn = int(g.n_ref.sum() - tp)
            agg.append({
                "class": name,
                "iDSC": df.loc[df["class"] == name, "idsc"].mean(),
                "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
            })
    summ = pd.DataFrame(agg)
    print(summ.to_string(index=False))
    summ.to_csv(Path(args.out).with_name(Path(args.out).stem + "_summary.csv"), index=False)

    # lesion-label F1/PPV on matched instances
    if args.labels:
        lab = pd.read_csv(args.labels)
        lab_cols = [c for c in lab.columns if c not in ("slide", "inst", "source")]
        lres = []
        for lc in lab_cols:
            tp = fp = fn = 0
            for s, pairs in all_pairs.items():
                lp = lab[(lab.slide == s) & (lab.source == "pred")].set_index("inst")[lc]
                lr = lab[(lab.slide == s) & (lab.source == "ref")].set_index("inst")[lc]
                for p, r, _ in pairs:
                    pv = int(lp.get(p, 0) > 0)
                    rv = int(lr.get(r, 0) > 0)
                    tp += pv and rv
                    fp += pv and not rv
                    fn += rv and not pv
            lres.append({
                "label": lc,
                "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
                "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
            })
        ltab = pd.DataFrame(lres)
        print(ltab.to_string(index=False))
        ltab.to_csv(Path(args.out).with_name(Path(args.out).stem + "_labels.csv"), index=False)


if __name__ == "__main__":
    main()
