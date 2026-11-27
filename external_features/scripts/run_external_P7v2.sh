#!/usr/bin/env bash
# ============================================================================
# 外验冲高一键脚本（P7v2 对齐版）—— 在服务器上跑
# 关键点：小管必须"投影"到发现集 P7v2 轴，禁止在外验上重训 slingshot。
# IFTA / 小球已在 base 表里对齐好，默认沿用，只重算 小管(投影) + 炎症(去触顶)。
# ============================================================================
set -euo pipefail
PKG="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$PKG/gate_out}"
mkdir -p "$OUT"

# ---- 必填：外验小管 instance V1+V2 合并表（含 frozen_P7v2/tubule_P7v2_meta.json 的 24 列）----
TUB_INSTANCE_V1V2="${TUB_INSTANCE_V1V2:?请设置外验小管 instance V1+V2 表路径}"

# ---- 选填：炎症重算所需（不填则跳过炎症，只冲 PT+Glo+IFTA 的 INTERIM 门禁）----
ALG_ROOT="${ALG_ROOT:-}"      # 外验 alg_results 根目录（含 <sample>_alg_result/）
TUB_INST="${TUB_INST:-$TUB_INSTANCE_V1V2}"  # 含 cx_um,cy_um,tub_W_int 的小管 instance
GLOM_INST="${GLOM_INST:-}"    # 小球 instance CSV
ART_INST="${ART_INST:-}"      # 动脉 instance CSV

# ---- 参考数据（已随包携带，一般不用改）----
BASE_FEATURES="${BASE_FEATURES:-$PKG/discovery_ref/patient_external_features_base.csv}"
OUTCOME="${OUTCOME:-$PKG/discovery_ref/预后随访合并表_140.csv}"

echo "PKG=$PKG"
echo "OUT=$OUT"

# --- 生成 IFTA 患者表（给投影 QC 用）：直接从 base 抽 IFTA_model ---
python3 - "$BASE_FEATURES" "$OUT/patient_ifta.csv" <<'PY'
import sys, pandas as pd
b = pd.read_csv(sys.argv[1])
b[["sample","IFTA_model"]].dropna().to_csv(sys.argv[2], index=False)
print("IFTA rows", len(b[["sample","IFTA_model"]].dropna()))
PY

# ---- 1) 小管 → P7v2 投影（关键一步）----
echo "== [1/5] project tubules → P7v2 =="
python3 "$PKG/scripts/11_project_external_to_P7v2.py" \
  --instance_csv "$TUB_INSTANCE_V1V2" \
  --frozen "$PKG/frozen_P7v2/tubule_P7v2_knn_frozen.joblib" \
  --ifta_csv "$OUT/patient_ifta.csv" \
  --out_tubule "$OUT/tubule_P7v2_projected.csv" \
  --out_patient "$OUT/patient_tubule_P7v2.csv" \
  --out_qc "$OUT/tubule_P7v2_project_qc.json"
echo "  QC:"; cat "$OUT/tubule_P7v2_project_qc.json"; echo
echo "  >>> 验收：ifta_sign_ok 必须为 true（tub_p87 vs IFTA 为负）"

# ---- 2) 炎症重算（去触顶，带 Winsor）----
INFLAM_ARG=""
if [[ -n "$ALG_ROOT" && -n "$GLOM_INST" && -n "$ART_INST" ]]; then
  echo "== [2/5] inflammation QC + fix =="
  python3 "$PKG/scripts/12_inflam_ceiling_qc_and_fix.py" --mode qc_external \
    --alg_root "$ALG_ROOT" --tubule_csv "$TUB_INST" --glom_csv "$GLOM_INST" --artery_csv "$ART_INST" \
    --frozen "$PKG/frozen/inflam_psi_frozen.joblib" \
    --out_report "$OUT/inflam_qc_report.json" --out_patient "$OUT/patient_inflam.csv"
  echo "  >>> 验收：psi_p87 中位在 79–93，≥99 比例 <20%"
  INFLAM_ARG="--inflam $OUT/patient_inflam.csv"
else
  echo "== [2/5] 跳过炎症（未提供 ALG_ROOT/GLOM_INST/ART_INST）→ 只冲 INTERIM 门禁 =="
fi

# ---- 3) 组装外验特征：沿用 base 的 IFTA/小球，换入新炎症 ----
echo "== [3/5] assemble external features =="
python3 "$PKG/scripts/15_assemble_external.py" \
  --base "$BASE_FEATURES" $INFLAM_ARG \
  --out "$OUT/patient_external_features.csv"

# ---- 4) 合并结局 ----
echo "== [4/5] merge outcomes =="
python3 "$PKG/scripts/13_fill_missing_and_merge_outcomes.py" \
  --features "$OUT/patient_external_features.csv" \
  --outcome "$OUTCOME" --out_dir "$OUT"

# ---- 5) 门禁打分（务必传入 P7v2 小管患者表）----
echo "== [5/5] gate scoring =="
python3 "$PKG/scripts/14_gate_score_external.py" \
  --external "$OUT/patient_external_features_with_outcomes.csv" \
  --tubule_p7v2_patient "$OUT/patient_tubule_P7v2.csv" \
  --discovery_dir "$PKG/discovery_ref" \
  --out_dir "$OUT"

echo
echo "DONE → 看 $OUT/gate_checklist.csv 与 $OUT/gate_summary.json（gates_pass_ABC）"
echo "把 $OUT 整个拷回来给我，我帮你出对齐后的外验 AUC / ROC / 森林图。"
