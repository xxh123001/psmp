#!/usr/bin/env bash
# 服务器上一键跑外验四类特征 + 合并
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ALG_ROOT="${ALG_ROOT:?set ALG_ROOT}"
TUB_INST="${TUB_INST:?set TUB_INST}"
GLOM_INST="${GLOM_INST:?set GLOM_INST}"
ART_INST="${ART_INST:?set ART_INST}"
TUB_FEAT="${TUB_FEAT:?set TUB_FEAT}"
OUT="${OUT:-$ROOT/example_output}"
FROZEN="${FROZEN:-$ROOT/frozen}"
mkdir -p "$OUT"

python3 "$ROOT/scripts/01_ifta_from_alg.py" \
  --alg_root "$ALG_ROOT" --out "$OUT/patient_ifta.csv"

python3 "$ROOT/scripts/02_inflam_psi.py" \
  --alg_root "$ALG_ROOT" \
  --tubule_csv "$TUB_INST" --glom_csv "$GLOM_INST" --artery_csv "$ART_INST" \
  --frozen "$FROZEN/inflam_psi_frozen.joblib" \
  --out_tubule "$OUT/tubule_inflam_psi.csv" \
  --out_patient "$OUT/patient_inflam.csv"

python3 "$ROOT/scripts/03_tubule_psi.py" \
  --features_csv "$TUB_FEAT" \
  --frozen "$FROZEN/tubule_psi_frozen.joblib" \
  --out_tubule "$OUT/tubule_morph_psi.csv" \
  --out_patient "$OUT/patient_tubule.csv"

python3 "$ROOT/scripts/04_glom_psi.py" \
  --glom_csv "$GLOM_INST" \
  --frozen "$FROZEN/glom_psi_frozen.joblib" \
  --out_glom "$OUT/glom_morph_psi.csv" \
  --out_patient "$OUT/patient_glom.csv"

python3 "$ROOT/scripts/05_merge_patient.py" \
  --ifta "$OUT/patient_ifta.csv" \
  --tubule "$OUT/patient_tubule.csv" \
  --glom "$OUT/patient_glom.csv" \
  --inflam "$OUT/patient_inflam.csv" \
  --out "$OUT/patient_external_features.csv"

echo "DONE → $OUT/patient_external_features.csv"
