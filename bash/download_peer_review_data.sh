#!/usr/bin/env bash
# ============================================================
# Download peer-review reproduction data from HPC to local.
#
# Usage (run from your LOCAL machine):
#   bash download_peer_review_data.sh <hpc_user@host> <local_dest>
#
# Example:
#   bash download_peer_review_data.sh binxuwang@login.rc.fas.harvard.edu ~/data/VVS_Accentuation
# ============================================================

set -euo pipefail

HPC="${1:-binxuwang@login.rc.fas.harvard.edu}"
LOCAL_DEST="${2:-$HOME/data/VVS_Accentuation}"

ENCODING_ROOT="/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Encoding_models"

echo "========================================================"
echo "Downloading VVS Accentuation peer-review data"
echo "  From : $HPC"
echo "  To   : $LOCAL_DEST"
echo "========================================================"
mkdir -p "$LOCAL_DEST"

# ── 1. PKL files (encoding + accentuated pred_resp) ─────────────────────────
# Total: ~30 MB for all 5 subjects × 2 pkls each

echo ""
echo "── Step 1: Prediction pkl files (~30 MB total) ──"

SESSIONS=(
  "red_20250428-20250430"
  "leap_250426-250501"
  "paul_20250428-20250430"
  "three0_250426-250501"
  "venus_250426-250429"
)

for SESSION in "${SESSIONS[@]}"; do
  SRC="$ENCODING_ROOT/$SESSION/posthoc_model_predict"
  DST="$LOCAL_DEST/Encoding_models/$SESSION/posthoc_model_predict"
  mkdir -p "$DST"

  echo "  $SESSION..."
  rsync -avz --progress "$HPC:$SRC/encoding_stim_info_w_pred_resp_${SESSION}.pkl" "$DST/"
  rsync -avz --progress "$HPC:$SRC/accentuated_stim_info_w_pred_resp_${SESSION}.pkl" "$DST/"
done

# ── 2. Neural HDF5 (all 5 monkeys, accentuation sessions) ───────────────────
# File: 5.6 GB

echo ""
echo "── Step 2: Neural HDF5 (~5.6 GB) ──"

HDF5_SRC="/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data/vvs-accentuate_controlsessions_5monkeys_250504-250512.h5"
mkdir -p "$LOCAL_DEST/Ephys_Data"

rsync -avz --progress \
  "$HPC:$HDF5_SRC" \
  "$LOCAL_DEST/Ephys_Data/"

echo ""
echo "========================================================"
echo "Done! Total downloaded to: $LOCAL_DEST"
echo ""
echo "Directory structure:"
echo "  $LOCAL_DEST/Ephys_Data/"
echo "    vvs-accentuate_controlsessions_5monkeys_250504-250512.h5  (5.6 GB)"
echo "  $LOCAL_DEST/Encoding_models/{session}/posthoc_model_predict/"
echo "    encoding_stim_info_w_pred_resp_{session}.pkl              (~330 KB each)"
echo "    accentuated_stim_info_w_pred_resp_{session}.pkl           (~2.6 MB each)"
echo "========================================================"
