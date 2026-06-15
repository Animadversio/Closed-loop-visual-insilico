#!/usr/bin/env bash
# ============================================================
# Download resized (425x425) accentuated images (~2.2 GB total)
#
# PREREQUISITE: Run resize_accentuated_images.sh on HPC first:
#   sbatch bash/resize_accentuated_images.sh
#
# Usage (run from your LOCAL machine):
#   bash download_accentuated_images_425px.sh <hpc_user@host> <local_dest>
#
# Example:
#   bash download_accentuated_images_425px.sh binxuwang@rcfas_login ~/Data/VVS_Accentuation
# ============================================================

set -euo pipefail

HPC="${1:-binxuwang@login.rc.fas.harvard.edu}"
LOCAL_DEST="${2:-$HOME/Data/VVS_Accentuation}"

SRC="/n/holylabs/LABS/alvarez_lab/Everyone/Accentuate_VVS/accentuation_outputs_425px"
DST="$LOCAL_DEST/accentuation_outputs_425px"
mkdir -p "$DST"

echo "========================================================"
echo "Downloading resized accentuated images (~2.2 GB)"
echo "  From : $HPC:$SRC"
echo "  To   : $DST"
echo "========================================================"
echo "Note: only PNG files are synced (.pt tensors excluded)"
echo ""

# rsync entire 425px staging dir, PNGs only, excluding .pt tensors and config files
rsync -avz --progress \
  --include="*/" \
  --include="*.png" \
  --exclude="*" \
  "$HPC:$SRC/" "$DST/"

echo ""
echo "Done! Images in: $DST"
