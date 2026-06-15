#!/usr/bin/env bash
# ============================================================
# Download encoding session HDF5 files (5 subjects, ~3.2 GB)
#
# Usage (run from your LOCAL machine):
#   bash download_encoding_hdf5.sh <hpc_user@host> <local_dest>
#
# Example:
#   bash download_encoding_hdf5.sh binxuwang@rcfas_login ~/Data/VVS_Accentuation
# ============================================================

set -euo pipefail

HPC="${1:-binxuwang@login.rc.fas.harvard.edu}"
LOCAL_DEST="${2:-$HOME/Data/VVS_Accentuation}"

EPHYS="/n/holylabs/LABS/alvarez_lab/Lab/VVS_Accentuation/Ephys_Data"
DST="$LOCAL_DEST/Ephys_Data"
mkdir -p "$DST"

echo "========================================================"
echo "Downloading encoding session HDF5s (~3.2 GB total)"
echo "  From : $HPC"
echo "  To   : $DST"
echo "========================================================"

# One rsync per file (compatible with macOS rsync 2.6.x)
echo "red (~153 MB)..."
rsync -avz --progress "$HPC:$EPHYS/red_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5" "$DST/"

echo "paul (~77 MB)..."
rsync -avz --progress "$HPC:$EPHYS/paul_20250428-20250430_vvs-encodingstimuli_z1_rw100-400.h5" "$DST/"

echo "leap (~596 MB)..."
rsync -avz --progress "$HPC:$EPHYS/leap_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5" "$DST/"

echo "three0 (~709 MB)..."
rsync -avz --progress "$HPC:$EPHYS/three0_250426-250501_vvs-encodingstimuli_z1_rw80-250.h5" "$DST/"

echo "venus (~1.6 GB)..."
rsync -avz --progress "$HPC:$EPHYS/venus_250426-250429_vvs-encodingstimuli_z1_rw80-250.h5" "$DST/"

echo ""
echo "Done! Files in: $DST"
