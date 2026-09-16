#!/bin/bash
#SBATCH --job-name=genplip_dock
#SBATCH --output=logs/dock_%A_%a.out
#SBATCH --error=logs/dock_%A_%a.err
#SBATCH --array=0-23%8
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#
# Docking, one SLURM array task per shard of complexes.
#
#   cd <repo root>
#   mkdir -p logs
#   JID=$(sbatch --parsable baseline_analysis/run_docking_array.sh)
#   sbatch --dependency=afterany:$JID baseline_analysis/run_finalize.sh
#
# WHY SHARDS WRITE SEPARATE FILES
#   Each task writes docking/_shards/docking_shard_<i>.csv; run_finalize.sh folds
#   them into docking_summary.csv afterwards. 24 tasks appending to one CSV on a
#   shared filesystem would interleave rows mid-write and corrupt it.
#
#   The array index IS the shard index, so --array=0-23 means 24 shards. If you
#   change the array range, nothing else needs changing: the task count is read
#   back from SLURM_ARRAY_TASK_COUNT.
#
# RESUMABLE
#   A ligand whose docked_poses.sdf already parses is not re-docked, so a task
#   that hits its wall clock can simply be resubmitted:
#       sbatch --array=7,13 baseline_analysis/run_docking_array.sh
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"     # repo root
source baseline_analysis/zih_env.sh
setup_baseline_env

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "host      : $(hostname)"
echo "bundle    : $BASELINE_DIR"
echo "shard     : $SHARD / $NUM_SHARDS"
echo "gnina     : $GNINA_BINARY"
echo "started   : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "gpu       : none visible"

if [[ ! -x "$GNINA_BINARY" ]]; then
    echo "ERROR: no executable GNINA at $GNINA_BINARY."
    echo "       Compute nodes have no internet and the binary is 1.4-2.1 GB."
    echo "       From a LOGIN node, run:  bash baseline_analysis/get_gnina.sh"
    exit 1
fi

python baseline_analysis/run_docking.py \
    --out-dir "$BASELINE_DIR" \
    --gnina-binary "$GNINA_BINARY" \
    --shard "$SHARD" \
    --num-shards "$NUM_SHARDS"

echo "finished  : $(date -Is)"
