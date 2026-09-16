#!/bin/bash
#SBATCH --job-name=genplip_finalize
#SBATCH --output=logs/finalize_%j.out
#SBATCH --error=logs/finalize_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Merge the docking shards, then build the tables, figures and RESULTS.md.
# No GPU needed.
#
#   sbatch --dependency=afterany:<array job id> baseline_analysis/run_finalize.sh
#
# afterany, NOT afterok, on purpose: if one shard dies (a bad structure, a wall
# clock), the analysis should still run over everything that DID dock rather
# than leaving you with nothing. analyze.py reports the n behind every number,
# and RESULTS.md names any complex whose reference could not be established.
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
source baseline_analysis/zih_env.sh
setup_baseline_env

echo "bundle    : $BASELINE_DIR"
echo "started   : $(date -Is)"

echo ""
echo "── merging docking shards ───────────────────────────────────────────"
python baseline_analysis/run_docking.py --out-dir "$BASELINE_DIR" --merge-shards

echo ""
echo "── analysis ─────────────────────────────────────────────────────────"
python baseline_analysis/analyze.py --out-dir "$BASELINE_DIR"

echo ""
echo "finished  : $(date -Is)"
echo "report    : $BASELINE_DIR/RESULTS.md"
echo "figures   : $BASELINE_DIR/figures/"
