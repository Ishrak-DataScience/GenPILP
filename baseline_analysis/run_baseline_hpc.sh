#!/bin/bash
#SBATCH --job-name=genplip_baseline
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#
# Baseline analysis, steps 2-5 as one job (step 1, select_samples.py, is cheap
# and usually already done -- it runs here too if selection.csv is missing).
#
#   sbatch baseline_analysis/run_baseline_hpc.sh
#   sbatch baseline_analysis/run_baseline_hpc.sh /path/to/bundle     # custom bundle
#
# Every step is resumable: if this job hits its wall clock, resubmit it as is.
# Generation skips rows already in predictions.csv, docking skips ligands whose
# docked_poses.sdf already parses, and both rewrite their CSV as they go.
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"     # repo root
echo "repo      : $(pwd)"
echo "node      : $(hostname)"
echo "started   : $(date -Is)"

BUNDLE="${1:-$(python -c 'import config; print(config.BASELINE_DIR)')}"
echo "bundle    : $BUNDLE"
mkdir -p "$BUNDLE"

run_step () {                      # run_step <label> <command...>
    local label="$1"; shift
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  $label   ($(date +%H:%M:%S))"
    echo "──────────────────────────────────────────────────────────────"
    "$@"
}

# ── 1. selection (only if the bundle has none) ───────────────────────────────
if [[ ! -f "$BUNDLE/selection.csv" ]]; then
    run_step "1/5  select complexes" \
        python baseline_analysis/select_samples.py --out-dir "$BUNDLE"
else
    echo "1/5  select complexes     : selection.csv present, skipped"
fi

# ── 2. masks (also stages complex.pdb + plip.xml for docking) ────────────────
# --verify-plip-pos re-runs Stage 1b mode 1 and asserts PLIP++ == complement of
# PLIP--. Drop it once you trust the shortcut; it costs one extra PLIP parse per
# complex and nothing else.
run_step "2/5  build + freeze masks" \
    python baseline_analysis/build_masks.py --out-dir "$BUNDLE" --verify-plip-pos

# ── 3. vanilla ChemBERTa fills the masks ─────────────────────────────────────
run_step "3/5  generate predictions" \
    python baseline_analysis/generate_predictions.py --out-dir "$BUNDLE"

# ── 4. GNINA: parent redock + every prediction ───────────────────────────────
run_step "4/5  dock" \
    python baseline_analysis/run_docking.py --out-dir "$BUNDLE"

# ── 5. statistics + figures + RESULTS.md ─────────────────────────────────────
run_step "5/5  analyse" \
    python baseline_analysis/analyze.py --out-dir "$BUNDLE"

echo ""
echo "finished  : $(date -Is)"
echo "report    : $BUNDLE/RESULTS.md"
echo "figures   : $BUNDLE/figures/"
