# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────────────
# ZIH / TU Dresden settings for the baseline analysis.  EDIT THIS FILE ONCE.
#
# Every baseline SLURM script sources it, so the sbatch scripts themselves stay
# generic and nothing about your account is hard-coded into tracked job scripts.
#
#   source baseline_analysis/zih_env.sh        # also works in a Jupyter terminal
# ─────────────────────────────────────────────────────────────────────────────

# ── your SLURM project ───────────────────────────────────────────────────────
# `sacctmgr show assoc where user=$USER format=account` lists what you may use.
export BASELINE_ACCOUNT="${BASELINE_ACCOUNT:-p_genplip}"          # <-- EDIT

# ── where the bundle lives ───────────────────────────────────────────────────
# NOT /home: home quota is small and this writes tens of thousands of pose files.
# Allocate a workspace once:
#     ws_allocate -F horse genplip 90
# then `ws_find genplip` prints the path used below.
if [[ -z "${BASELINE_DIR:-}" ]]; then
    if command -v ws_find >/dev/null 2>&1 && ws_find genplip >/dev/null 2>&1; then
        export BASELINE_DIR="$(ws_find genplip)/baseline"
    else
        export BASELINE_DIR="${HOME}/GenPLIP/baseline"
    fi
fi

# ── python ───────────────────────────────────────────────────────────────────
# Adjust to whatever `module spider Python` offers on your cluster; the release
# stage must be loaded first on ZIH's LMOD setup.
export BASELINE_MODULES="${BASELINE_MODULES:-release/24.04 GCC/12.3.0 Python/3.11.3}"
# A venv holding rdkit / torch / transformers / scipy / matplotlib.
export BASELINE_VENV="${BASELINE_VENV:-$HOME/genplip-venv}"

# ── GNINA ────────────────────────────────────────────────────────────────────
# v1.3.x binaries are 1.4-2.1 GB, so they go in the WORKSPACE, not the small
# home quota. Download once from a LOGIN node (compute nodes have no internet):
#     bash baseline_analysis/get_gnina.sh
# That fetches config.BASELINE_GNINA_URL (v1.3.3, CUDA 12.8 static -- required
# for H100/Capella, fine on A100), checks it executes, and falls back to the
# older-CUDA build automatically if it does not.
if [[ -z "${GNINA_BINARY:-}" ]]; then
    if command -v ws_find >/dev/null 2>&1 && ws_find genplip >/dev/null 2>&1; then
        export GNINA_BINARY="$(ws_find genplip)/gnina"
    else
        export GNINA_BINARY="$HOME/gnina"
    fi
fi

# ── partition ────────────────────────────────────────────────────────────────
# "alpha" = Alpha Centauri (A100), "capella" = H100.
# With the v1.3.3 CUDA-12.8 build both work, provided the node driver is new
# enough for CUDA 12.8; get_gnina.sh falls back to an older-CUDA build if not.
# (The CUDA-11-era v1.0.3 the pipeline used before 2026-09-17 would NOT run on
# Capella's sm_90 at all.)
export BASELINE_PARTITION="${BASELINE_PARTITION:-alpha}"

setup_baseline_env () {
    # shellcheck disable=SC1090
    if command -v module >/dev/null 2>&1; then
        module purge 2>/dev/null || true
        # shellcheck disable=SC2086
        module load $BASELINE_MODULES 2>/dev/null || \
            echo "WARNING: could not load: $BASELINE_MODULES (check 'module spider Python')"
    fi
    if [[ -f "$BASELINE_VENV/bin/activate" ]]; then
        source "$BASELINE_VENV/bin/activate"
    else
        echo "WARNING: no venv at $BASELINE_VENV -- using whatever python is on PATH"
    fi
    export PYTHONUNBUFFERED=1
    # HuggingFace must not write into the tiny home quota, and compute nodes
    # cannot download: generate_predictions.py needs this cache pre-warmed from a
    # login node (see ZIH_HPC.md step 4).
    export HF_HOME="${HF_HOME:-$(dirname "$BASELINE_DIR")/hf_cache}"
    mkdir -p "$HF_HOME"
}
