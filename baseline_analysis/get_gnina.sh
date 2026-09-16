#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Fetch a working GNINA for the baseline. RUN THIS ON A LOGIN NODE -- compute
# nodes have no direct internet, and these binaries are 1.4-2.1 GB.
#
#   bash baseline_analysis/get_gnina.sh              # -> $GNINA_BINARY from zih_env.sh
#   bash baseline_analysis/get_gnina.sh /path/gnina  # -> explicit destination
#
# WHICH BUILD
#   primary   v1.3.3, CUDA 12.8 static (2.1 GB). Required for H100 / sm_90
#             (Capella); works on A100 too, PROVIDED the node's driver supports
#             CUDA 12.8.
#   fallback  v1.3.2 default build (1.4 GB), compiled against an older CUDA for
#             compatibility with older drivers. Same v1.3 scoring functions.
#
# The driver level cannot be known from a login node (it is the COMPUTE node's
# driver that matters), so this script does not try to guess: it downloads the
# primary, checks it actually executes, and falls back if it does not. If the
# primary runs here but fails inside a job with a CUDA error, rerun with
#   BASELINE_GNINA_FORCE_FALLBACK=1 bash baseline_analysis/get_gnina.sh
#
# NOTE ON VERSIONS: v1.3 moved CNN scoring to Torch and retrained the scoring
# functions, so CNNaffinity values are NOT comparable with the v1.0.3 the
# pipeline used before 2026-09-17 -- those results are superseded, not mergeable.
# Do not mix binaries within one docking_summary.csv; run_docking.py records the
# version in manifest.json so you can tell.
set -euo pipefail

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"     # repo root
if [[ -f baseline_analysis/zih_env.sh ]]; then
    # shellcheck disable=SC1091
    source baseline_analysis/zih_env.sh
fi

DEST="${1:-${GNINA_BINARY:-$HOME/gnina}}"
PRIMARY="$(python -c 'import config; print(config.BASELINE_GNINA_URL)')"
FALLBACK="$(python -c 'import config; print(config.BASELINE_GNINA_URL_FALLBACK)')"

mkdir -p "$(dirname "$DEST")"

# 2 GB does not belong in a ZIH home directory (small quota).
case "$DEST" in
    "$HOME"/*)
        echo "NOTE: $DEST is under \$HOME. These binaries are 1.4-2.1 GB and ZIH"
        echo "      home quotas are small -- consider a workspace instead:"
        echo "        ws_allocate -F horse genplip 90"
        echo "        bash baseline_analysis/get_gnina.sh \"\$(ws_find genplip)/gnina\""
        ;;
esac

try_binary () {                    # try_binary <url> <label>
    local url="$1" label="$2" tmp="${DEST}.part"
    echo ""
    echo "── downloading $label"
    echo "   $url"
    rm -f "$tmp"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --progress-bar -o "$tmp" "$url"
    else
        wget -q --show-progress -O "$tmp" "$url"
    fi
    chmod +x "$tmp"
    echo "── checking it runs"
    if "$tmp" --version >/tmp/gnina_version.$$ 2>&1; then
        mv -f "$tmp" "$DEST"
        echo "   OK: $(head -n 2 /tmp/gnina_version.$$ | tr '\n' ' ')"
        rm -f /tmp/gnina_version.$$
        return 0
    fi
    echo "   FAILED to execute:"
    sed 's/^/     /' /tmp/gnina_version.$$ | head -n 8
    rm -f "$tmp" /tmp/gnina_version.$$
    return 1
}

if [[ "${BASELINE_GNINA_FORCE_FALLBACK:-0}" == "1" ]]; then
    try_binary "$FALLBACK" "GNINA fallback build (older CUDA)"
elif ! try_binary "$PRIMARY" "GNINA primary build (CUDA 12.8 static)"; then
    echo ""
    echo "Primary build would not run here -- trying the older-CUDA build."
    try_binary "$FALLBACK" "GNINA fallback build (older CUDA)"
fi

echo ""
echo "gnina  : $DEST"
echo "size   : $(du -h "$DEST" | cut -f1)"
echo ""
echo "Point the jobs at it by setting GNINA_BINARY in baseline_analysis/zih_env.sh:"
echo "    export GNINA_BINARY=\"$DEST\""
echo ""
echo "A login node usually has no GPU, so '--version' succeeding here does not"
echo "prove CUDA works on the compute node. The first array task will say so"
echo "immediately if it does not."
