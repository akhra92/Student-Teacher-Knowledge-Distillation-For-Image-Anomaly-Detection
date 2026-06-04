#!/usr/bin/env bash
# =====================================================================
# Simple runner for training / testing the Student-Teacher model.
#
# Usage:
#   ./run.sh train                       # train with default config
#   ./run.sh test                        # evaluate the best checkpoint
#   ./run.sh train -c configs/mvtec_capsule.yaml
#   ./run.sh test  -c configs/config.yaml --localization
#
# Any extra flags after the mode are forwarded straight to the python script.
# =====================================================================
set -euo pipefail

# --- Settings ---------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-myenv}"          # override with CONDA_ENV=... ./run.sh
CONFIG="configs/config.yaml"             # default; override with -c / --config

# Always run from the project root (directory of this script).
cd "$(dirname "$0")"

# --- Parse args -------------------------------------------------------
MODE="${1:-}"
if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "Usage: $0 {train|test} [-c CONFIG] [extra args...]" >&2
    exit 1
fi
shift

# Pull out an optional -c/--config; pass everything else through.
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config) CONFIG="$2"; shift 2 ;;
        *)           EXTRA+=("$1"); shift ;;
    esac
done

# --- Activate the conda env -------------------------------------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# --- Run --------------------------------------------------------------
SCRIPT="scripts/${MODE}.py"
echo ">> env=$CONDA_ENV  config=$CONFIG  script=$SCRIPT  extra=${EXTRA[*]:-}"
python "$SCRIPT" --config "$CONFIG" "${EXTRA[@]}"
