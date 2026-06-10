#!/usr/bin/env bash
# =====================================================================
# Simple runner for the Student-Teacher model — locally (conda) or in Docker.
#
# Local (conda env) modes:
#   ./run.sh train                       # train with default config
#   ./run.sh test                        # evaluate the best checkpoint
#   ./run.sh train -c configs/mvtec_capsule_dino.yaml
#   ./run.sh test  -c configs/config.yaml --localization
#
# Docker modes (builds the image if needed, mounts data/ and outputs/):
#   ./run.sh docker-build                # just build the image
#   ./run.sh docker-train                # build (if needed) + train in a container
#   ./run.sh docker-test  --localization # build (if needed) + test in a container
#
# TensorBoard (compare all experiments under outputs/):
#   ./run.sh tb                          # serve outputs/ on http://localhost:6006
#   ./run.sh tb --port 6007              # extra flags pass through to tensorboard
#
# Any extra flags after the mode are forwarded straight to the python script.
# =====================================================================
set -euo pipefail

# --- Settings ---------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-myenv}"          # override with CONDA_ENV=... ./run.sh
IMAGE="${IMAGE:-stad}"                    # docker image tag
CONFIG="configs/config.yaml"             # default; override with -c / --config

# Always run from the project root (directory of this script).
cd "$(dirname "$0")"

# --- Parse args -------------------------------------------------------
MODE="${1:-}"
case "$MODE" in
    train|test|docker-build|docker-train|docker-test|tb) ;;
    *)
        echo "Usage: $0 {train|test|docker-build|docker-train|docker-test|tb} [-c CONFIG] [extra args...]" >&2
        exit 1
        ;;
esac
shift || true

# Pull out an optional -c/--config; pass everything else through.
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config) CONFIG="$2"; shift 2 ;;
        *)           EXTRA+=("$1"); shift ;;
    esac
done

# --- Docker helpers ---------------------------------------------------
docker_build() {
    echo ">> docker build -t $IMAGE ."
    docker build -t "$IMAGE" .
}

# Run a stad CLI module inside the container with the GPU and host data/outputs.
docker_run() {
    local module="$1"; shift
    # Build the image if it doesn't exist yet.
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        docker_build
    fi
    echo ">> docker run ($IMAGE)  config=$CONFIG  module=$module  extra=${EXTRA[*]:-}"
    docker run --rm --gpus all \
        -v "$PWD/data:/workspace/data" \
        -v "$PWD/outputs:/workspace/outputs" \
        "$IMAGE" \
        python -m "$module" --config "$CONFIG" "${EXTRA[@]}"
}

# --- Dispatch ---------------------------------------------------------
case "$MODE" in
    docker-build)
        docker_build
        ;;
    docker-train)
        docker_run stad.cli.train
        ;;
    docker-test)
        docker_run stad.cli.test
        ;;
    tb)
        # Serve every experiment under outputs/ as a separate run for comparison.
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        echo ">> tensorboard --logdir outputs ${EXTRA[*]:-}  (http://localhost:6006)"
        tensorboard --logdir outputs "${EXTRA[@]}"
        ;;
    train|test)
        # Local conda run. `python -m` from the project root works with or
        # without `pip install -e .`.
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        MODULE="stad.cli.${MODE}"
        echo ">> env=$CONDA_ENV  config=$CONFIG  module=$MODULE  extra=${EXTRA[*]:-}"
        python -m "$MODULE" --config "$CONFIG" "${EXTRA[@]}"
        ;;
esac
