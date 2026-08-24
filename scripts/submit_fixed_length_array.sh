#!/usr/bin/env bash
# Fork the base checkpoint into five independent controls that retain the
# original FPRM training length: exactly 24 recurrent iterations per batch.
#
# Usage:
#   sbatch scripts/submit_fixed_length_array.sh
#
# Optional config override:
#   sbatch --export=ALL,CONTROL_CONFIG=config/fixed_length_experiment.yaml \
#     scripts/submit_fixed_length_array.sh

#SBATCH --account=def-rahulgk
#SBATCH --job-name=fprm_fixed24
#SBATCH --array=0-4
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/home/m7adnan/scratch/fprm/fixed_length_experiment/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/m7adnan/scratch/fprm/fixed_length_experiment/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=mohd.adnan1306@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/m7adnan/projects/def-yani/m7adnan/looped-lands/fprm}"
CONTROL_CONFIG="${CONTROL_CONFIG:-config/fixed_length_experiment.yaml}"
FORK_ID="${SLURM_ARRAY_TASK_ID}"

cd "$REPO_ROOT"
mkdir -p /home/m7adnan/scratch/fprm/fixed_length_experiment/slurm_logs

export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

python scripts/fork_common.py check-paths "$CONTROL_CONFIG"
eval "$(python scripts/fork_common.py regime "$CONTROL_CONFIG" "$FORK_ID" --shell)"

BASE_DIR="$(dirname "$BASE_CHECKPOINT")"
BASE_BUNDLE="${BASE_CHECKPOINT}_train_state.pt"
BASE_ALL_CONFIG="${BASE_DIR}/all_config.yaml"

if [[ ! -f "$BASE_BUNDLE" ]]; then
  echo "Missing full resume bundle: ${BASE_BUNDLE}" >&2
  exit 1
fi
if [[ ! -f "$BASE_ALL_CONFIG" ]]; then
  echo "Missing base configuration: ${BASE_ALL_CONFIG}" >&2
  exit 1
fi
if [[ -n "$ALREADY_FINAL" ]]; then
  echo "${CHECKPOINT_PATH} already has a final checkpoint; nothing to do."
  exit 0
fi
if [[ -z "$DIR_SAFE" ]]; then
  echo "Refusing to use non-empty unrecognized directory: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_PATH"

eval "$(python scripts/fork_common.py epochs-info "$BASE_ALL_CONFIG" --shell)"
EPOCHS=$(( BASE_EPOCHS + ( (ADDITIONAL_EPOCHS_BUDGET + EVAL_INTERVAL - 1) / EVAL_INTERVAL ) * EVAL_INTERVAL ))

echo "FPRM fixed-length control -- array task ${FORK_ID} (${REGIME_NAME})"
echo "  base checkpoint: ${BASE_CHECKPOINT}"
echo "  resuming from:   ${RESUME_FROM}"
echo "  run name:        ${RUN_NAME}"
echo "  checkpoint path: ${CHECKPOINT_PATH}"
echo "  seed:            ${SEED}"
echo "  loop policy:     deterministic ${MAX_LOOPS} iterations"
echo "  epochs:          ${BASE_EPOCHS} -> ${EPOCHS}"
echo "  wandb group:     ${WANDB_GROUP}"

OVERRIDES=(
  "resume_from=${RESUME_FROM}"
  "seed=${SEED}"
  "epochs=${EPOCHS}"
  "run_name=${RUN_NAME}"
  "checkpoint_path=${CHECKPOINT_PATH}"
  "arch.max_iter_dist=det"
  "arch.max_iter=${MAX_LOOPS}"
  "+fork_id=${FORK_ID}"
  "+fork_regime=${REGIME_NAME}"
  "+wandb_group=${WANDB_GROUP}"
)
if [[ -n "${EARLY_STOP_PATIENCE_EVALS}" ]]; then
  OVERRIDES+=(
    "+early_stop_patience_evals=${EARLY_STOP_PATIENCE_EVALS}"
    "+early_stop_metric_key=${EARLY_STOP_METRIC_KEY}"
    "+early_stop_min_delta=${EARLY_STOP_MIN_DELTA}"
  )
fi

torchrun --standalone --nproc-per-node=4 \
  pretrain.py --config-path "$BASE_DIR" --config-name all_config \
  "${OVERRIDES[@]}"
