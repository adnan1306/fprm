#!/usr/bin/env bash
# Forked length-specialization training — 5-way SLURM job array.
#
# Forks one partially-trained FPRM checkpoint into 5 independent continued
# training runs, each under a different Uniform(min_loops, max_loops)
# recurrent-computation-length regime (config/fork_experiment.yaml).
# SLURM_ARRAY_TASK_ID (0-4) selects the regime; each array task is
# otherwise an independent torchrun job, resuming full training state
# (model + optimizer + step + EMA) from the same base checkpoint.
#
# Usage:
#   sbatch scripts/submit_fork_array.sh
#
# Optional overrides at submission time:
#   sbatch --export=ALL,FORK_CONFIG=config/fork_experiment.yaml scripts/submit_fork_array.sh
#
# Requires the base checkpoint directory to contain step_<N>,
# step_<N>_train_state.pt, and all_config.yaml (the last is written
# automatically by a normal training run once wandb has initialized —
# it's what guarantees all 5 forks share the exact architecture the base
# checkpoint was trained with, which model-soup averaging later depends on).

#SBATCH --account=def-rahulgk
#SBATCH --job-name=fprm_fork
#SBATCH --array=0-4
#SBATCH --time=13:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/home/m7adnan/scratch/fprm/fork_experiment/slurm_logs/%x_%A_%a.out
#SBATCH --error=/home/m7adnan/scratch/fprm/fork_experiment/slurm_logs/%x_%A_%a.err
#SBATCH --mail-user=mohd.adnan1306@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/m7adnan/projects/def-yani/m7adnan/looped-lands/fprm}"
FORK_CONFIG="${FORK_CONFIG:-config/fork_experiment.yaml}"
FORK_ID="${SLURM_ARRAY_TASK_ID}"

cd "$REPO_ROOT"
mkdir -p /home/m7adnan/scratch/fprm/fork_experiment/slurm_logs

export DISABLE_COMPILE="${DISABLE_COMPILE:-1}"

# Preflight: all 5 forks' checkpoint dirs distinct and unused. Run by every
# array task (cheap, idempotent) so a bad config fails fast on every task
# rather than only on task 0.
uv run python scripts/fork_common.py check-paths "$FORK_CONFIG"

# Resolve this task's regime + the shared experiment settings from
# FORK_CONFIG (single source of truth — do not duplicate these values here).
# Includes RESUME_FROM: the shared base checkpoint on this fork's first
# launch, or this fork's own latest bundle if it already made progress
# (e.g. a SLURM requeue after preemption) — see fork_common.py's
# own_resume_bundle().
eval "$(uv run python scripts/fork_common.py regime "$FORK_CONFIG" "$FORK_ID" --shell)"

BASE_DIR="$(dirname "$BASE_CHECKPOINT")"
BASE_BUNDLE="${BASE_CHECKPOINT}_train_state.pt"
BASE_ALL_CONFIG="${BASE_DIR}/all_config.yaml"

if [[ ! -f "$BASE_BUNDLE" ]]; then
  echo "Missing full resume bundle: ${BASE_BUNDLE}" >&2
  exit 1
fi
if [[ ! -f "$BASE_ALL_CONFIG" ]]; then
  echo "Missing ${BASE_ALL_CONFIG} next to the base checkpoint — refusing to" >&2
  echo "guess a config/*.yaml instead, since that risks an architecture" >&2
  echo "mismatch across forks. See the script header for what's required." >&2
  exit 1
fi
if [[ -n "$ALREADY_FINAL" ]]; then
  echo "${CHECKPOINT_PATH} already has a 'final' checkpoint — this fork" >&2
  echo "already completed training. Nothing to do; exiting." >&2
  exit 0
fi
if [[ -z "$DIR_SAFE" ]]; then
  echo "Refusing to proceed: ${CHECKPOINT_PATH} is non-empty but contains no" >&2
  echo "recognizable resume bundle of this fork's own — looks like foreign or" >&2
  echo "stale content, not this fork's own in-progress checkpoints. Move it" >&2
  echo "aside or point output_root somewhere clean." >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_PATH"

# epochs override: base epochs + additional_epochs_budget, rounded up to a
# multiple of eval_interval (pretrain.py asserts epochs % eval_interval == 0).
eval "$(uv run python scripts/fork_common.py epochs-info "$BASE_ALL_CONFIG" --shell)"
EPOCHS=$(( BASE_EPOCHS + ( (ADDITIONAL_EPOCHS_BUDGET + EVAL_INTERVAL - 1) / EVAL_INTERVAL ) * EVAL_INTERVAL ))

echo "FPRM fork experiment — array task ${FORK_ID} (${REGIME_NAME})"
echo "  base checkpoint: ${BASE_CHECKPOINT}"
echo "  resuming from:   ${RESUME_FROM} $( [[ -n "$IS_SELF_RESUME" ]] && echo '(own prior progress — requeue)' || echo '(shared base — first launch)' )"
echo "  run name:        ${RUN_NAME}"
echo "  checkpoint path: ${CHECKPOINT_PATH}"
echo "  seed:            ${SEED}"
echo "  loop regime:     Uniform(${MIN_LOOPS}, ${MAX_LOOPS})"
echo "  epochs:          ${BASE_EPOCHS} -> ${EPOCHS} (+${ADDITIONAL_EPOCHS_BUDGET} budget)"
echo "  wandb group:     ${WANDB_GROUP}"

OVERRIDES=(
  "resume_from=${RESUME_FROM}"
  "seed=${SEED}"
  "epochs=${EPOCHS}"
  "run_name=${RUN_NAME}"
  "checkpoint_path=${CHECKPOINT_PATH}"
  "arch.max_iter_dist=uniform"
  "arch.max_iter=${MAX_LOOPS}"
  "+fork_id=${FORK_ID}"
  "+fork_regime=${REGIME_NAME}"
  "+wandb_group=${WANDB_GROUP}"
  "+arch.min_iter=${MIN_LOOPS}"
)
if [[ -n "${EARLY_STOP_PATIENCE_EVALS}" ]]; then
  OVERRIDES+=(
    "+early_stop_patience_evals=${EARLY_STOP_PATIENCE_EVALS}"
    "+early_stop_metric_key=${EARLY_STOP_METRIC_KEY}"
    "+early_stop_min_delta=${EARLY_STOP_MIN_DELTA}"
  )
fi

# 1 node, 4 GPUs — resumes full training state (model + optimizer + step +
# EMA) from the base checkpoint via --config-path/--config-name pointing at
# its own all_config.yaml, so the fork's architecture matches exactly.
torchrun --standalone --nproc-per-node=4 \
    pretrain.py --config-path "$BASE_DIR" --config-name all_config \
    "${OVERRIDES[@]}"
