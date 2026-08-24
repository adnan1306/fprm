# fprm
This code accompanies the paper:\
\[[Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers](https://arxiv.org/abs/2606.18206)\]

<p align="center">
  <img src="fig1.png" width="760" alt="FPRM vs. TRM: Sudoku-Extreme accuracy vs. compute across difficulty">
  <br>
  <em>FPRM vs. TRM on Sudoku-Extreme: despite being non-hierarchical, FPRM scales better with compute and halts at the right point via fixed-points.</em>
</p>

## Installation

Requirements: Linux with an NVIDIA GPU (CUDA 12.x driver) and Python 3.11–3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nilskiKonjIzDunava/fprm.git
cd fprm
uv sync
```

`uv sync` creates a project-local `.venv/` and installs everything (PyTorch is pulled from the CUDA 12.9 wheel index). Run commands either via `uv run <cmd>` or after `source .venv/bin/activate` — always from the repository root.

Training logs to Weights & Biases: run `wandb login` first, or set `WANDB_MODE=offline` to disable.

## Usage

### 1. Build the datasets

Datasets are not shipped; build them into `data/` first. Sudoku and Maze download automatically from the Hugging Face Hub:

```bash
# Sudoku-Extreme: 1000 puzzles x 1000 augmentations
uv run python dataset/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze-Hard 30x30: non-augmented (the default; the 8x augmented build collapses to ~5%)
uv run python dataset/build_maze_dataset.py \
    --output-dir data/maze-30x30-hard-1k-noaug

# State-tracking: generated locally from the permutation group (no download).
# A5 (alternating group) and S5 (symmetric group), 2M training sequences each,
# trained at length 32 and evaluated on prefixes up to 128.
uv run python dataset/build_state_tracking_dataset.py \
    --group A5 --k-train 32 --k-eval-max 128 --train-samples 2000000 \
    --output-dir data/state_tracking-A5-train32-eval128-2M

uv run python dataset/build_state_tracking_dataset.py \
    --group S5 --k-train 32 --k-eval-max 128 --train-samples 2000000 \
    --output-dir data/state_tracking-S5-train32-eval128-2M
```

### 2. Train

`pretrain.py` is the (Hydra) training entrypoint; the reproduced experiment configs live in `config/`. `arch=fprm` is the paper's model; `trm`, `trm_singlez`, `trm_hier6`, `hrm`, and `transformers_baseline` are also available. `global_batch_size=768` is fixed regardless of GPU count, so results match across these configurations. Sudoku trains on a single GPU as shown below. Maze on multiple GPUs. For the multi-seed runs, set `seed=0`, `seed=1`, `seed=2`.

#### Single-GPU (Sudoku)

The Sudoku config uses `n_backwards_L=6` for 81 tokens per sequence, which fits in a single 40 GB GPU at `global_batch_size=768`. So you can train it without `torchrun`:

```bash
uv run python pretrain.py --config-name cfg_pretrain_sudoku
```

Any field can be overridden on the command line — e.g. the seed and W&B naming:

```bash
uv run python pretrain.py --config-name cfg_pretrain_sudoku \
    seed=1 +project_name=fprm-sudoku +run_name=my-run +checkpoint_path=checkpoints/my-run
```

#### Multi-GPU (Maze)

The Maze config uses `n_backwards_L=6` for 900 tokens per sequence, which exceeds a single 40 GB GPU at `global_batch_size=768`. Train it across multiple GPUs with `torchrun` — e.g. 4× A100-80GB on one node, or 8× A100-40GB across two:

```bash
# 1 node, 4 GPUs
torchrun --standalone --nproc-per-node=4 \
    pretrain.py --config-name cfg_pretrain_maze

# 2 nodes x 4 GPUs (one torchrun per node; HEAD_NODE = rank-0 hostname)
torchrun --nnodes=2 --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$HEAD_NODE:29850" \
    pretrain.py --config-name cfg_pretrain_maze
```

#### Single-GPU (State tracking)

State tracking has one config per group — `cfg_pretrain_state_tracking_fprm_a5`
(alternating group A5) and `cfg_pretrain_state_tracking_fprm_s5` (symmetric group
S5). Both are small (hidden 512, 2 L-layers, causal attention, no puzzle embedding,
no positional encoding) and train on a single 40 GB GPU at `global_batch_size=1024`:

```bash
# A5
uv run python pretrain.py --config-name cfg_pretrain_state_tracking_fprm_a5

# S5
uv run python pretrain.py --config-name cfg_pretrain_state_tracking_fprm_s5
```

The two configs are identical except for the group-specific settings baked in:
`data_paths`, `arch.decay_patience` (A5: 5, S5: 10), and `arch.stepsize_decay_train`
(A5: 0.90, S5: 0.95). As above, override the seed and W&B bookkeeping on the command
line for multi-seed runs:

```bash
uv run python pretrain.py --config-name cfg_pretrain_state_tracking_fprm_s5 \
    seed=1 +project_name=fprm-state-tracking +run_name=my-run +checkpoint_path=checkpoints/my-run
```

## Forked Length-Specialization Experiment

Forks one partially-trained checkpoint into 5 independent continued-training
runs, each specializing in a different recurrent-computation-length regime
(`T ~ Uniform(min_loops, max_loops)`, sampled fresh per training batch via
`set_num_iters()`), while sharing the same architecture, optimizer state,
and training step at the fork point. All other hyperparameters — data,
`halting_mechanism`, loss weights — stay identical across forks, so
computation length is isolated as the sole varying axis. Intended for
later parameter-averaging ("model soup") across the 5 resulting checkpoints.

**Prerequisites.** The base checkpoint directory must contain `step_<N>`
(weights), `step_<N>_train_state.pt` (full resume bundle: optimizer, step,
EMA, dataset shuffle counter), and `all_config.yaml` (the fully-resolved
config from that run, written automatically by any normal training run
once W&B has initialized). `all_config.yaml` is what guarantees all 5 forks
share the source run's exact architecture — the launcher refuses to start
without it rather than guessing a config from `config/`.

**Configure.** Edit `config/fork_experiment.yaml`: `base_checkpoint`,
`output_root`, per-fork `seed_base` offset, and the 5 `regimes`
(`min_loops`/`max_loops`). The shipped values are tuned against the
Maze-Hard base run's original `max_iter=24` — treat them as a starting
point, not a validated default, for other datasets.
`additional_epochs_budget` defaults to `0`, which leaves `epochs` exactly
as the base run had it — correct whenever the base checkpoint isn't near
the end of its original schedule, since there's already training budget
left. Only raise it (it's rounded up to a multiple of `eval_interval`) if
the checkpoint is close to/at its original `epochs` and the forks need
more room to train past that point.

**Launch** (SLURM job array, one task per fork):

```bash
sbatch scripts/submit_fork_array.sh
```

Each array task (`SLURM_ARRAY_TASK_ID` 0-4) resolves its regime from
`config/fork_experiment.yaml`, resumes full training state via
`--config-path <base_dir> --config-name all_config`, and trains under
`arch.max_iter_dist=uniform` with that regime's `min_iter`/`max_iter`.
`FORK_CONFIG=path sbatch --export=ALL,FORK_CONFIG=... scripts/submit_fork_array.sh`
points at a different fork config.

Requeue-safe: on a fork's first launch it resumes from the shared base
checkpoint; if that array task is preempted and requeued (`REQUEUE` in the
`#SBATCH --mail-type`), it detects its own previously-saved
`step_<N>_train_state.pt` inside its own checkpoint dir and resumes from
there instead — continuing training and the same W&B run rather than
restarting from the base checkpoint and forking a duplicate W&B run each
time. A fork that already has a `final` checkpoint exits immediately
without relaunching.

**What each fork writes**, under `<output_root>/fork<id>_<regime>/`:

| File | When |
|---|---|
| `step_<N>`, `step_<N>_train_state.pt` | periodic, per `checkpoint_every_eval` |
| `post_load`, `post_load_train_state.pt` | once, immediately after resuming — the reproducibility-check snapshot |
| `final`, `final_train_state.pt` | once, at the end of training (epoch budget reached or early-stopped) |
| `all_config.yaml` | once, at start — this fork's fully-resolved config |
| `fork_metadata.json` | once, at end — base checkpoint, seed, regime, step range, best/final eval loss, mean loop count, checksums |

**Verify.** Confirm all 5 forks started parameter-identical and later
diverged:

```bash
uv run python scripts/verify_fork_checkpoints.py --fork-config config/fork_experiment.yaml
```

**Model soup, later.** `pretrain.mix_weights_direct` already implements
linear weight averaging over a list of loaded models — load the 5
`final` state_dicts and pass them there (not implemented as part of
this experiment).

## Citation

```bibtex
@misc{movahedi2026fixedpointreasonersstableadaptive,
      title={Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers}, 
      author={Sajad Movahedi and Vera Milovanović and Shlomo Libo Feigin and Alexander Theus and Thomas Hofmann and Valentina Boeva and T. Konstantin Rusch and Antonio Orvieto},
      year={2026},
      eprint={2606.18206},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.18206}, 
}
```

## Acknowledgments

The puzzle tasks built on [TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) and state-tracking on [FP-RNN](https://github.com/dr-faustus/fp-rnn)
