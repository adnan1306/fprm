"""Shared helpers for the forked length-specialization experiment scripts
(submit_fork_array.sh, verify_fork_checkpoints.py). Single source of truth
for parsing config/fork_experiment.yaml and deriving each fork's run_name /
checkpoint_path, so the SLURM launcher and the verifier can't drift apart.

Also usable from the shell: `regime`/`epochs-info` subcommands print either
JSON or, with --shell, `KEY='value'` lines safe to `eval` directly.
"""
import argparse
import glob
import json
import os
import shlex
import sys

import yaml


def own_resume_bundle(checkpoint_path: str):
    """Highest-step step_*_train_state.pt already inside this fork's own
    checkpoint_path, if any — i.e. this fork previously made progress (e.g.
    before a SLURM preemption) and a requeue should continue from there
    rather than re-resuming the shared base checkpoint from scratch."""
    cands = glob.glob(os.path.join(checkpoint_path, "step_*_train_state.pt"))
    if not cands:
        return None
    return max(cands, key=lambda p: int(os.path.basename(p).split("_")[1]))


def load_fork_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def regime_for_id(cfg: dict, fork_id: int) -> dict:
    try:
        return next(r for r in cfg["regimes"] if r["id"] == fork_id)
    except StopIteration:
        raise SystemExit(f"No regime with id={fork_id} in fork config (have: "
                          f"{[r['id'] for r in cfg['regimes']]})")


def run_name(fork_id: int, regime_name: str) -> str:
    return f"fork{fork_id}_{regime_name}"


def checkpoint_path_for(cfg: dict, fork_id: int, regime_name: str) -> str:
    return os.path.join(cfg["output_root"], run_name(fork_id, regime_name))


def all_checkpoint_paths(cfg: dict) -> list:
    return [checkpoint_path_for(cfg, r["id"], r["name"]) for r in cfg["regimes"]]


def _print(fields: dict, shell: bool):
    if shell:
        for k, v in fields.items():
            print(f"{k}={shlex.quote('' if v is None else str(v))}")
    else:
        print(json.dumps(fields))


def _cmd_regime(args):
    cfg = load_fork_config(args.fork_config)
    r = regime_for_id(cfg, args.fork_id)
    checkpoint_path = checkpoint_path_for(cfg, args.fork_id, r["name"])
    base_bundle = cfg["base_checkpoint"] + "_train_state.pt"

    own_bundle = own_resume_bundle(checkpoint_path)
    already_final = os.path.isfile(os.path.join(checkpoint_path, "final_train_state.pt"))
    # First launch: resume from the shared base checkpoint. A requeue after
    # preemption: resume from this fork's own latest bundle instead, so
    # progress since forking isn't discarded and its wandb run continues
    # rather than a duplicate being created (see pretrain.py's prev_wandb_id
    # handling, which only trusts a resumed bundle's wandb_id when it's this
    # fork's own — this field is what lets it tell the two cases apart).
    resume_from = own_bundle if own_bundle else base_bundle

    has_content = os.path.isdir(checkpoint_path) and bool(os.listdir(checkpoint_path))
    # A non-empty checkpoint_path is only safe to proceed into when it's
    # this fork's own recognizable prior progress (an own bundle, or it's
    # already finished) — anything else is unrecognized/foreign content and
    # the launcher should refuse rather than silently building on top of it.
    dir_safe = (not has_content) or bool(own_bundle) or already_final

    fields = {
        "BASE_CHECKPOINT": cfg["base_checkpoint"],
        "OUTPUT_ROOT": cfg["output_root"],
        "SEED": cfg["seed_base"] + args.fork_id,
        "ADDITIONAL_EPOCHS_BUDGET": cfg["additional_epochs_budget"],
        "WANDB_GROUP": cfg["wandb_group"],
        "EARLY_STOP_PATIENCE_EVALS": cfg.get("early_stop_patience_evals"),
        "EARLY_STOP_METRIC_KEY": cfg.get("early_stop_metric_key", "eval/lm_loss"),
        "EARLY_STOP_MIN_DELTA": cfg.get("early_stop_min_delta", 0.0),
        "REGIME_NAME": r["name"],
        "MIN_LOOPS": r["min_loops"],
        "MAX_LOOPS": r["max_loops"],
        "RUN_NAME": run_name(args.fork_id, r["name"]),
        "CHECKPOINT_PATH": checkpoint_path,
        "RESUME_FROM": resume_from,
        "IS_SELF_RESUME": "1" if own_bundle else "",
        "ALREADY_FINAL": "1" if already_final else "",
        "DIR_SAFE": "1" if dir_safe else "",
    }
    _print(fields, args.shell)


def _cmd_epochs_info(args):
    with open(args.all_config) as f:
        cfg = yaml.safe_load(f)
    _print({"BASE_EPOCHS": cfg["epochs"], "EVAL_INTERVAL": cfg["eval_interval"]}, args.shell)


def _cmd_check_paths(args):
    """Assert all 5 forks' checkpoint_path values are distinct (a config
    sanity check — catches duplicate regime ids/names). Per-fork directory
    safety (empty / own prior progress / foreign content) is resolved by
    `regime`'s DIR_SAFE field instead, since only that command knows which
    fork a given directory belongs to."""
    cfg = load_fork_config(args.fork_config)
    paths = all_checkpoint_paths(cfg)
    if len(set(paths)) != len(paths):
        raise SystemExit(f"Fork checkpoint paths are not all distinct: {paths}")
    print(f"OK: {len(paths)} fork checkpoint paths are distinct.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_regime = sub.add_parser("regime", help="Resolve one fork's settings.")
    p_regime.add_argument("fork_config")
    p_regime.add_argument("fork_id", type=int)
    p_regime.add_argument("--shell", action="store_true")
    p_regime.set_defaults(func=_cmd_regime)

    p_epochs = sub.add_parser("epochs-info", help="Read epochs/eval_interval off a resolved all_config.yaml.")
    p_epochs.add_argument("all_config")
    p_epochs.add_argument("--shell", action="store_true")
    p_epochs.set_defaults(func=_cmd_epochs_info)

    p_check = sub.add_parser("check-paths", help="Verify all 5 fork checkpoint paths are distinct and unused.")
    p_check.add_argument("fork_config")
    p_check.set_defaults(func=_cmd_check_paths)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
