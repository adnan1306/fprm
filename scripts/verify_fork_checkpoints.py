"""Verify the forked length-specialization experiment's reproducibility
contract: right after loading the shared base checkpoint, all 5 forks must
be parameter-identical (max diff == 0); by the time training finishes, the
5 final checkpoints must have diverged from each other and from that
post_load snapshot.

Usage:
    uv run python scripts/verify_fork_checkpoints.py [--fork-config config/fork_experiment.yaml] [--experiment-root <output_root>] [--stage post_load|final|both]
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from fork_common import load_fork_config, all_checkpoint_paths  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.checkpoint_diag import state_dict_checksum, max_abs_param_diff  # noqa: E402


def discover_fork_dirs(fork_config_path: str, experiment_root: str = None) -> list:
    if fork_config_path and os.path.isfile(fork_config_path):
        cfg = load_fork_config(fork_config_path)
        if experiment_root:
            cfg["output_root"] = experiment_root
        return sorted(all_checkpoint_paths(cfg))

    if experiment_root:
        dirs = sorted(glob.glob(os.path.join(experiment_root, "fork*")))
        if dirs:
            return dirs

    raise SystemExit(f"Could not resolve fork checkpoint dirs: fork-config "
                      f"'{fork_config_path}' not found, and no fork* dirs under "
                      f"experiment-root '{experiment_root}'.")


def load_state_dict(path: str):
    return torch.load(path, map_location="cpu")


def verify_post_load(fork_dirs: list) -> bool:
    print(f"\n=== post_load reproducibility check ({len(fork_dirs)} forks) ===")
    ok = True
    ref_path = os.path.join(fork_dirs[0], "post_load")
    if not os.path.isfile(ref_path):
        print(f"FAIL: missing {ref_path}")
        return False
    ref_sd = load_state_dict(ref_path)

    for d in fork_dirs:
        path = os.path.join(d, "post_load")
        meta_path = os.path.join(d, "fork_metadata.json")
        if not os.path.isfile(path):
            print(f"FAIL [{d}]: missing post_load checkpoint")
            ok = False
            continue
        sd = load_state_dict(path)
        diff = max_abs_param_diff(ref_sd, sd) if d != fork_dirs[0] else 0.0
        status = "PASS" if diff == 0.0 else "FAIL"
        if diff != 0.0:
            ok = False
        print(f"{status} [{os.path.basename(d)}]: max param diff vs. fork0 = {diff}")

        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            recorded = meta.get("post_load_checksum")
            recomputed = state_dict_checksum(sd)
            if recorded is not None and recorded != recomputed:
                print(f"FAIL [{d}]: post_load_checksum mismatch (recorded={recorded}, recomputed={recomputed})")
                ok = False
    return ok


def verify_final(fork_dirs: list) -> bool:
    print(f"\n=== final-checkpoint divergence check ({len(fork_dirs)} forks) ===")
    ok = True
    sds = {}
    for d in fork_dirs:
        path = os.path.join(d, "final")
        if not os.path.isfile(path):
            print(f"FAIL [{d}]: missing final checkpoint")
            ok = False
            continue
        sds[d] = load_state_dict(path)

    names = list(sds.keys())
    print("Pairwise max |param diff| matrix:")
    header = "".join(f"{os.path.basename(n)[:14]:>16}" for n in names)
    print(f"{'':>16}{header}")
    for a in names:
        row = []
        for b in names:
            diff = 0.0 if a == b else max_abs_param_diff(sds[a], sds[b])
            row.append(diff)
        print(f"{os.path.basename(a)[:16]:>16}" + "".join(f"{d:16.6g}" for d in row))
        for b, d in zip(names, row):
            if a != b and d == 0.0:
                print(f"FAIL: {a} and {b} are identical at 'final' — forks did not diverge")
                ok = False

    post_load_path = os.path.join(fork_dirs[0], "post_load")
    if os.path.isfile(post_load_path):
        post_load_sd = load_state_dict(post_load_path)
        for a in names:
            diff = max_abs_param_diff(post_load_sd, sds[a])
            if diff == 0.0:
                print(f"FAIL: {a}'s final checkpoint is identical to post_load — no training happened")
                ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fork-config", default="config/fork_experiment.yaml")
    ap.add_argument("--experiment-root", default=None,
                     help="Overrides fork_experiment.yaml's output_root, or (if --fork-config "
                          "doesn't exist) a directory to glob fork* dirs from directly.")
    ap.add_argument("--stage", choices=["post_load", "final", "both"], default="both")
    args = ap.parse_args()

    fork_dirs = discover_fork_dirs(args.fork_config, args.experiment_root)
    print(f"Found {len(fork_dirs)} fork checkpoint dirs:")
    for d in fork_dirs:
        print(f"  {d}")

    ok = True
    if args.stage in ("post_load", "both"):
        ok = verify_post_load(fork_dirs) and ok
    if args.stage in ("final", "both"):
        ok = verify_final(fork_dirs) and ok

    print("\n=== RESULT:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
