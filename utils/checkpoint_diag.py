"""Shared state_dict hashing/diffing used by both the training loop (to write
checksums into fork_metadata.json) and verify_fork_checkpoints.py (to read
them back) — kept in one place so the two sides can't drift apart."""
import hashlib
from typing import Dict

import torch


def state_dict_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        h.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return f"sha256:{h.hexdigest()}"


def max_abs_param_diff(sd_a: Dict[str, torch.Tensor], sd_b: Dict[str, torch.Tensor]) -> float:
    keys_a, keys_b = set(sd_a.keys()), set(sd_b.keys())
    if keys_a != keys_b:
        raise ValueError(
            f"state_dict key mismatch: only in a={keys_a - keys_b}, only in b={keys_b - keys_a}"
        )
    max_diff = 0.0
    for key in keys_a:
        diff = (sd_a[key].detach().cpu().float() - sd_b[key].detach().cpu().float()).abs().max().item()
        max_diff = max(max_diff, diff)
    return max_diff
