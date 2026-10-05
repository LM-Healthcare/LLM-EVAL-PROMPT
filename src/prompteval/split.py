"""Frozen evaluation / fine-tuning split.

The evaluation set is a stratified random sample of the eligible pool
(default: case-based, text-only items), stratified by primary specialty with
proportional allocation (largest-remainder rounding) and an optional minimum
per specialty. Within each specialty, items can be over-weighted by year
(e.g. post-cutoff editions). Every item not in the evaluation set goes to the
fine-tuning pool, which is further split into train / validation.

The split file is written once and committed; experiments read it, so the test
items can never leak into a later fine-tuning study.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import Item, dataset_commit, filter_items


def allocate(sizes: dict[str, int], n: int, min_per_stratum: int = 1) -> dict[str, int]:
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"Requested {n} items but the eligible pool has {total}")
    quotas = {s: n * k / total for s, k in sizes.items()}
    alloc = {s: min(int(np.floor(q)), sizes[s]) for s, q in quotas.items()}
    for s in alloc:
        alloc[s] = max(alloc[s], min(min_per_stratum, sizes[s]))
    # distribute the remainder by largest fractional part
    while sum(alloc.values()) < n:
        cands = [s for s in alloc if alloc[s] < sizes[s]]
        s = max(cands, key=lambda x: (quotas[x] - alloc[x], sizes[x]))
        alloc[s] += 1
    # minimums may overshoot: trim strata furthest above their quota
    while sum(alloc.values()) > n:
        cands = [s for s in alloc if alloc[s] > min(min_per_stratum, sizes[s])]
        s = max(cands, key=lambda x: (alloc[x] - quotas[x], sizes[x]))
        alloc[s] -= 1
    return alloc


def make_split(
    items: dict[str, Item],
    n_eval: int,
    seed: int,
    filters: dict,
    min_per_stratum: int = 1,
    year_weights: dict | None = None,
    ft_val_fraction: float = 0.1,
    dataset_path: str | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    pool = filter_items(items, filters)
    by_spec: dict[str, list[Item]] = {}
    for it in pool:
        by_spec.setdefault(it.primary_specialty, []).append(it)
    alloc = allocate({s: len(v) for s, v in by_spec.items()}, n_eval, min_per_stratum)

    year_weights = {int(k): float(v) for k, v in (year_weights or {}).items()}
    eval_codes: list[str] = []
    for spec in sorted(by_spec):
        members = sorted(by_spec[spec], key=lambda x: x.code)
        k = alloc[spec]
        if k == 0:
            continue
        w = np.array([year_weights.get(m.year, 1.0) for m in members], dtype=float)
        idx = rng.choice(len(members), size=k, replace=False, p=w / w.sum())
        eval_codes.extend(members[i].code for i in idx)

    eval_set = set(eval_codes)
    order = sorted(items.values(), key=lambda x: (x.year, x.number))
    eval_sorted = [i.code for i in order if i.code in eval_set]
    rest = [i.code for i in order if i.code not in eval_set]
    perm = rng.permutation(len(rest))
    n_val = int(round(len(rest) * ft_val_fraction))
    val = sorted([rest[i] for i in perm[:n_val]], key=lambda c: (items[c].year, items[c].number))
    train = [c for c in rest if c not in set(val)]

    def summary(codes):
        sel = [items[c] for c in codes]
        return {
            "n": len(sel),
            "by_year": dict(sorted(Counter(i.year for i in sel).items())),
            "by_type": dict(Counter(i.question_type for i in sel)),
            "with_image": sum(i.has_image for i in sel),
            "by_primary_specialty": dict(Counter(i.primary_specialty for i in sel).most_common()),
        }

    return {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_commit": dataset_commit(dataset_path) if dataset_path else None,
        "params": {
            "n_eval": n_eval,
            "seed": seed,
            "filters": filters,
            "min_per_stratum": min_per_stratum,
            "year_weights": year_weights,
            "ft_val_fraction": ft_val_fraction,
            "stratification": "primary specialty, proportional allocation (largest remainder)",
        },
        "eval": eval_sorted,
        "finetune": {"train": train, "val": val},
        "summary": {
            "eligible_pool": len(pool),
            "eval": summary(eval_sorted),
            "finetune_train": summary(train),
            "finetune_val": summary(val),
        },
    }


def write_split(split: dict, path: str | Path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists. The evaluation split is frozen; use --force only "
            "if no experiment has been run on it yet."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
