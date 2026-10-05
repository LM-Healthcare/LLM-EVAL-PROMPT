"""Experiment configuration: YAML loading with defaults and `extends` inheritance."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "experiment",
        "output_dir": None,  # default: results/<name>
        "seed": 42,
    },
    "dataset": {
        "path": "data/ITAMed",
        "commit": None,  # expected ITAMed git commit; a mismatch raises a warning
        "languages": ["en"],
        "split_file": None,  # frozen split; when set, items = split["eval"]
        "filters": {
            "years": None,  # explicit list [2024, 2025] or range {min: 2017, max: 2021}
            "question_types": ["case-based"],
            "include_images": False,
            "specialties": None,  # primary specialty (English names)
            "question_codes": None,
        },
        "limit": None,  # keep only the first N items (smoke tests)
        "shuffle_options": True,
        "shuffle_seed": 2026,
    },
    "prompts": {
        "file": "prompts/prompts.yaml",
        "language": "en",  # en | it | match (same language as the item)
        "conditions": ["C0", "C1", "C2", "C3"],
        "best_file": None,  # needed when a condition uses the BEST alias
        "best_model_map": {},  # model id -> id whose best condition it inherits
    },
    "sampling": {
        "runs": 10,
    },
    "models": [],
    "runner": {
        "max_retries": 6,
        "retry_base_delay": 2.0,
        "retry_max_delay": 120.0,
        "order": "run_major",  # run_major | item_major
    },
    "analysis": {
        "reference_condition": "C0",
        "oci_threshold": 80,
        "ece_bins": 10,
        "bootstrap": 1000,
        "alpha": 0.05,
        "multiplicity": "holm",
        "language_contrasts": True,
        "model_contrasts": [],  # list of [model_id_a, model_id_b]
        "difficulty": {
            "reference_condition": "C0",
            "easy_max": 0.2,  # error proportion <= easy_max -> easy
            "hard_min": 0.6,  # error proportion > hard_min -> hard
        },
        "import_responses": [],  # [{dir: results/main, best_alias: true, conditions: [...]}]
        "annotations": {
            "distractor_ratings": "annotations/distractor_harm_final.csv",
            "urgency_gold": "annotations/urgency_final.csv",
        },
        "gee": True,
        "plots": True,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "extends" in raw:
        parent = Path(raw.pop("extends"))
        if not parent.is_absolute():
            parent = path.parent / parent
        # dicts are merged recursively; lists (models, conditions...) are replaced
        return deep_merge(_load_raw(parent), raw)
    return deep_merge(DEFAULTS, raw)


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = _load_raw(path)
    # the output directory is never inherited from a parent config
    with open(path, encoding="utf-8") as f:
        own = yaml.safe_load(f) or {}
    cfg["experiment"]["output_dir"] = (own.get("experiment") or {}).get("output_dir")
    if not cfg["experiment"].get("output_dir"):
        cfg["experiment"]["output_dir"] = f"results/{cfg['experiment']['name']}"
    cfg["_config_path"] = str(path)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    langs = cfg["dataset"]["languages"]
    for lang in langs:
        if lang not in ("it", "en"):
            raise ValueError(f"Unsupported dataset language: {lang}")
    if cfg["prompts"]["language"] not in ("it", "en", "match"):
        raise ValueError("prompts.language must be 'it', 'en' or 'match'")
    if int(cfg["sampling"]["runs"]) < 1:
        raise ValueError("sampling.runs must be >= 1")
    ids = [m.get("id") for m in cfg["models"]]
    if any(i is None for i in ids):
        raise ValueError("every model needs an 'id'")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate model ids: {ids}")
    for m in cfg["models"]:
        if m.get("backend") not in ("anthropic", "openai", "openai_compatible", "mock"):
            raise ValueError(f"model {m['id']}: unknown backend {m.get('backend')}")


def output_dir(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"])
