"""Prompt rendering for prompting conditions (with composable add-ons)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import LETTERS
from .dataset import Item


@dataclass
class RenderedPrompt:
    system: str
    user: str
    prompt_hash: str
    expects_triage: bool
    expects_references: bool


class PromptBank:
    def __init__(self, path: str | Path):
        with open(path, encoding="utf-8") as f:
            self.data = yaml.safe_load(f)
        self.version = str(self.data.get("version", "unversioned"))
        self.file_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]

    def blocks(self, condition: str) -> list[str]:
        return [b.strip() for b in condition.split("+") if b.strip()]

    def validate_condition(self, condition: str, lang: str = "en") -> None:
        spec = self.data[lang]
        parts = self.blocks(condition)
        if not parts or parts[0] not in spec["conditions"]:
            raise ValueError(f"Unknown base condition in '{condition}'")
        for p in parts[1:]:
            if p not in spec.get("addons", {}):
                raise ValueError(f"Unknown add-on '{p}' in '{condition}'")

    def condition_name(self, condition: str, lang: str = "en") -> str:
        spec = self.data[lang]
        parts = self.blocks(condition)
        names = [spec["conditions"][parts[0]]["name"]]
        names += [spec["addons"][p]["name"] for p in parts[1:]]
        return " + ".join(names)

    def render(self, item: Item, item_lang: str, prompt_lang: str, condition: str,
               perm: list[int]) -> RenderedPrompt:
        self.validate_condition(condition, prompt_lang)
        spec = self.data[prompt_lang]
        parts = self.blocks(condition)
        instructions = [spec["conditions"][parts[0]]["instructions"].strip()]
        instructions += [spec["addons"][p]["instructions"].strip() for p in parts[1:]]
        triage = "TRIAGE" in parts[1:]
        fmt = spec["output_format"]["triage" if triage else "default"].strip()
        opts, _ = item.presented(item_lang, perm)
        options_txt = "\n".join(f"{LETTERS[k]}) {o}" for k, o in enumerate(opts))
        user = spec["template"].format(
            instructions="\n".join(instructions),
            question_label=spec["labels"]["question"],
            question=item.question[item_lang],
            options_label=spec["labels"]["options"],
            options=options_txt,
            output_format=fmt,
        ).strip()
        system = (spec.get("system") or "").strip()
        h = hashlib.sha256(json.dumps([system, user], ensure_ascii=False).encode()).hexdigest()[:16]
        return RenderedPrompt(system, user, h, triage, parts[0] == "C3")


def resolve_condition(condition: str, model_id: str, best: dict | None) -> str:
    """Replace the BEST alias with the best condition selected for this model."""
    if "BEST" not in condition.split("+"):
        return condition
    if not best or model_id not in best.get("best", {}):
        raise ValueError(
            f"Condition '{condition}' needs prompts.best_file with an entry for model "
            f"'{model_id}' (run `prompteval select-best` on the main experiment)."
        )
    base = best["best"][model_id]
    return "+".join(base if b == "BEST" else b for b in condition.split("+"))


def load_best(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
