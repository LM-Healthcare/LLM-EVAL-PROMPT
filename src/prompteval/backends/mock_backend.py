"""Simulated model for tests and dry runs (no network).

The runner passes the item code, condition, language and correct letter as
keyword arguments, so the whole pipeline (parsing, metrics, stats, annotation
workflows) can be exercised end to end with realistic patterns: item
difficulty, condition effects, overconfidence and occasional format errors.
Its numbers mean nothing about real models.
"""

from __future__ import annotations

import hashlib
import random

from .. import LETTERS
from .base import Backend, Generation

CONDITION_EFFECT = {"C0": 0.0, "C1": 0.25, "C2": 0.1, "C3": 0.05}


def _h(*parts) -> float:
    d = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(d[:12], 16) / 16**12


class MockBackend(Backend):
    supports_seed = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.skill = float(self.params.get("skill", 1.0))
        self.parse_error_rate = float(self.params.get("parse_error_rate", 0.01))

    def generate(self, system, user, image_path=None, seed=None, *, code="", condition="C0",
                 lang="en", correct="A") -> Generation:
        import math

        rng = random.Random(f"{self.model}|{code}|{condition}|{lang}|{seed}")
        difficulty = _h("difficulty", code) * 4 - 1.5  # item latent difficulty
        base = condition.split("+")[0]
        eff = CONDITION_EFFECT.get(base, 0.0) + (-0.2 if lang == "it" else 0.0)
        p = 1 / (1 + math.exp(-(self.skill + eff - difficulty)))
        wrong = [c for c in LETTERS if c != correct]
        # a preferred distractor per item makes errors consistent across runs
        pref = wrong[int(_h("pref", code) * 4)]
        if rng.random() < p:
            ans = correct
        else:
            ans = pref if rng.random() < 0.7 else rng.choice(wrong)
        conf = int(min(100, max(30, rng.gauss(88 if base in ("C0", "C3") else 78, 9))))
        lines = []
        if base == "C1":
            lines.append("Key findings: ... Differential: 1) ... 2) ... Next step: ...")
        if base == "C3":
            lines += ["REFERENCES:", "- ESC Guidelines on acute coronary syndromes, 2023",
                      "- no verifiable source"]
        if "TRIAGE" in condition:
            urg = ["emergency", "urgent", "non-urgent"][int(_h("urg", code, seed) * 3)]
            lines += [f"URGENCY: {urg}", "NEXT_STEP: obtain an ECG"]
        if rng.random() < self.parse_error_rate:
            lines.append(f"I think the answer is {ans}.")
        else:
            lines += [f"ANSWER: {ans}", f"CONFIDENCE: {conf}"]
        return Generation(text="\n".join(lines), finish_reason="stop", model_returned=self.model,
                          usage={"input_tokens": len(user) // 4, "output_tokens": 40})
