"""Synthetic ITAMed-shaped dataset so tests run without the real data."""

import json
from pathlib import Path

import pytest

SPECS = ["Cardiology and Cardiac Surgery", "Pediatrics", "Nephrology", "Psychiatry"]


def _write_fake_itamed(root: Path, n: int = 40) -> Path:
    it, en = [], []
    for k in range(n):
        year = 2017 + k % 9
        base = {
            "year": year, "question_number": k + 1, "question_code": f"ssm{year}{k:03d}",
            "correct_answer": "A", "has_image": k % 13 == 0,
            "image_category": "", "image_path": "",
            "question_type": "case-based" if k % 4 else "knowledge-based",
        }
        spec = SPECS[k % len(SPECS)]
        it.append({**base, "question": f"Caso clinico {k}: paziente con sintomi.",
                   "answer_a": f"corretta {k}", "answer_b": "distrattore b", "answer_c": "distrattore c",
                   "answer_d": "distrattore d", "answer_e": "distrattore e", "category": spec})
        en.append({**base, "question": f"Clinical case {k}: patient with symptoms.",
                   "answer_a": f"correct {k}", "answer_b": "distractor b", "answer_c": "distractor c",
                   "answer_d": "distractor d", "answer_e": "distractor e", "category": spec})
    for lang, data, name in (("IT", it, "ITAMed_complete.json"), ("EN", en, "ITAMed_complete_EN.json")):
        d = root / "Dataset" / lang / "json"
        d.mkdir(parents=True)
        (d / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return root


@pytest.fixture()
def fake_itamed(tmp_path):
    return _write_fake_itamed(tmp_path / "ITAMed")


@pytest.fixture()
def repo_root():
    return Path(__file__).resolve().parents[1]
