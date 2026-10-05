"""ITAMed loading, filtering and deterministic answer-option shuffling.

ITAMed stores the correct option in position A (official SSM release format).
Every item is shown to the models with a fixed permutation of its five options,
derived from (shuffle_seed, question_code). The permutation is identical across
languages, prompting conditions, models and runs, so that run-to-run variability
reflects the model and not the option position.

Options are tracked by their *original* index (0..4, 0 = original A = correct in
the official release), which is the stable identity used for distractor harm
ratings.
"""

from __future__ import annotations

import json
import random
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from . import LETTERS

LANG_FILES = {
    "it": "Dataset/IT/json/ITAMed_complete.json",
    "en": "Dataset/EN/json/ITAMed_complete_EN.json",
}
ITAMED_REPO = "https://github.com/LM-Healthcare/ITAMed.git"


@dataclass
class Item:
    code: str
    year: int
    number: int
    question_type: str
    specialties: list[str]  # English names, 1-2
    has_image: bool
    image_category: str
    image_path: str
    question: dict[str, str] = field(default_factory=dict)  # lang -> text
    options: dict[str, list[str]] = field(default_factory=dict)  # lang -> 5 options, original order
    correct_orig: int = 0  # original index of the correct option

    @property
    def primary_specialty(self) -> str:
        return self.specialties[0] if self.specialties else "Unknown"

    def permutation(self, seed: int, shuffle: bool = True) -> list[int]:
        """perm[k] = original index of the option shown at position k."""
        if not shuffle:
            return list(range(5))
        rng = random.Random(f"{seed}:{self.code}")
        perm = list(range(5))
        rng.shuffle(perm)
        return perm

    def presented(self, lang: str, perm: list[int]) -> tuple[list[str], str]:
        """Options in presented order and the letter of the correct one."""
        opts = [self.options[lang][i] for i in perm]
        correct_letter = LETTERS[perm.index(self.correct_orig)]
        return opts, correct_letter


def _split_categories(raw: str) -> list[str]:
    return [c.strip() for c in (raw or "").split(";") if c.strip()]


def dataset_commit(path: str | Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def load_itamed(path: str | Path, expected_commit: str | None = None) -> dict[str, Item]:
    """Load both language versions and align them by question code."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"ITAMed not found at {path}. Run `prompteval fetch-dataset` first."
        )
    if expected_commit:
        actual = dataset_commit(path)
        if actual and not actual.startswith(expected_commit) and not expected_commit.startswith(actual):
            warnings.warn(
                f"ITAMed commit mismatch: expected {expected_commit}, found {actual}",
                stacklevel=2,
            )

    raw = {}
    for lang, rel in LANG_FILES.items():
        with open(path / rel, encoding="utf-8") as f:
            raw[lang] = {q["question_code"]: q for q in json.load(f)}

    if set(raw["it"]) != set(raw["en"]):
        raise ValueError("Italian and English files are not aligned by question_code")

    items: dict[str, Item] = {}
    for code, q_it in raw["it"].items():
        q_en = raw["en"][code]
        correct = q_it["correct_answer"].strip().upper()
        if correct != q_en["correct_answer"].strip().upper():
            raise ValueError(f"{code}: correct answer differs between languages")
        item = Item(
            code=code,
            year=int(q_it["year"]),
            number=int(q_it["question_number"]),
            question_type=q_it.get("question_type", ""),
            specialties=_split_categories(q_en.get("category", "")),
            has_image=bool(q_it.get("has_image", False)),
            image_category=q_en.get("image_category", "") or "",
            image_path=q_it.get("image_path", "") or "",
            correct_orig=LETTERS.index(correct),
        )
        for lang, q in (("it", q_it), ("en", q_en)):
            item.question[lang] = q["question"].strip()
            item.options[lang] = [q[f"answer_{c.lower()}"].strip() for c in LETTERS]
        items[code] = item
    return items


def _year_ok(year: int, years) -> bool:
    """`years` is None, an explicit list ([2024, 2025]) or a range ({min: 2017, max: 2021})."""
    if years is None:
        return True
    if isinstance(years, dict):
        return years.get("min", -10**9) <= year <= years.get("max", 10**9)
    return year in {int(y) for y in years}


def filter_items(items: dict[str, Item], filters: dict) -> list[Item]:
    out = []
    qtypes = filters.get("question_types")
    specs = filters.get("specialties")
    codes = filters.get("question_codes")
    for it in items.values():
        if not _year_ok(it.year, filters.get("years")):
            continue
        if qtypes and it.question_type not in qtypes:
            continue
        if not filters.get("include_images", False) and it.has_image:
            continue
        if specs and it.primary_specialty not in specs:
            continue
        if codes and it.code not in set(codes):
            continue
        out.append(it)
    out.sort(key=lambda x: (x.year, x.number))
    return out


def select_items(cfg: dict, items: dict[str, Item] | None = None) -> list[Item]:
    """Items to evaluate in an experiment: from the frozen split or from the filters."""
    ds = cfg["dataset"]
    if items is None:
        items = load_itamed(ds["path"], ds.get("commit"))
    if ds.get("split_file"):
        with open(ds["split_file"], encoding="utf-8") as f:
            split = json.load(f)
        codes = split["eval"]
        missing = [c for c in codes if c not in items]
        if missing:
            raise ValueError(f"{len(missing)} split codes not in dataset, e.g. {missing[:3]}")
        selected = [items[c] for c in codes]
        # filters still apply on top of the split (e.g. a single year for a pilot)
        f = dict(ds.get("filters") or {})
        f["question_types"] = f.get("question_types") or None
        keep = {i.code for i in filter_items({i.code: i for i in selected}, f)}
        selected = [i for i in selected if i.code in keep]
    else:
        selected = filter_items(items, ds.get("filters") or {})
    if ds.get("limit"):
        selected = selected[: int(ds["limit"])]
    if not selected:
        raise ValueError("No items selected: check dataset filters / split file")
    return selected


def fetch_dataset(dest: str | Path, ref: str | None = None) -> str:
    """Clone the public ITAMed repository and optionally check out a pinned commit."""
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", ITAMED_REPO, str(dest)], check=True)
    if ref:
        subprocess.run(["git", "-C", str(dest), "fetch", "--all", "--tags"], check=False)
        subprocess.run(["git", "-C", str(dest), "checkout", ref], check=True)
    return dataset_commit(dest) or "unknown"
