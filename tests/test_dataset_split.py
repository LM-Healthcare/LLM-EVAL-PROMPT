from collections import Counter

import pytest

from prompteval import LETTERS
from prompteval.dataset import filter_items, load_itamed
from prompteval.prompts import PromptBank
from prompteval.split import allocate, make_split


def test_load_and_align(fake_itamed):
    items = load_itamed(fake_itamed)
    assert len(items) == 40
    it = next(iter(items.values()))
    assert it.correct_orig == 0 and len(it.options["it"]) == 5 and len(it.options["en"]) == 5


def test_permutation_is_deterministic_and_language_independent(fake_itamed):
    items = load_itamed(fake_itamed)
    it = next(iter(items.values()))
    p1, p2 = it.permutation(2026), it.permutation(2026)
    assert p1 == p2 and sorted(p1) == list(range(5))
    opts_it, c_it = it.presented("it", p1)
    opts_en, c_en = it.presented("en", p1)
    assert c_it == c_en
    assert opts_en[LETTERS.index(c_en)].startswith("correct")


def test_shuffle_spreads_correct_position(fake_itamed):
    items = load_itamed(fake_itamed)
    pos = Counter(i.presented("en", i.permutation(1))[1] for i in items.values())
    assert len(pos) >= 4  # not always A


def test_filters(fake_itamed):
    items = load_itamed(fake_itamed)
    sel = filter_items(items, {"question_types": ["case-based"], "include_images": False,
                               "years": {"min": 2018, "max": 2020}})
    assert sel and all(i.question_type == "case-based" and not i.has_image
                       and 2018 <= i.year <= 2020 for i in sel)
    sel = filter_items(items, {"years": [2025]})
    assert {i.year for i in sel} == {2025}


def test_allocation_sums_and_minimum():
    a = allocate({"a": 92, "b": 30, "c": 2, "d": 1}, 50, min_per_stratum=1)
    assert sum(a.values()) == 50 and a["d"] == 1 and a["a"] > a["b"]
    with pytest.raises(ValueError):
        allocate({"a": 3}, 5)


def test_split_disjoint_and_complete(fake_itamed):
    items = load_itamed(fake_itamed)
    s = make_split(items, 12, 42, {"question_types": ["case-based"], "include_images": False})
    ev, tr, va = set(s["eval"]), set(s["finetune"]["train"]), set(s["finetune"]["val"])
    assert len(ev) == 12 and not (ev & tr) and not (ev & va) and not (tr & va)
    assert ev | tr | va == set(items)
    assert all(items[c].question_type == "case-based" for c in ev)
    again = make_split(items, 12, 42, {"question_types": ["case-based"], "include_images": False})
    assert again["eval"] == s["eval"] and again["finetune"] == s["finetune"]  # deterministic


def test_prompts_render_all_conditions(fake_itamed, repo_root):
    bank = PromptBank(repo_root / "prompts" / "prompts.yaml")
    it = next(iter(load_itamed(fake_itamed).values()))
    perm = it.permutation(1)
    for lang in ("en", "it"):
        for cond in ("C0", "C1", "C2", "C3", "C1+TRIAGE"):
            rp = bank.render(it, "it", lang, cond, perm)
            assert "ANSWER:" in rp.user and "CONFIDENCE:" in rp.user
            assert ("URGENCY:" in rp.user) == cond.endswith("TRIAGE")
            assert rp.user.count("\nA) ") == 1 and "E) " in rp.user
    with pytest.raises(ValueError):
        bank.render(it, "en", "en", "C9", perm)
