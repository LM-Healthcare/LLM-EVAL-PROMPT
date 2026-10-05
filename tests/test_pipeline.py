"""End-to-end: split -> run (mock) -> resume -> analyze -> select-best -> triage arm."""

import json

import pandas as pd
import yaml

from prompteval.analysis import analyse, select_best
from prompteval.config import load_config
from prompteval.dataset import load_itamed
from prompteval.runner import completed_keys, run_experiment
from prompteval.split import make_split, write_split


def _cfg(tmp_path, fake_itamed, repo_root, name="main", extra=None):
    cfg = {
        "experiment": {"name": name, "output_dir": str(tmp_path / "results" / name), "seed": 1},
        "dataset": {"path": str(fake_itamed), "languages": ["it", "en"],
                    "split_file": str(tmp_path / "split.json")},
        "prompts": {"file": str(repo_root / "prompts" / "prompts.yaml"), "language": "en",
                    "conditions": ["C0", "C1", "C2", "C3"]},
        "sampling": {"runs": 4},
        "models": [
            {"id": "big", "backend": "mock", "model": "big", "params": {"skill": 2.0},
             "seed_per_run": True},
            {"id": "small", "backend": "mock", "model": "small",
             "params": {"skill": 0.3, "parse_error_rate": 0.05}, "seed_per_run": True},
        ],
        "analysis": {"bootstrap": 50, "annotations": {"distractor_ratings": None, "urgency_gold": None}},
    }
    if extra:
        from prompteval.config import deep_merge
        cfg = deep_merge(cfg, extra)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_config(p)


def test_end_to_end(tmp_path, fake_itamed, repo_root):
    items = load_itamed(fake_itamed)
    write_split(make_split(items, 15, 3, {"question_types": ["case-based"]}), tmp_path / "split.json")
    cfg = _cfg(tmp_path, fake_itamed, repo_root)

    # partial run, then resume: no duplicates, all tasks done
    s1 = run_experiment(cfg, max_tasks=50)
    assert s1["per_model"]["big"]["ok"] == 50
    s2 = run_experiment(cfg)
    total = 15 * 2 * 4 * 4
    assert s2["per_model"]["big"]["already_done"] == 50
    out = tmp_path / "results" / "main" / "responses"
    for m in ("big", "small"):
        lines = (out / f"{m}.jsonl").read_text().strip().splitlines()
        assert len(lines) == total == len(completed_keys(out / f"{m}.jsonl"))

    # option order identical across conditions/runs/languages for an item
    recs = [json.loads(l) for l in (out / "big.jsonl").read_text().splitlines()]
    orders = {}
    for r in recs:
        orders.setdefault(r["question_code"], set()).add(tuple(r["option_order"]))
    assert all(len(v) == 1 for v in orders.values())

    res = analyse(cfg)
    m = pd.read_csv(tmp_path / "results/main/analysis/metrics_by_cell.csv")
    assert len(m) == 2 * 2 * 4 and m["accuracy"].between(0, 1).all()
    big = m[m.model_id == "big"]["accuracy"].mean()
    small = m[m.model_id == "small"]["accuracy"].mean()
    assert big > small
    c = pd.read_csv(tmp_path / "results/main/analysis/contrasts.csv")
    assert {"condition_vs_reference", "language_it_vs_en"} <= set(c["family"])
    assert (c["mcnemar_p_adj"] >= c["mcnemar_p"] - 1e-12).all()
    assert (tmp_path / "results/main/analysis/report.md").exists()

    best_path = tmp_path / "results/main/best_conditions.json"
    best = select_best(cfg, best_path)
    assert set(best["best"]) == {"big", "small"}

    # triage arm: BEST+TRIAGE compared with imported BEST runs
    tcfg = _cfg(tmp_path, fake_itamed, repo_root, "triage", {
        "prompts": {"conditions": ["BEST+TRIAGE"], "best_file": str(best_path)},
        "analysis": {"reference_condition": "BEST",
                     "import_responses": [{"dir": str(tmp_path / "results/main"), "best_alias": True}]},
    })
    tcfg["prompts"]["conditions"] = ["BEST+TRIAGE"]
    run_experiment(tcfg)
    recs = [json.loads(l) for l in (tmp_path / "results/triage/responses/big.jsonl").read_text().splitlines()]
    assert {r["condition"] for r in recs} == {best["best"]["big"] + "+TRIAGE"}
    analyse(tcfg)
    tm = pd.read_csv(tmp_path / "results/triage/analysis/metrics_by_cell.csv")
    assert set(tm["condition"]) == {"BEST", "BEST+TRIAGE"}
