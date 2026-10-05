"""Analysis pipeline: parse responses, compute metrics, run contrasts, write report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import LETTERS
from .annotations import load_harm_map, load_urgency_gold
from .config import output_dir
from .dataset import load_itamed
from .metrics import CellData, bootstrap_cell, paired_bootstrap
from .parsing import parse_response
from .prompts import load_best
from .stats import (adjust, cochran_q_unanimity, gee_condition_effects, gee_interactions,
                    mcnemar_majority)

CONTRAST_METRICS = ["accuracy", "majority_accuracy", "unanimity", "modal_agreement",
                    "fleiss_kappa", "oci", "auroc_verbal", "auroc_consistency", "brier", "ece",
                    "parse_failure"]
HARM_METRICS = ["mean_harm", "severe_error_rate", "confident_severe_error_rate"]
TRIAGE_METRICS = ["urgency_accuracy", "undertriage_rate", "overtriage_rate"]


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ loading
def read_responses(directory: Path) -> pd.DataFrame:
    recs = []
    for p in sorted((Path(directory) / "responses").glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") == "ok":
                    recs.append(r)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).drop_duplicates("key", keep="last")
    return df


def parse_frame(df: pd.DataFrame) -> pd.DataFrame:
    parsed = [parse_response(t, fr).to_dict() for t, fr in zip(df["response_text"], df["finish_reason"])]
    p = pd.DataFrame(parsed, index=df.index)
    out = pd.concat([df, p], axis=1)
    out["correct"] = (out["answer"] == out["correct_letter"])
    out["pred_orig"] = [
        (order[LETTERS.index(a)] if isinstance(a, str) else np.nan)
        for a, order in zip(out["answer"], out["option_order"])
    ]

    def tok(u, *names):
        if not isinstance(u, dict):
            return np.nan
        for n in names:
            if n in u and isinstance(u[n], (int, float)):
                return u[n]
        return np.nan

    out["input_tokens"] = [tok(u, "input_tokens", "prompt_tokens") for u in out["usage"]]
    out["output_tokens"] = [tok(u, "output_tokens", "completion_tokens") for u in out["usage"]]
    return out


def load_all(cfg: dict) -> pd.DataFrame:
    """Own responses plus any imported experiment (e.g. main runs for the triage arm)."""
    frames = []
    own = read_responses(output_dir(cfg))
    if len(own):
        own = own.copy()
        own["cond_label"] = [a if "BEST" in str(a) else c
                             for a, c in zip(own["condition_alias"], own["condition"])]
        own["source_experiment"] = cfg["experiment"]["name"]
        frames.append(own)
    best = load_best(cfg["prompts"].get("best_file"))
    for spec in cfg["analysis"].get("import_responses") or []:
        imp = read_responses(Path(spec["dir"]))
        if not len(imp):
            continue
        imp = imp.copy()
        if spec.get("models"):
            imp = imp[imp["model_id"].isin(spec["models"])]
        keep = pd.Series(False, index=imp.index)
        label = imp["condition"].copy()
        if spec.get("conditions"):
            keep |= imp["condition"].isin(spec["conditions"])
        if spec.get("best_alias"):
            if not best:
                raise ValueError("import_responses.best_alias needs prompts.best_file")
            bmap = best["best"]
            src_model = spec.get("best_model_map", {})  # imported model id -> id in best file
            is_best = [c == bmap.get(src_model.get(m, m)) for m, c in zip(imp["model_id"], imp["condition"])]
            is_best = pd.Series(is_best, index=imp.index)
            keep |= is_best
            label = label.where(~is_best, "BEST")
        if not spec.get("conditions") and not spec.get("best_alias"):
            keep[:] = True
        imp = imp[keep]
        imp["cond_label"] = label[keep]
        if spec.get("rename_models"):
            imp["model_id"] = imp["model_id"].replace(spec["rename_models"])
        imp["source_experiment"] = spec["dir"]
        frames.append(imp)
    if not frames:
        raise ValueError(f"No responses found in {output_dir(cfg)}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["question_code"].isin(_eval_codes(cfg, df))]
    return parse_frame(df)


def _eval_codes(cfg: dict, df: pd.DataFrame) -> set:
    """Analyse only items of the configured evaluation set (when a split is used)."""
    sf = cfg["dataset"].get("split_file")
    if sf and Path(sf).exists():
        with open(sf, encoding="utf-8") as f:
            return set(json.load(f)["eval"])
    return set(df["question_code"])


# ------------------------------------------------------------------ analysis
def build_cells(df: pd.DataFrame, theta: float, harm, urg) -> dict[tuple, CellData]:
    cells = {}
    for key, g in df.groupby(["model_id", "language", "cond_label"]):
        cells[key] = CellData(g, theta, harm, urg if "TRIAGE" in key[2] else None)
    return cells


def analyse(cfg: dict) -> dict:
    an = cfg["analysis"]
    out = output_dir(cfg) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df = load_all(cfg)
    items = load_itamed(cfg["dataset"]["path"])
    meta = pd.DataFrame([{"question_code": c, "year": i.year, "primary_specialty": i.primary_specialty,
                          "question_type": i.question_type} for c, i in items.items()])
    df = df.merge(meta, on="question_code", how="left")
    theta = float(an["oci_threshold"])
    bins = int(an["ece_bins"])
    nb = int(an["bootstrap"])
    harm = load_harm_map(an["annotations"].get("distractor_ratings"))
    urg = load_urgency_gold(an["annotations"].get("urgency_gold"))
    triage_present = df["cond_label"].str.contains("TRIAGE").any()
    cells = build_cells(df, theta, harm, urg if triage_present else None)

    # ---- per-cell metrics with cluster-bootstrap CIs
    rows = []
    for (m, l, c), cell in cells.items():
        r = bootstrap_cell(cell, nb, _seed(m, l, c), bins)
        rows.append({"model_id": m, "language": l, "condition": c, "n_items": cell.n_items,
                     "n_responses": len(cell.df), "runs_per_item": float(np.median(cell.n_runs)),
                     **r})
    metrics = pd.DataFrame(rows).sort_values(["model_id", "language", "condition"])
    metrics.to_csv(out / "metrics_by_cell.csv", index=False)

    contrast_metrics = list(CONTRAST_METRICS)
    if harm is not None:
        contrast_metrics += HARM_METRICS
    if urg is not None and triage_present:
        contrast_metrics += TRIAGE_METRICS

    # ---- contrasts
    ref = an["reference_condition"]
    con_rows = []

    def contrast(family, a_key, b_key, label_a, label_b):
        a, b = cells[a_key], cells[b_key]
        r = {"family": family, "a": label_a, "b": label_b}
        r.update(mcnemar_majority(a, b))
        r.update(paired_bootstrap(a, b, contrast_metrics, nb, _seed(family, a_key, b_key), bins))
        con_rows.append(r)

    models = sorted(df["model_id"].unique())
    langs = sorted(df["language"].unique())
    conds = sorted(df["cond_label"].unique())
    for m in models:
        for l in langs:
            if (m, l, ref) not in cells:
                continue
            for c in conds:
                if c != ref and (m, l, c) in cells:
                    contrast("condition_vs_reference", (m, l, ref), (m, l, c),
                             f"{m}/{l}/{ref}", f"{m}/{l}/{c}")
    if an.get("language_contrasts") and {"it", "en"} <= set(langs):
        for m in models:
            for c in conds:
                if (m, "en", c) in cells and (m, "it", c) in cells:
                    contrast("language_it_vs_en", (m, "en", c), (m, "it", c),
                             f"{m}/en/{c}", f"{m}/it/{c}")
    for pair in an.get("model_contrasts") or []:
        ma, mb = pair
        for l in langs:
            for c in conds:
                if (ma, l, c) in cells and (mb, l, c) in cells:
                    contrast("model_pair", (ma, l, c), (mb, l, c), f"{ma}/{l}/{c}", f"{mb}/{l}/{c}")
    contrasts = pd.DataFrame(con_rows)
    if len(contrasts):
        contrasts["mcnemar_p_adj"] = np.nan
        for fam, g in contrasts.groupby("family"):
            contrasts.loc[g.index, "mcnemar_p_adj"] = adjust(g["mcnemar_p"].tolist(), an["multiplicity"])
        contrasts.to_csv(out / "contrasts.csv", index=False)

    # ---- Cochran's Q on unanimity across conditions
    q_rows = []
    for m in models:
        for l in langs:
            sub = {c: cells[(m, l, c)] for c in conds if (m, l, c) in cells}
            if len(sub) >= 2:
                q_rows.append({"model_id": m, "language": l, **cochran_q_unanimity(sub)})
    cq = pd.DataFrame(q_rows)
    if len(cq):
        cq["p_adj"] = adjust(cq["p"].tolist(), an["multiplicity"])
        cq.to_csv(out / "cochran_q_unanimity.csv", index=False)

    # ---- GEE logistic models (response level, clustered by item)
    gee = pd.DataFrame()
    if an.get("gee", True):
        gee = gee_condition_effects(df.rename(columns={"condition": "condition_raw"})
                                    .rename(columns={"cond_label": "condition"}), ref)
        if len(gee) and "p" in gee:
            gee["p_adj"] = adjust(gee["p"].tolist(), an["multiplicity"])
            gee.to_csv(out / "gee_condition_effects.csv", index=False)
        txt = gee_interactions(df.rename(columns={"condition": "condition_raw"})
                               .rename(columns={"cond_label": "condition"}), ref)
        (out / "gee_interactions.txt").write_text(txt, encoding="utf-8")

    # ---- ex-post difficulty
    dcfg = an["difficulty"]
    dref = dcfg["reference_condition"]
    dsub = df[df["cond_label"] == dref]
    diff_tbl = pd.DataFrame()
    if len(dsub):
        err = 1 - dsub.groupby("question_code")["correct"].mean()
        cls = pd.cut(err, bins=[-0.001, dcfg["easy_max"], dcfg["hard_min"], 1.0],
                     labels=["easy", "medium", "hard"])
        diff_items = pd.DataFrame({"error_rate": err, "difficulty": cls})
        diff_items.to_csv(out / "difficulty_items.csv")
        dd = df.merge(diff_items[["difficulty"]], left_on="question_code", right_index=True, how="left")
        diff_tbl = (dd.groupby(["model_id", "language", "cond_label", "difficulty"], observed=True)
                    .agg(n_items=("question_code", "nunique"), accuracy=("correct", "mean"))
                    .reset_index())
        diff_tbl.to_csv(out / "accuracy_by_difficulty.csv", index=False)

    # ---- descriptive breakdowns
    for col, name in (("year", "accuracy_by_year.csv"), ("primary_specialty", "accuracy_by_specialty.csv")):
        (df.groupby(["model_id", "language", "cond_label", col])
         .agg(n_items=("question_code", "nunique"), accuracy=("correct", "mean"),
              parse_failure=("answer", lambda s: s.isna().mean()))
         .reset_index().to_csv(out / name, index=False))
    usage = (df.groupby(["model_id", "language", "cond_label"])
             .agg(responses=("key", "size"), input_tokens=("input_tokens", "sum"),
                  output_tokens=("output_tokens", "sum"),
                  mean_output_tokens=("output_tokens", "mean"),
                  truncated=("truncated", "sum"), mean_latency_s=("latency_s", "mean"))
             .reset_index())
    usage.to_csv(out / "usage.csv", index=False)

    # ---- reliability tables + long file for R
    rel = []
    for (m, l, c), cell in cells.items():
        t = cell.reliability_table(bins)
        t.insert(0, "condition", c); t.insert(0, "language", l); t.insert(0, "model_id", m)
        rel.append(t)
    if rel:
        pd.concat(rel).to_csv(out / "reliability_bins.csv", index=False)
    long = df[["model_id", "language", "cond_label", "question_code", "run", "correct",
               "confidence", "answer", "year", "primary_specialty"]].rename(
        columns={"cond_label": "condition"})
    long.assign(correct=long["correct"].astype(int)).to_csv(out / "responses_long.csv", index=False)
    keep = [c for c in df.columns if c not in ("response_text", "thinking_text", "usage", "request")]
    df[keep].to_csv(out / "responses_parsed.csv.gz", index=False)

    if an.get("plots", True):
        from .plots import accuracy_plot, reliability_plots
        reliability_plots(pd.concat(rel) if rel else pd.DataFrame(), out)
        accuracy_plot(metrics, out)

    report = write_report(cfg, df, metrics, contrasts, cq, gee, out, harm is not None,
                          urg is not None and triage_present)
    return {"output": str(out), "cells": len(cells), "responses": len(df), "report": report}


def select_best(cfg: dict, out_path: str | Path) -> dict:
    """Best condition per model, pooled over languages: highest accuracy, ties -> lower OCI.

    The rule is fixed a priori in METHODOLOGY.md. Only conditions without add-ons
    in this experiment are eligible.
    """
    df = load_all(cfg)
    theta = float(cfg["analysis"]["oci_threshold"])
    rows = []
    for (m, c), g in df.groupby(["model_id", "cond_label"]):
        if "+" in c or c == "BEST":
            continue
        cell = CellData(g, theta)
        r = cell.compute()
        rows.append({"model_id": m, "condition": c, "accuracy": r["accuracy"], "oci": r["oci"]})
    t = pd.DataFrame(rows)
    best = {}
    for m, g in t.groupby("model_id"):
        g = g.assign(oci_sort=g["oci"].fillna(1.0)).sort_values(
            ["accuracy", "oci_sort"], ascending=[False, True])
        best[m] = g.iloc[0]["condition"]
    res = {"rule": "highest pooled accuracy across languages; ties broken by lower OCI",
           "source_experiment": cfg["experiment"]["name"], "best": best,
           "table": t.to_dict(orient="records")}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


# ------------------------------------------------------------------ report
def _fmt(v, pct=False, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{100 * v:.1f}" if pct else f"{v:.{nd}f}"


def _ci(row, k, pct=False, nd=3):
    return f"{_fmt(row.get(k), pct, nd)} ({_fmt(row.get(k + '_lo'), pct, nd)}–{_fmt(row.get(k + '_hi'), pct, nd)})"


def write_report(cfg, df, metrics, contrasts, cq, gee, out: Path, has_harm: bool, has_triage: bool) -> str:
    lines = [f"# Results: {cfg['experiment']['name']}", ""]
    lines.append(f"- Items: {df['question_code'].nunique()}; responses: {len(df)}; "
                 f"models: {', '.join(sorted(df['model_id'].unique()))}; "
                 f"languages: {', '.join(sorted(df['language'].unique()))}; "
                 f"conditions: {', '.join(sorted(df['cond_label'].unique()))}")
    lines.append(f"- Runs per item (median): {int(df.groupby(['model_id','language','cond_label','question_code']).size().median())}")
    lines.append(f"- 95% CIs: cluster bootstrap over questions ({cfg['analysis']['bootstrap']} replicates). "
                 f"OCI threshold: {cfg['analysis']['oci_threshold']}.")
    lines += ["", "## Main metrics", "",
              "| Model | Lang | Cond | Accuracy % | Unanimity % | Fleiss κ | OCI % | AUROC (verbal) | AUROC (consistency) | ECE | Parse fail % |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        lines.append(f"| {r['model_id']} | {r['language']} | {r['condition']} | {_ci(r, 'accuracy', True)} | "
                     f"{_ci(r, 'unanimity', True)} | {_fmt(r['fleiss_kappa'])} | {_ci(r, 'oci', True)} | "
                     f"{_fmt(r['auroc_verbal'])} | {_fmt(r['auroc_consistency'])} | {_fmt(r['ece'])} | "
                     f"{_fmt(r['parse_failure'], True)} |")
    if has_harm:
        lines += ["", "## Harm (distractor-based)", "",
                  "| Model | Lang | Cond | Severe error % | Confident severe error % | Mean harm | Unrated chosen % |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in metrics.iterrows():
            lines.append(f"| {r['model_id']} | {r['language']} | {r['condition']} | "
                         f"{_ci(r, 'severe_error_rate', True)} | {_ci(r, 'confident_severe_error_rate', True)} | "
                         f"{_fmt(r.get('mean_harm'))} | {_fmt(r.get('harm_unrated_share'), True)} |")
    if has_triage:
        lines += ["", "## Triage", "",
                  "| Model | Lang | Cond | Urgency acc. % | Undertriage % | Overtriage % | κ (quadratic) |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in metrics[metrics["condition"].str.contains("TRIAGE")].iterrows():
            lines.append(f"| {r['model_id']} | {r['language']} | {r['condition']} | "
                         f"{_ci(r, 'urgency_accuracy', True)} | {_ci(r, 'undertriage_rate', True)} | "
                         f"{_ci(r, 'overtriage_rate', True)} | {_fmt(r.get('urgency_kappa_quadratic'))} |")
    if len(contrasts):
        lines += ["", "## Contrasts (b − a)", "",
                  "McNemar on item-level majority vote, adjusted within family "
                  f"({cfg['analysis']['multiplicity']}). Response-level differences with paired bootstrap CIs.", "",
                  "| Family | a | b | Δ accuracy (pp) | Δ majority acc. (pp) | Cohen's h | McNemar p (adj) | Δ OCI (pp) | Δ AUROC | Δ unanimity (pp) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in contrasts.iterrows():
            def d(k, pct=True):
                return (f"{_fmt(r.get('diff_' + k), pct)} ({_fmt(r.get('diff_' + k + '_lo'), pct)} to "
                        f"{_fmt(r.get('diff_' + k + '_hi'), pct)})")
            lines.append(f"| {r['family']} | {r['a']} | {r['b']} | {d('accuracy')} | "
                         f"{_fmt(r['majority_acc_diff'], True)} | {_fmt(r['cohens_h'])} | "
                         f"{_fmt(r['mcnemar_p'])} ({_fmt(r['mcnemar_p_adj'])}) | {d('oci')} | "
                         f"{d('auroc_verbal', False)} | {d('unanimity')} |")
    if len(cq):
        lines += ["", "## Cochran's Q on unanimity", "", "| Model | Lang | Q | df | p (adj) |", "|---|---|---|---|---|"]
        for _, r in cq.iterrows():
            lines.append(f"| {r['model_id']} | {r['language']} | {_fmt(r['q'], nd=2)} | {r['df']} | "
                         f"{_fmt(r['p'])} ({_fmt(r['p_adj'])}) |")
    if len(gee) and "odds_ratio" in gee:
        lines += ["", "## GEE logistic model (odds ratio vs reference)", "",
                  "| Model | Lang | Cond | OR (95% CI) | p (adj) |", "|---|---|---|---|---|"]
        for _, r in gee.iterrows():
            if pd.isna(r.get("odds_ratio")):
                lines.append(f"| {r['model_id']} | {r['language']} | {r['condition']} | "
                             f"not estimable: {r.get('note', '')} | – |")
                continue
            lines.append(f"| {r['model_id']} | {r['language']} | {r['condition']} | "
                         f"{r['odds_ratio']:.2f} ({r['or_lo']:.2f}–{r['or_hi']:.2f}) | "
                         f"{_fmt(r['p'])} ({_fmt(r['p_adj'])}) |")
    lines += ["", "Files: metrics_by_cell.csv, contrasts.csv, cochran_q_unanimity.csv, "
              "gee_condition_effects.csv, gee_interactions.txt, accuracy_by_*.csv, reliability_bins.csv, "
              "usage.csv, responses_long.csv (for analysis_r/glmm.R), responses_parsed.csv.gz."]
    p = out / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)
