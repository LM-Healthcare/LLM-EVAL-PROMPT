"""Inferential statistics: paired tests, GEE logistic models, multiplicity."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from .metrics import CellData


def cohens_h(p1: float, p2: float) -> float:
    return float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))


def mcnemar_majority(a: CellData, b: CellData) -> dict:
    """McNemar on item-level majority-vote correctness (b vs a), paired by item."""
    common = np.intersect1d(a.items, b.items)
    sa = pd.Series(a.majority_correct, index=a.items).loc[common].to_numpy()
    sb = pd.Series(b.majority_correct, index=b.items).loc[common].to_numpy()
    n01 = int(np.sum((sa == 1) & (sb == 0)))  # a right, b wrong
    n10 = int(np.sum((sa == 0) & (sb == 1)))  # a wrong, b right
    n11 = int(np.sum((sa == 1) & (sb == 1)))
    n00 = int(np.sum((sa == 0) & (sb == 0)))
    table = [[n11, n01], [n10, n00]]
    exact = (n01 + n10) < 25
    res = mcnemar(table, exact=exact, correction=not exact)
    pa, pb = sa.mean(), sb.mean()
    return {
        "n_items": int(len(common)), "a_right_b_wrong": n01, "a_wrong_b_right": n10,
        "majority_acc_a": float(pa), "majority_acc_b": float(pb),
        "majority_acc_diff": float(pb - pa), "cohens_h": cohens_h(pa, pb),
        "mcnemar_exact": exact, "mcnemar_stat": float(res.statistic), "mcnemar_p": float(res.pvalue),
    }


def cochran_q_unanimity(cells: dict[str, CellData]) -> dict:
    """Cochran's Q on the unanimity indicator across conditions (items in all cells)."""
    names = list(cells)
    common = cells[names[0]].items
    for n in names[1:]:
        common = np.intersect1d(common, cells[n].items)
    if len(names) < 2 or len(common) < 2:
        return {"q": np.nan, "df": np.nan, "p": np.nan, "n_items": int(len(common))}
    x = np.column_stack([
        pd.Series(cells[n].unanimous, index=cells[n].items).loc[common].to_numpy() for n in names
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cochrans_q(x)
    return {"q": float(res.statistic), "df": int(res.df), "p": float(res.pvalue),
            "n_items": int(len(common)), "conditions": ";".join(names)}


def adjust(pvals: list[float], method: str = "holm") -> list[float]:
    p = np.array(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method=method)[1]
    return out.tolist()


def gee_condition_effects(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Per model x language: logistic GEE of correctness on condition, clustered by item.

    Returns odds ratios of each condition vs the reference with robust 95% CIs.
    Population-averaged counterpart of the mixed-effects model fitted in
    analysis_r/glmm.R.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    rows = []
    for (model, lang), g in df.groupby(["model_id", "language"]):
        conds = sorted(g["condition"].unique())
        if reference not in conds or len(conds) < 2:
            continue
        g = g.assign(correct=g["correct"].astype(int))
        acc = g.groupby("condition")["correct"].mean()

        def not_estimable(cond, why):
            rows.append({"model_id": model, "language": lang, "condition": cond,
                         "reference": reference, "odds_ratio": np.nan, "or_lo": np.nan,
                         "or_hi": np.nan, "p": np.nan, "estimable": False, "note": why})

        # complete separation (0% or 100% correct) makes the logistic estimate infinite
        if acc[reference] in (0.0, 1.0):
            for c in conds:
                if c != reference:
                    not_estimable(c, f"reference accuracy {acc[reference]:.0%} (separation)")
            continue
        separated = [c for c in conds if c != reference and acc[c] in (0.0, 1.0)]
        for c in separated:
            not_estimable(c, f"accuracy {acc[c]:.0%} (separation)")
        keep = [c for c in conds if c not in separated]
        if len(keep) < 2:
            continue
        sub = g[g["condition"].isin(keep)]
        formula = f"correct ~ C(condition, Treatment(reference='{reference}'))"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.gee(formula, groups="question_code", data=sub,
                              family=sm.families.Binomial(),
                              cov_struct=sm.cov_struct.Exchangeable()).fit()
                ci = fit.conf_int()
                bse, pvals = fit.bse, fit.pvalues
        except Exception as e:  # pragma: no cover
            for c in keep:
                if c != reference:
                    not_estimable(c, f"fit failed: {e}")
            continue
        for term in fit.params.index:
            if term == "Intercept":
                continue
            cond = term.split("[T.")[-1].rstrip("]")
            coef, se = fit.params[term], bse[term]
            if not np.isfinite(se) or se <= 0 or abs(coef) > 10:
                not_estimable(cond, "unstable estimate (no discordant responses or quasi-separation)")
                continue
            rows.append({
                "model_id": model, "language": lang, "condition": cond, "reference": reference,
                "odds_ratio": float(np.exp(coef)),
                "or_lo": float(np.exp(ci.loc[term, 0])), "or_hi": float(np.exp(ci.loc[term, 1])),
                "p": float(pvals[term]), "estimable": True, "note": "",
            })
    return pd.DataFrame(rows)


def gee_interactions(df: pd.DataFrame, reference: str) -> str:
    """Joint GEE: condition x model x language; Wald tests for each term."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = df.assign(correct=df["correct"].astype(int))
    cell_acc = d.groupby(["model_id", "language", "condition"])["correct"].mean()
    sep = cell_acc[(cell_acc == 0) | (cell_acc == 1)]
    if len(sep):
        return ("Joint GEE not fitted: complete separation (0% or 100% accuracy) in cells\n"
                + sep.to_string())
    factors = [f"C(condition, Treatment(reference='{reference}'))"]
    if d["model_id"].nunique() > 1:
        factors.append("C(model_id)")
    if d["language"].nunique() > 1:
        factors.append("C(language)")
    formula = "correct ~ " + " * ".join(factors)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.gee(formula, groups="question_code", data=d, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit()
            wald = fit.wald_test_terms(skip_single=False, scalar=True)
        return f"Formula: {formula}\nClusters: question_code (exchangeable)\n\n{wald}\n\n{fit.summary()}"
    except Exception as e:  # pragma: no cover
        return f"GEE interaction model failed: {e}"


def interrater(r1: pd.Series, r2: pd.Series, ordinal: bool = True) -> dict:
    m = r1.notna() & r2.notna()
    a, b = r1[m].astype(str), r2[m].astype(str)
    out = {"n": int(m.sum()), "percent_agreement": float((a == b).mean()) if m.any() else np.nan,
           "cohen_kappa": float(cohen_kappa_score(a, b)) if m.sum() > 1 else np.nan}
    if ordinal and m.sum() > 1:
        try:
            out["cohen_kappa_linear"] = float(cohen_kappa_score(
                r1[m].astype(int), r2[m].astype(int), weights="linear"))
        except (ValueError, TypeError):
            pass
    return out
