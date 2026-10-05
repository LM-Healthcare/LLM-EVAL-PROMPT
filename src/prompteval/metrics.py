"""Outcome metrics, computed per cell (model x language x condition).

All metrics accept item weights so the same code gives point estimates
(weights = 1) and cluster-bootstrap replicates (weights = multinomial counts
over items). Questions are the resampling unit because runs of the same
question are not independent.

Response level
  accuracy            correct / all responses (a parsing failure counts as wrong)
  parse_failure       responses without a valid ANSWER line
  OCI                 wrong answers with confidence >= theta / wrong answers with a confidence
  AUROC (verbal)      discrimination of correct vs wrong by stated confidence
  AUROC (consistency) same, using the share of runs that gave the same answer
  Brier, ECE          calibration of stated confidence (confidence / 100)
Item level
  unanimity           all runs gave the same valid answer
  modal agreement     share of runs agreeing with the most frequent answer
  majority accuracy   more than half of the runs are correct
  Fleiss' kappa       agreement between runs (runs = raters; categories A-E + invalid)
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from .parsing import URGENCY_LEVELS

CATS = ["A", "B", "C", "D", "E", "INVALID"]


class CellData:
    """Pre-computed arrays for one cell, for fast weighted recomputation."""

    def __init__(self, df: pd.DataFrame, theta: float, harm: dict | None = None,
                 urgency_gold: dict | None = None):
        df = df.reset_index(drop=True)
        self.df = df
        self.theta = theta
        self.items = np.array(sorted(df["question_code"].unique()))
        idx = {c: i for i, c in enumerate(self.items)}
        self.item_idx = df["question_code"].map(idx).to_numpy()
        self.n_items = len(self.items)
        self.valid = df["answer"].notna().to_numpy()
        self.correct = df["correct"].to_numpy().astype(float)
        conf = df["confidence"].to_numpy(dtype=float)
        self.has_conf = self.valid & ~np.isnan(conf)
        self.conf = np.where(self.has_conf, conf, np.nan)

        # item-level counts
        counts = np.zeros((self.n_items, len(CATS)))
        cat_idx = df["answer"].fillna("INVALID").map({c: i for i, c in enumerate(CATS)}).to_numpy()
        np.add.at(counts, (self.item_idx, cat_idx), 1)
        self.counts = counts
        self.n_runs = counts.sum(axis=1)
        self.n_correct = np.bincount(self.item_idx, weights=self.correct, minlength=self.n_items)
        valid_counts = counts[:, :5]
        self.unanimous = (valid_counts.max(axis=1) == self.n_runs).astype(float)
        self.modal_share = valid_counts.max(axis=1) / self.n_runs
        self.majority_correct = (self.n_correct > self.n_runs / 2).astype(float)
        max_runs = self.n_runs.max() if self.n_items else 0
        self.complete = self.n_runs == max_runs

        # consistency-based confidence of each response
        same = counts[self.item_idx, cat_idx]
        self.consistency = np.where(self.valid, same / self.n_runs[self.item_idx], np.nan)

        # harm of each response (0 when correct; distractor rating when wrong)
        self.harm = None
        if harm is not None:
            h = np.full(len(df), np.nan)
            for k, (code, corr, val, po) in enumerate(zip(
                    df["question_code"], df["correct"], self.valid, df["pred_orig"])):
                if not val:
                    continue
                h[k] = 0.0 if corr else harm.get((code, int(po)), np.nan)
            self.harm = h

        # triage
        self.urg_gold = None
        if urgency_gold is not None and "urgency" in df:
            order = {u: i for i, u in enumerate(URGENCY_LEVELS)}
            g = df["question_code"].map(lambda c: order.get(urgency_gold.get(c), np.nan))
            p = df["urgency"].map(lambda u: order.get(u, np.nan) if isinstance(u, str) else np.nan)
            self.urg_gold = g.to_numpy(dtype=float)
            self.urg_pred = p.to_numpy(dtype=float)

    # ------------------------------------------------------------------
    def compute(self, item_w: np.ndarray | None = None, bins: int = 10) -> dict:
        if item_w is None:
            item_w = np.ones(self.n_items)
        w = item_w[self.item_idx]
        out: dict[str, float] = {}
        sw = w.sum()
        out["accuracy"] = _safe(np.sum(w * self.correct), sw)
        out["parse_failure"] = _safe(np.sum(w * ~self.valid), sw)
        out["missing_confidence"] = _safe(np.sum(w * (self.valid & ~self.has_conf)), np.sum(w * self.valid))

        iw = item_w.sum()
        out["majority_accuracy"] = _safe(np.sum(item_w * self.majority_correct), iw)
        out["unanimity"] = _safe(np.sum(item_w * self.unanimous), iw)
        out["modal_agreement"] = _safe(np.sum(item_w * self.modal_share), iw)
        out["fleiss_kappa"] = fleiss_kappa_weighted(self.counts[self.complete], item_w[self.complete])

        # calibration
        hc = self.has_conf
        wrong = hc & (self.correct == 0)
        out["oci"] = _safe(np.sum(w[wrong] * (self.conf[wrong] >= self.theta)), np.sum(w[wrong]))
        out["mean_conf_correct"] = _wmean(self.conf[hc & (self.correct == 1)], w[hc & (self.correct == 1)])
        out["mean_conf_wrong"] = _wmean(self.conf[wrong], w[wrong])
        y, s, ww = self.correct[hc], self.conf[hc] / 100.0, w[hc]
        out["auroc_verbal"] = _auc(y, s, ww)
        out["brier"] = _wmean((s - y) ** 2, ww)
        out["ece"] = ece(y, s, ww, bins)
        v = self.valid
        out["auroc_consistency"] = _auc(self.correct[v], self.consistency[v], w[v])
        out["brier_consistency"] = _wmean((self.consistency[v] - self.correct[v]) ** 2, w[v])

        if self.harm is not None:
            known = ~np.isnan(self.harm)
            out["harm_unrated_share"] = _safe(np.sum(w * (v & ~known)), np.sum(w * v))
            out["mean_harm"] = _wmean(self.harm[known], w[known])
            out["severe_error_rate"] = _safe(np.sum(w[known] * (self.harm[known] == 2)), sw)
            sev_conf = known & (np.nan_to_num(self.harm) == 2) & hc & (np.nan_to_num(self.conf) >= self.theta)
            out["confident_severe_error_rate"] = _safe(np.sum(w * sev_conf), sw)

        if self.urg_gold is not None:
            g = ~np.isnan(self.urg_gold)
            gp = g & ~np.isnan(self.urg_pred)
            out["urgency_parse_failure"] = _safe(np.sum(w[g] * np.isnan(self.urg_pred[g])), np.sum(w[g]))
            if gp.any():
                d = self.urg_pred[gp] - self.urg_gold[gp]
                out["urgency_accuracy"] = _wmean((d == 0).astype(float), w[gp])
                out["undertriage_rate"] = _wmean((d < 0).astype(float), w[gp])
                out["overtriage_rate"] = _wmean((d > 0).astype(float), w[gp])
                out["urgency_kappa_quadratic"] = quadratic_kappa(
                    self.urg_gold[gp].astype(int), self.urg_pred[gp].astype(int), w[gp], 3)
        return out

    def reliability_table(self, bins: int = 10) -> pd.DataFrame:
        hc = self.has_conf
        s = self.conf[hc] / 100.0
        y = self.correct[hc]
        edges = np.linspace(0, 1, bins + 1)
        b = np.clip(np.digitize(s, edges[1:-1], right=False), 0, bins - 1)
        rows = []
        for k in range(bins):
            m = b == k
            rows.append({"bin_low": edges[k], "bin_high": edges[k + 1], "n": int(m.sum()),
                         "mean_conf": float(s[m].mean()) if m.any() else np.nan,
                         "accuracy": float(y[m].mean()) if m.any() else np.nan})
        return pd.DataFrame(rows)


def _safe(num, den) -> float:
    return float(num / den) if den > 0 else np.nan


def _wmean(x, w) -> float:
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x)
    return _safe(np.sum(x[m] * w[m]), np.sum(w[m]))


def _auc(y, s, w) -> float:
    """Weighted ROC-AUC (Mann-Whitney with ties counted 1/2); equals sklearn's roc_auc_score."""
    m = (w > 0) & ~np.isnan(s)
    y, s, w = y[m], s[m], w[m]
    pos = y == 1
    wp_tot, wn_tot = w[pos].sum(), w[~pos].sum()
    if wp_tot == 0 or wn_tot == 0:
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    wp = np.bincount(inv, weights=w * pos, minlength=len(u))
    wn = np.bincount(inv, weights=w * ~pos, minlength=len(u))
    wn_below = np.concatenate([[0.0], np.cumsum(wn)[:-1]])
    return float(np.sum(wp * (wn_below + 0.5 * wn)) / (wp_tot * wn_tot))


def quadratic_kappa(a: np.ndarray, b: np.ndarray, w: np.ndarray, k: int) -> float:
    """Weighted Cohen's kappa with quadratic weights (matches sklearn, with sample weights)."""
    conf = np.zeros((k, k))
    np.add.at(conf, (a, b), w)
    tot = conf.sum()
    if tot == 0:
        return np.nan
    expected = np.outer(conf.sum(axis=1), conf.sum(axis=0)) / tot
    i, j = np.indices((k, k))
    wts = (i - j) ** 2 / (k - 1) ** 2
    den = np.sum(wts * expected)
    return float(1 - np.sum(wts * conf) / den) if den > 0 else np.nan


def ece(y, s, w, bins: int = 10) -> float:
    if len(y) == 0 or w.sum() == 0:
        return np.nan
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(s, edges[1:-1], right=False), 0, bins - 1)
    tot = w.sum()
    e = 0.0
    for k in range(bins):
        m = b == k
        wk = w[m].sum()
        if wk == 0:
            continue
        e += wk / tot * abs(np.sum(w[m] * y[m]) / wk - np.sum(w[m] * s[m]) / wk)
    return float(e)


def fleiss_kappa_weighted(counts: np.ndarray, item_w: np.ndarray) -> float:
    """Fleiss' kappa with item weights (weights = 1 gives the standard statistic)."""
    if counts.size == 0:
        return np.nan
    n = counts.sum(axis=1)
    if np.any(n != n[0]) or n[0] < 2:
        return np.nan
    n = n[0]
    W = item_w.sum()
    if W == 0:
        return np.nan
    p_j = (item_w[:, None] * counts).sum(axis=0) / (W * n)
    P_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = np.sum(item_w * P_i) / W
    P_e = np.sum(p_j ** 2)
    if np.isclose(P_e, 1.0):
        return 1.0 if np.isclose(P_bar, 1.0) else np.nan
    return float((P_bar - P_e) / (1 - P_e))


def bootstrap_cell(cell: CellData, n_boot: int, seed: int, bins: int = 10) -> dict:
    """Point estimates plus 95% percentile intervals (cluster bootstrap on items)."""
    point = cell.compute(bins=bins)
    if n_boot <= 0 or cell.n_items < 2:
        return {**point, **{f"{k}_lo": np.nan for k in point}, **{f"{k}_hi": np.nan for k in point}}
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in point}
    for _ in range(n_boot):
        wts = rng.multinomial(cell.n_items, np.full(cell.n_items, 1 / cell.n_items)).astype(float)
        r = cell.compute(wts, bins=bins)
        for k in point:
            reps[k].append(r.get(k, np.nan))
    out = dict(point)
    for k, vals in reps.items():
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        out[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if len(arr) else np.nan
        out[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if len(arr) else np.nan
    return out


def paired_bootstrap(a: CellData, b: CellData, metrics: list[str], n_boot: int, seed: int,
                     bins: int = 10) -> dict:
    """Difference b - a on the items both cells share, resampling items jointly."""
    common = np.intersect1d(a.items, b.items)
    ia = np.isin(a.items, common)
    ib = np.isin(b.items, common)
    # restrict both cells to the common items through zero weights
    wa0 = ia.astype(float)
    wb0 = ib.astype(float)
    pa, pb = a.compute(wa0, bins), b.compute(wb0, bins)
    out = {"n_common_items": int(len(common))}
    for m in metrics:
        out[f"diff_{m}"] = pb.get(m, np.nan) - pa.get(m, np.nan)
    if n_boot <= 0 or len(common) < 2:
        return out
    rng = np.random.default_rng(seed)
    pos_a = {c: i for i, c in enumerate(a.items)}
    pos_b = {c: i for i, c in enumerate(b.items)}
    ca = np.array([pos_a[c] for c in common])
    cb = np.array([pos_b[c] for c in common])
    diffs = {m: [] for m in metrics}
    for _ in range(n_boot):
        cnt = rng.multinomial(len(common), np.full(len(common), 1 / len(common))).astype(float)
        wa = np.zeros(a.n_items); wa[ca] = cnt
        wb = np.zeros(b.n_items); wb[cb] = cnt
        ra, rb = a.compute(wa, bins), b.compute(wb, bins)
        for m in metrics:
            diffs[m].append(rb.get(m, np.nan) - ra.get(m, np.nan))
    for m in metrics:
        arr = np.array(diffs[m], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr):
            out[f"diff_{m}_lo"] = float(np.percentile(arr, 2.5))
            out[f"diff_{m}_hi"] = float(np.percentile(arr, 97.5))
            p = 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))
            out[f"diff_{m}_p_boot"] = float(min(1.0, p))
    return out
