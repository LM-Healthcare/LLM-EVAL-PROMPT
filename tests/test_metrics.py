import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from statsmodels.stats.inter_rater import fleiss_kappa

from prompteval.metrics import CellData, _auc, ece, fleiss_kappa_weighted, quadratic_kappa


def test_fast_implementations_match_references():
    rng = np.random.default_rng(0)
    for _ in range(10):
        y = rng.integers(0, 2, 400).astype(float)
        s = rng.integers(0, 11, 400) / 10
        w = rng.integers(0, 4, 400).astype(float)
        assert abs(_auc(y, s, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-12
        a, b = rng.integers(0, 3, 200), rng.integers(0, 3, 200)
        ww = rng.integers(1, 3, 200).astype(float)
        ref = cohen_kappa_score(a, b, weights="quadratic", sample_weight=ww)
        assert abs(quadratic_kappa(a, b, ww, 3) - ref) < 1e-12
        c = rng.multinomial(10, [.5, .2, .1, .1, .05, .05], size=150).astype(float)
        assert abs(fleiss_kappa_weighted(c, np.ones(150)) - fleiss_kappa(c)) < 1e-12


def test_ece_perfect_calibration_is_zero():
    y = np.array([1, 0] * 50, float)
    s = np.full(100, 0.55)
    assert abs(ece(y, s, np.ones(100), 10) - 0.05) < 1e-12


def _cell(rows):
    df = pd.DataFrame(rows, columns=["question_code", "answer", "correct", "confidence", "pred_orig"])
    return CellData(df, theta=80)


def test_cell_metrics_hand_computed():
    rows = [
        # q1: 3 runs all correct (A), confident
        ("q1", "A", True, 90, 0), ("q1", "A", True, 95, 0), ("q1", "A", True, 85, 0),
        # q2: 2 wrong (B) confident + 1 correct
        ("q2", "B", False, 90, 1), ("q2", "B", False, 60, 1), ("q2", "C", True, 70, 0),
        # q3: one parse failure, two wrong
        ("q3", None, False, np.nan, np.nan), ("q3", "D", False, 85, 3), ("q3", "D", False, 40, 3),
    ]
    m = _cell(rows).compute()
    assert abs(m["accuracy"] - 4 / 9) < 1e-12
    assert abs(m["parse_failure"] - 1 / 9) < 1e-12
    assert abs(m["unanimity"] - 1 / 3) < 1e-12  # only q1 (q3 has an invalid run)
    assert abs(m["majority_accuracy"] - 1 / 3) < 1e-12
    assert abs(m["oci"] - 2 / 4) < 1e-12  # wrong with conf: 90, 60, 85, 40
    assert abs(m["modal_agreement"] - (1 + 2 / 3 + 2 / 3) / 3) < 1e-12


def test_harm_from_distractor_ratings():
    rows = [("q1", "A", True, 90, 0), ("q1", "B", False, 95, 1), ("q2", "C", False, 50, 2),
            ("q2", "D", False, 99, 3)]
    df = pd.DataFrame(rows, columns=["question_code", "answer", "correct", "confidence", "pred_orig"])
    harm = {("q1", 1): 2.0, ("q2", 2): 1.0}  # (q2, 3) unrated
    m = CellData(df, 80, harm=harm).compute()
    assert abs(m["severe_error_rate"] - 1 / 4) < 1e-12
    assert abs(m["confident_severe_error_rate"] - 1 / 4) < 1e-12
    assert abs(m["harm_unrated_share"] - 1 / 4) < 1e-12
    assert abs(m["mean_harm"] - (0 + 2 + 1) / 3) < 1e-12
