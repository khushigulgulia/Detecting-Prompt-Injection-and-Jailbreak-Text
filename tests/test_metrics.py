import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics, load_predictions, save_predictions, tune_threshold


def test_basic_metrics_match_hand_calc():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([1, 0, 0, 1, 1, 0])
    m = compute_metrics(y, p)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["false_negative_rate"] == pytest.approx(1 / 3)


def test_auc_and_ap_with_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    m = compute_metrics(y, (s >= 0.5).astype(int), s)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["average_precision"] == pytest.approx(0.8333, abs=1e-3)
    assert m["pr_auc_trapezoid"] is not None and m["average_precision"] != m["pr_auc_trapezoid"]


def test_single_class_does_not_crash():
    m = compute_metrics(np.array([1, 1, 1]), np.array([1, 0, 1]), np.array([0.9, 0.2, 0.8]))
    assert m["roc_auc"] is None and any("one class" in n for n in m["notes"])
    assert m["recall"] == pytest.approx(2 / 3)


def test_binary_scores_are_flagged():
    m = compute_metrics(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), np.array([0.0, 1.0, 1.0, 1.0]))
    assert any("binary" in n for n in m["notes"])


def test_tune_threshold_uses_only_given_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    t = tune_threshold(y, s)
    assert 0.3 < t <= 0.6


def test_prediction_file_schema_roundtrip(tmp_path):
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
    p = tmp_path / "m_test_predictions.csv"
    save_predictions(df, np.array([0.2, 0.9]), np.array([0, 1]), p)
    out = load_predictions(p)
    assert list(out.columns) == ["text", "label", "score", "pred"]


def test_load_predictions_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run the corresponding"):
        load_predictions(tmp_path / "nope.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_predictions(bad)
