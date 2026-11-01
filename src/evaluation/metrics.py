"""Classification metrics, threshold tuning and prediction-file helpers.

Every model writes ``<name>_<split>_predictions.csv`` with columns
text,label,score,pred. Metrics are always recomputed from those files, so
reported numbers are traceable to saved predictions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

LABELS = [0, 1]
LABEL_NAMES = ["benign", "injection"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, Any]:
    """Compute the required metrics; undefined ones are None with a note, never a crash."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out: dict[str, Any] = {"n": int(len(y_true)), "notes": []}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    out["confusion_matrix"] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    out["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) else None
    out["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else None

    out["roc_auc"] = out["average_precision"] = out["pr_auc_trapezoid"] = None
    if scores is not None and len(np.unique(y_true)) < 2:
        out["notes"].append("ROC-AUC/PR undefined: only one class present in y_true")
    elif scores is not None:
        scores = np.asarray(scores, dtype=float)
        if len(np.unique(scores)) < 3:
            out["notes"].append("Scores are (near-)binary; ROC/PR curves have few points and AUC is coarse")
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["average_precision"] = float(average_precision_score(y_true, scores))
        p, r, _ = precision_recall_curve(y_true, scores)
        out["pr_auc_trapezoid"] = float(auc(r, p))
    return out


def tune_threshold(y_val: np.ndarray, scores: np.ndarray, metric: str = "f1", beta: float = 1.0) -> float:
    """Pick the score threshold maximising F-beta on *validation* data only."""
    y_val, scores = np.asarray(y_val), np.asarray(scores, dtype=float)
    cands = np.unique(scores)
    if len(cands) < 2:
        return float(cands[0]) if len(cands) else 0.5
    # candidate thresholds: midpoints between consecutive distinct scores
    mids = (cands[:-1] + cands[1:]) / 2
    best_t, best_s = float(mids[0]), -1.0
    for t in mids:
        pred = (scores >= t).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        s = 0.0 if (p + r) == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


def pred_path(results_dir: Path, name: str, split: str) -> Path:
    return results_dir / f"{name}_{split}_predictions.csv"


def save_predictions(df: pd.DataFrame, scores: np.ndarray, preds: np.ndarray, path: Path) -> None:
    """Persist predictions with the standard schema."""
    out = pd.DataFrame(
        {"text": df["text"].values, "label": df["label"].values, "score": np.asarray(scores, dtype=float), "pred": np.asarray(preds, dtype=int)}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_predictions(path: Path) -> pd.DataFrame:
    """Load a predictions file, validating the schema."""
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}. Run the corresponding training script first.")
    df = pd.read_csv(path)
    missing = {"text", "label", "score", "pred"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df
