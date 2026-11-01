"""Solution 1: TF-IDF (word + char n-grams) with LinearSVC / Logistic Regression."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(cfg: dict[str, Any], model: str, C: float, seed: int) -> Pipeline:
    """Build vectoriser + classifier pipeline. TF-IDF is fit only inside ``fit``."""
    t = cfg["classical"]["tfidf"]
    # Casing handling is configurable (classical.tfidf.lowercase); no stop-word or token removal is applied.
    words = TfidfVectorizer(
        analyzer="word", ngram_range=tuple(t["word_ngram_range"]), max_features=t["max_features"],
        min_df=t["min_df"], lowercase=t["lowercase"], sublinear_tf=True,
    )
    chars = TfidfVectorizer(
        analyzer="char_wb", ngram_range=tuple(t["char_ngram_range"]), max_features=t["max_features"],
        min_df=t["min_df"], lowercase=t["lowercase"], sublinear_tf=True,
    )
    feats = FeatureUnion([("word", words), ("char", chars)])
    if model == "svm":
        clf = LinearSVC(C=C, class_weight="balanced", random_state=seed)
    elif model == "logreg":
        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=cfg["classical"]["logreg"]["max_iter"], random_state=seed
        )
    else:
        raise ValueError(f"Unknown classical model: {model}")
    return Pipeline([("tfidf", feats), ("clf", clf)])


def decision_scores(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    """Return a ranking score. LinearSVC: signed margin (NOT a probability); LogReg: P(injection)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(texts)[:, 1]
    return pipe.decision_function(texts)


def default_threshold(model: str) -> float:
    """Native decision boundary: 0 for SVM margin, 0.5 for LogReg probability."""
    return 0.0 if model == "svm" else 0.5


def select_best(
    cfg: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, seed: int
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Grid-search C for each model on validation F1; return best pipeline, its spec and the full grid log."""
    grid_log: list[dict[str, Any]] = []
    best: tuple[float, Pipeline, dict[str, Any]] | None = None
    for model, key in (("svm", "svm"), ("logreg", "logreg")):
        for C in cfg["classical"][key]["C_grid"]:
            pipe = build_pipeline(cfg, model, C, seed)
            pipe.fit(train["text"].tolist(), train["label"].values)
            pred = (decision_scores(pipe, val["text"].tolist()) >= default_threshold(model)).astype(int)
            f1 = float(f1_score(val["label"].values, pred, zero_division=0))
            grid_log.append({"model": model, "C": C, "val_f1": f1})
            if best is None or f1 > best[0]:
                best = (f1, pipe, {"model": model, "C": C, "val_f1": f1})
    assert best is not None
    return best[1], best[2], grid_log
