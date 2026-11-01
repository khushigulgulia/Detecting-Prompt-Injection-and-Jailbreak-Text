"""Solution 1 variants with dimensionality reduction: TF-IDF -> [chi2 selection] -> PCA / TruncatedSVD -> LinearSVC.

True PCA needs a dense, centred matrix, so it is applied after chi-squared pre-selection (e.g. 5,000 columns).
TruncatedSVD (latent semantic analysis) is the sparse-friendly equivalent and runs on the full TF-IDF space.
All steps are fit on the training split only (they live inside one sklearn Pipeline).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from src.models.classical_ml import build_pipeline


def to_dense(X) -> np.ndarray:
    """Module-level (picklable) sparse -> dense conversion."""
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def build_reduced_pipeline(cfg: dict[str, Any], method: str, n_components: float | int, C: float, seed: int,
                           k_pre: int | None = 5000) -> Pipeline:
    """method: 'pca' (chi2 pre-selection -> dense -> PCA) or 'svd' (TruncatedSVD on the full sparse TF-IDF)."""
    feats = build_pipeline(cfg, "svm", C, seed).named_steps["tfidf"]
    clf = LinearSVC(C=C, class_weight="balanced", random_state=seed, max_iter=20000)
    if method == "pca":
        return Pipeline([
            ("tfidf", feats),
            ("select", SelectKBest(chi2, k=k_pre)),
            ("dense", FunctionTransformer(to_dense, accept_sparse=True)),
            ("pca", PCA(n_components=n_components, svd_solver="full", random_state=seed)),
            ("clf", clf),
        ])
    if method == "svd":
        return Pipeline([
            ("tfidf", feats),
            ("svd", TruncatedSVD(n_components=int(n_components), random_state=seed)),
            ("clf", clf),
        ])
    raise ValueError(f"unknown method {method}")
