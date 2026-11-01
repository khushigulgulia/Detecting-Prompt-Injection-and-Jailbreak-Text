"""Stratified train/val/test splitting (after deduplication)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def stratified_split(
    df: pd.DataFrame, train: float, val: float, test: float, seed: int, label_col: str = "label"
) -> dict[str, pd.DataFrame]:
    """Split into train/val/test, stratified by label, reproducibly.

    The input must already be deduplicated; this function asserts no text appears
    in more than one split.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Split proportions must sum to 1.0")
    if df["text"].duplicated().any():
        raise ValueError("Deduplicate before splitting to avoid leakage")

    train_df, rest = train_test_split(
        df, train_size=train, stratify=df[label_col], random_state=seed
    )
    rel_test = test / (val + test)
    val_df, test_df = train_test_split(
        rest, test_size=rel_test, stratify=rest[label_col], random_state=seed
    )
    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Raise if any text occurs in more than one split."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]["text"]) & set(splits[b]["text"])
            if overlap:
                raise AssertionError(f"{len(overlap)} texts leak between {a} and {b}")


def near_duplicate_groups(texts: list[str], threshold: float) -> np.ndarray:
    """Cluster texts whose char-n-gram TF-IDF cosine similarity >= threshold (connected components).

    Returns an integer group id per text. Catches near-duplicates and many
    translation/paraphrase variants that exact-match deduplication misses.
    """
    from scipy.sparse.csgraph import connected_components
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    adj = cosine_similarity(X) >= threshold
    _, groups = connected_components(adj, directed=False)
    return groups


def group_stratified_split(
    df: pd.DataFrame, train: float, val: float, test: float, seed: int, threshold: float, n_folds: int = 20
) -> dict[str, pd.DataFrame]:
    """Stratified, group-aware split: near-duplicate clusters never straddle splits.

    Uses StratifiedGroupKFold with ``n_folds`` folds and assigns folds to splits in
    the requested proportions, so realised sizes are approximate.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Split proportions must sum to 1.0")
    if df["text"].duplicated().any():
        raise ValueError("Deduplicate before splitting to avoid leakage")
    groups = near_duplicate_groups(df["text"].tolist(), threshold)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups)):
        fold_of[idx] = k
    n_val, n_test = round(val * n_folds), round(test * n_folds)
    test_folds, val_folds = set(range(n_test)), set(range(n_test, n_test + n_val))
    name = np.where(np.isin(fold_of, list(test_folds)), "test", np.where(np.isin(fold_of, list(val_folds)), "val", "train"))
    splits = {s: df[name == s].reset_index(drop=True) for s in ("train", "val", "test")}
    assert_no_leakage(splits)
    return splits
