import random
import string

import pandas as pd
import pytest

from src.data.split_data import assert_no_leakage, group_stratified_split, near_duplicate_groups, stratified_split


def _df(n=200):
    return pd.DataFrame({"text": [f"unique sentence number {i} about topic {i * 7}" for i in range(n)],
                         "label": [i % 4 == 0 for i in range(n)]}).astype({"label": int})


def test_stratified_split_sizes_and_stratification():
    s = stratified_split(_df(), 0.7, 0.15, 0.15, seed=1)
    assert sum(len(v) for v in s.values()) == 200
    for v in s.values():
        assert abs(v["label"].mean() - 0.25) < 0.06


def test_split_is_reproducible():
    a = stratified_split(_df(), 0.7, 0.15, 0.15, seed=3)
    b = stratified_split(_df(), 0.7, 0.15, 0.15, seed=3)
    assert a["test"]["text"].tolist() == b["test"]["text"].tolist()


def test_rejects_bad_proportions_and_duplicates():
    with pytest.raises(ValueError):
        stratified_split(_df(), 0.5, 0.2, 0.2, seed=1)
    d = pd.concat([_df(10), _df(10)])
    with pytest.raises(ValueError):
        stratified_split(d, 0.7, 0.15, 0.15, seed=1)


def test_leakage_detector():
    a = pd.DataFrame({"text": ["x", "y"], "label": [0, 1]})
    b = pd.DataFrame({"text": ["y", "z"], "label": [1, 0]})
    with pytest.raises(AssertionError):
        assert_no_leakage({"train": a, "test": b})


def test_near_duplicates_are_grouped_and_never_split():
    rng = random.Random(0)
    base = ["".join(rng.choices(string.ascii_lowercase + " ", k=60)) for _ in range(60)]
    near = [t + " ." for t in base[:30]]
    labels = [i % 2 for i in range(60)]
    df = pd.DataFrame({"text": base + near, "label": labels + labels[:30]})
    groups = near_duplicate_groups(df["text"].tolist(), 0.9)
    assert groups[0] == groups[60]  # original and its near-copy share a group
    s = group_stratified_split(df, 0.7, 0.15, 0.15, seed=0, threshold=0.9)
    for orig, copy in zip(base[:30], near):
        where = {k: (orig in set(v["text"]), copy in set(v["text"])) for k, v in s.items()}
        assert sum(1 for k in where if any(where[k])) == 1, "near-duplicate pair straddles splits"
