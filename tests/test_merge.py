import pandas as pd

from src.data.preprocess import clean_dataframe
from src.data.split_data import group_stratified_split


def test_merge_removes_cross_dataset_duplicates_and_conflicts_and_keeps_both_sources():
    a = pd.DataFrame({"text": [f"alpha sample {i} qwertyuiop" * 2 for i in range(40)], "label": [i % 2 for i in range(40)]})
    b = pd.DataFrame({"prompt": [f"beta jailbreak prompt {i} zxcvbnm" * 2 for i in range(40)] + [a.text[0], a.text[1]],
                      "type": ["benign" if i % 2 else "jailbreak" for i in range(40)] + ["benign", "jailbreak"]})
    ca, _ = clean_dataframe(a, "text", "label", "A")
    cb, _ = clean_dataframe(b, "prompt", "type", "B")
    pool, rep = clean_dataframe(pd.concat([ca, cb], ignore_index=True), "text", "label", "merged")
    # a.text[0] has label 0 in A and 'benign'=0 in B -> exact duplicate; a.text[1] label 1 vs 'jailbreak'=1 -> duplicate too
    assert rep["dropped_exact_duplicates"] == 2 and rep["dropped_conflicting_duplicates"] == 0
    assert pool["text"].is_unique
    splits = group_stratified_split(pool, 0.7, 0.15, 0.15, seed=0, threshold=0.8)
    assert sum(len(v) for v in splits.values()) == len(pool)
