"""Error analysis from saved test predictions.

Categories are *rule-assisted heuristics* to help a human reviewer group examples;
they are not ground truth. Review ``reports/tables/errors_*.csv`` before drawing
conclusions. Dataset text is treated as inert data (never executed).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import load_predictions, pred_path

MODELS = ["classical", "distilbert", "dspy_baseline", "dspy_optimized"]

_OVERRIDE = re.compile(r"\b(ignore|forget|disregard|override|stop)\b.{0,40}\b(previous|prior|above|instruction|task|everything|all)\b", re.I)
_ROLEPLAY = re.compile(r"\b(you are now|pretend|act as|role ?play|in character|stay in character|as an? [a-z]+ (?:actor|character))\b", re.I)
_REVEAL = re.compile(r"\b(system prompt|prompt text|your instructions|reveal|show me (?:all|your))\b", re.I)
_GERMAN = re.compile(r"\b(und|der|die|das|nicht|ich|ist|zu|mit|für|wie|was|Sie|bitte|auf)\b")
_ENCODED = re.compile(r"(?:[A-Za-z0-9+/]{24,}={0,2})|(?:\\x[0-9a-f]{2})|(?:&#\d+;)")


def categorize(text: str) -> list[str]:
    """Heuristic tags for a text. A text may carry several tags."""
    tags = []
    if _OVERRIDE.search(text):
        tags.append("instruction_override_phrasing")
    if _ROLEPLAY.search(text):
        tags.append("roleplay_or_persona")
    if _REVEAL.search(text):
        tags.append("prompt_reveal_request")
    if len(_GERMAN.findall(text)) >= 3:
        tags.append("non_english_(german)")
    if _ENCODED.search(text):
        tags.append("encoded_or_obfuscated")
    n = len(text)
    tags.append("long_input(>400ch)" if n > 400 else "short_input(<=400ch)")
    if "\n" in text:
        tags.append("multiline")
    return tags


def run(results: Path, tables: Path) -> dict[str, pd.DataFrame]:
    """Write error tables and return them. Models without predictions are skipped."""
    tables.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        p = pred_path(results, m, "test")
        if p.exists():
            frames[m] = load_predictions(p)
    if not frames:
        raise FileNotFoundError("No test predictions found; train a model first.")
    base = next(iter(frames.values()))[["text", "label"]].copy()
    for m, df in frames.items():
        if not (df["text"].values == base["text"].values).all():
            # e.g. DSPy evaluated on a capped subset; align on text instead of position.
            base = base[base["text"].isin(df["text"])]
    for m, df in frames.items():
        base = base.merge(df[["text", "pred"]].rename(columns={"pred": f"pred_{m}"}), on="text", how="left")
    base["length"] = base["text"].str.len()
    base["tags"] = base["text"].map(lambda t: "|".join(categorize(t)))
    pred_cols = [f"pred_{m}" for m in frames]
    sub = base.dropna(subset=pred_cols)
    sub[pred_cols] = sub[pred_cols].astype(int)

    summary = []
    for m in frames:
        col = f"pred_{m}"
        fp, fn = sub[(sub[col] == 1) & (sub["label"] == 0)], sub[(sub[col] == 0) & (sub["label"] == 1)]
        fp.to_csv(tables / f"errors_{m}_false_positives.csv", index=False)
        fn.to_csv(tables / f"errors_{m}_false_negatives.csv", index=False)
        summary += [{"model": m, "error_type": "false_positive", "count": len(fp)},
                    {"model": m, "error_type": "false_negative", "count": len(fn)}]
    sub[sub.apply(lambda r: all(r[c] == r["label"] for c in pred_cols), axis=1)].to_csv(tables / "all_models_correct.csv", index=False)
    disagree = sub[sub[pred_cols].nunique(axis=1) > 1]
    disagree.to_csv(tables / "models_disagree.csv", index=False)
    pd.DataFrame(summary).to_csv(tables / "error_summary.csv", index=False)

    # Tag frequency among errors vs all test examples (descriptive only).
    tag_rows = []
    all_tags = sub["tags"].str.split("|").explode().value_counts()
    for m in frames:
        err = sub[sub[f"pred_{m}"] != sub["label"]]
        for tag, c in err["tags"].str.split("|").explode().value_counts().items():
            tag_rows.append({"model": m, "tag": tag, "errors_with_tag": int(c), "test_examples_with_tag": int(all_tags.get(tag, 0))})
    pd.DataFrame(tag_rows).to_csv(tables / "error_tags.csv", index=False)
    return {"all": sub, "disagree": disagree}
