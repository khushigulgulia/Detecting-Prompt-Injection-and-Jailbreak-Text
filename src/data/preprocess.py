"""Text normalisation, label mapping, cleaning and deduplication.

Normalisation is deliberately conservative: attack signals (casing, odd
punctuation, unicode tricks, embedded instructions) must survive, so only
whitespace/control-character cleanup and Unicode NFC normalisation are applied.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

BENIGN, INJECTION = 0, 1
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t]+")
_STRING_LABELS = {
    "benign": BENIGN,
    "safe": BENIGN,
    "jailbreak": INJECTION,
    "injection": INJECTION,
    "malicious": INJECTION,
}


def normalize_text(text: str) -> str:
    """NFC-normalise, strip control chars, collapse runs of spaces/tabs, trim.

    Newlines, casing and punctuation are preserved on purpose.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def map_label(value: object) -> int:
    """Map a raw label (int/bool/str) to 0=benign, 1=injection."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _STRING_LABELS:
            raise ValueError(f"Unknown string label: {value!r}")
        return _STRING_LABELS[key]
    if isinstance(value, (bool, int)) and int(value) in (0, 1):
        return int(value)
    raise ValueError(f"Cannot map label: {value!r}")


def clean_dataframe(
    df: pd.DataFrame, text_col: str, label_col: str, source: str, min_chars: int = 1
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a cleaned frame with columns text/label/source plus a report of drops.

    Drops rows with missing/empty text or unmappable labels and exact duplicates
    (after normalisation). Conflicting duplicates (same text, different label) are
    removed entirely since the label is unreliable.
    """
    report = {"input_rows": len(df)}
    work = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).copy()

    missing = work["text"].isna() | work["label"].isna()
    report["dropped_missing"] = int(missing.sum())
    work = work[~missing]

    work["text"] = work["text"].map(normalize_text)
    short = work["text"].str.len() < min_chars
    report["dropped_empty"] = int(short.sum())
    work = work[~short]

    def _safe_map(v: object) -> float:
        try:
            return float(map_label(v))
        except ValueError:
            return float("nan")

    work["label"] = work["label"].map(_safe_map)
    bad = work["label"].isna()
    report["dropped_bad_label"] = int(bad.sum())
    work = work[~bad]
    work["label"] = work["label"].astype(int)

    conflict_texts = work.groupby("text")["label"].nunique()
    conflict_texts = set(conflict_texts[conflict_texts > 1].index)
    is_conflict = work["text"].isin(conflict_texts)
    report["dropped_conflicting_duplicates"] = int(is_conflict.sum())
    work = work[~is_conflict]

    before = len(work)
    work = work.drop_duplicates(subset="text", keep="first")
    report["dropped_exact_duplicates"] = before - len(work)

    work["source"] = source
    work = work.reset_index(drop=True)
    report["output_rows"] = len(work)
    return work, report
