"""Dataset loading with on-disk caching and schema inspection."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def load_hf_dataset(name: str, cache_dir: Path) -> pd.DataFrame:
    """Load all splits of a HF dataset into one DataFrame (cached as parquet).

    Original split membership is kept in the ``orig_split`` column; we re-split
    ourselves later, so it is not used for training decisions.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (name.replace("/", "__") + ".parquet")
    if cache.exists():
        log.info("Loading %s from cache %s", name, cache)
        return pd.read_parquet(cache)

    from datasets import load_dataset  # imported lazily: heavy

    log.info("Downloading %s from Hugging Face", name)
    ds = load_dataset(name)
    frames = []
    for split, d in ds.items():
        f = d.to_pandas()
        f["orig_split"] = split
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache)
    return df


def inspect_dataset(df: pd.DataFrame, text_col: str, label_col: str, n_samples: int = 3) -> dict:
    """Return schema, size, class distribution, missing values, duplicates and samples."""
    info = {
        "rows": len(df),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "class_distribution": df[label_col].value_counts(dropna=False).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "duplicate_texts": int(df[text_col].duplicated().sum()),
        "text_len_chars": df[text_col].astype(str).str.len().describe().round(1).to_dict(),
        "samples": {
            str(lab): g[text_col].astype(str).str.slice(0, 160).head(n_samples).tolist()
            for lab, g in df.groupby(label_col)
        },
    }
    return info
