"""Aggregate saved predictions into metric tables (CSV/JSON).

Metrics are recomputed from results/*_predictions.csv. Thresholds are tuned on the
validation predictions only and then applied unchanged to test. Models whose
predictions do not exist yet are reported as pending, never filled in.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation.metrics import compute_metrics, load_predictions, pred_path, tune_threshold
from src.utils.config import load_config, resolve, save_json
from src.utils.logging_utils import get_logger

log = get_logger("evaluate_models")

# name -> display label
MODELS = {
    "classical": "TF-IDF + SVM",
    "distilbert": "DistilBERT",
    "dspy_baseline": "DSPy (baseline)",
    "dspy_optimized": "DSPy (optimized)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res, tables = resolve(cfg, "results_dir"), resolve(cfg, "tables_dir")

    proc = resolve(cfg, "processed_dir")
    src_map = pd.read_parquet(proc / "test.parquet").set_index("text")["source"]
    rows, by_source, pending = [], [], []
    for name, label in MODELS.items():
        test_p, val_p = pred_path(res, name, "test"), pred_path(res, name, "val")
        if not test_p.exists():
            pending.append(name)
            log.warning("%s: no predictions yet -> pending", name)
            continue
        test, val = load_predictions(test_p), load_predictions(val_p)
        meta_p = res / f"{name.split('_')[0]}_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        m = compute_metrics(test["label"].values, test["pred"].values, test["score"].values)
        row = {"model": name, "display": label, "variant": "default_threshold", **{k: v for k, v in m.items() if k != "confusion_matrix"}}
        row["notes"] = "; ".join(m["notes"])
        rows.append(row)
        # Per-source breakdown: overall scores can hide a model that only works on one data source.
        t_src = test.assign(source=test["text"].map(src_map))
        for src, g in t_src.groupby("source"):
            ms = compute_metrics(g["label"].values, g["pred"].values)
            by_source.append({"model": name, "display": label, "source": src,
                              **{k: v for k, v in ms.items() if k not in ("confusion_matrix", "notes")}})

        # Threshold tuned on validation (F1, and a recall-weighted F2 for security use).
        if cfg["evaluation"]["tune_threshold_on_validation"] and val["score"].nunique() > 2:
            for tag, beta in (("val_tuned_f1", 1.0), ("val_tuned_f2", 2.0)):
                thr = tune_threshold(val["label"].values, val["score"].values, beta=beta)
                pred = (test["score"].values >= thr).astype(int)
                mt = compute_metrics(test["label"].values, pred, test["score"].values)
                rows.append({"model": name, "display": label, "variant": tag, "threshold": thr,
                             **{k: v for k, v in mt.items() if k != "confusion_matrix"}, "notes": "; ".join(mt["notes"])})
    df = pd.DataFrame(rows)
    df.to_csv(tables / "metrics_test.csv", index=False)
    if by_source:
        pd.DataFrame(by_source).to_csv(tables / "metrics_by_source.csv", index=False)
    save_json({"evaluated": sorted(df["model"].unique().tolist()) if len(df) else [], "pending": pending},
              tables / "evaluation_status.json")
    cols = ["display", "variant", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "fp", "fn"]
    if len(df):
        print(df[cols].round(4).to_string(index=False))
        if by_source:
            print(pd.DataFrame(by_source)[["display", "source", "n", "accuracy", "precision", "recall", "f1", "fp", "fn"]].round(3).to_string(index=False))
    log.info("Pending (no predictions): %s", pending or "none")


if __name__ == "__main__":
    main()
