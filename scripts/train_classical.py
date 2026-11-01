"""Train Solution 1 (TF-IDF + SVM/LogReg), select on validation, evaluate on test.

Usage: python scripts/train_classical.py [--seed 42] [--train-fraction 1.0]
"""
from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import pred_path, save_predictions
from src.models.classical_ml import decision_scores, default_threshold, select_best
from src.utils.config import environment_info, load_config, resolve, save_json, set_seed
from src.utils.logging_utils import get_logger
from src.utils.timing import timed

log = get_logger("train_classical")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    proc, art, res = resolve(cfg, "processed_dir"), resolve(cfg, "artifacts_dir"), resolve(cfg, "results_dir")
    train, val, test = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))

    with timed() as t_train:
        pipe, spec, grid = select_best(cfg, train, val, seed)
    log.info("Selected %s (C=%s) val_f1=%.4f", spec["model"], spec["C"], spec["val_f1"])
    # Selection already refit on train only; test remains untouched until here.

    model_path = art / "classical_model.joblib"
    joblib.dump(pipe, model_path)

    thr = default_threshold(spec["model"])
    for split, df in (("val", val), ("test", test)):
        with timed() as t_inf:
            scores = decision_scores(pipe, df["text"].tolist())
        save_predictions(df, scores, (scores >= thr).astype(int), pred_path(res, "classical", split))
        if split == "test":
            latency_ms = 1000 * t_inf.elapsed / len(df)
    # Single-sample latency (realistic online use), median of repeated calls.
    one = test["text"].iloc[:1].tolist()
    times = []
    for _ in range(50):
        s = time.perf_counter()
        decision_scores(pipe, one)
        times.append(time.perf_counter() - s)
    single_ms = 1000 * float(np.median(times))

    meta = {
        "name": "classical",
        "selected": spec,
        "score_type": "probability" if spec["model"] == "logreg" else "svm_margin",
        "default_threshold": thr,
        "grid": grid,
        "train_time_s_including_grid_search": t_train.elapsed,
        "batch_latency_ms_per_sample": latency_ms,
        "single_sample_latency_ms": single_ms,
        "throughput_samples_per_s": 1000.0 / latency_ms,
        "model_size_mb": os.path.getsize(model_path) / 1e6,
        "n_train": len(train),
        "environment": environment_info(),
    }
    save_json(meta, res / "classical_meta.json")
    log.info("Saved artifacts and predictions. Size %.2f MB", meta["model_size_mb"])


if __name__ == "__main__":
    main()
