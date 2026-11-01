"""Fine-tune DistilBERT (Solution 2).

Usage:
  python scripts/train_distilbert.py --mode quick   # tiny smoke test, writes nothing to results/
  python scripts/train_distilbert.py --mode full    # real run, saves checkpoint + predictions
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.metrics import pred_path, save_predictions
from src.models.distilbert_classifier import pick_device, predict_scores, save_checkpoint, train_distilbert
from src.utils.config import environment_info, load_config, resolve, save_json, set_seed
from src.utils.logging_utils import get_logger

log = get_logger("train_distilbert")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    proc, art, res = resolve(cfg, "processed_dir"), resolve(cfg, "artifacts_dir"), resolve(cfg, "results_dir")
    train, val, test = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))

    model, tok, hist = train_distilbert(cfg, train, val, args.mode, seed, args.epochs)
    if args.mode == "quick":
        scores = predict_scores(model, tok, val["text"].head(8).tolist(), cfg["distilbert"]["max_seq_len"], pick_device())
        log.info("Smoke test OK: history=%s, sample scores=%s", hist, np.round(scores, 3).tolist())
        log.info("Quick mode saves nothing. Run --mode full for the real experiment.")
        return

    device = pick_device()
    max_len = cfg["distilbert"]["max_seq_len"]
    size_mb = save_checkpoint(model, tok, art / "distilbert")
    for split, df in (("val", val), ("test", test)):
        t = time.perf_counter()
        scores = predict_scores(model, tok, df["text"].tolist(), max_len, device)
        elapsed = time.perf_counter() - t
        save_predictions(df, scores, (scores >= 0.5).astype(int), pred_path(res, "distilbert", split))
        if split == "test":
            batch_ms = 1000 * elapsed / len(df)
    one = test["text"].iloc[:1].tolist()
    predict_scores(model, tok, one, max_len, device)  # warm-up
    times = []
    for _ in range(30):
        s = time.perf_counter()
        predict_scores(model, tok, one, max_len, device)
        times.append(time.perf_counter() - s)
    meta = {
        "name": "distilbert", "score_type": "probability", "default_threshold": 0.5,
        "history": hist, "n_train": len(train), "device": str(device),
        "batch_latency_ms_per_sample": batch_ms, "single_sample_latency_ms": 1000 * float(np.median(times)),
        "throughput_samples_per_s": 1000.0 / batch_ms, "model_size_mb": size_mb,
        "train_time_s": hist["train_time_s"], "environment": environment_info(),
    }
    save_json(meta, res / "distilbert_meta.json")
    log.info("Saved. best val F1=%.4f, size=%.1f MB", hist["best_val_f1"], size_mb)


if __name__ == "__main__":
    main()
