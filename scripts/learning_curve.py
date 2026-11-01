"""Learning curves: retrain Solution 1 and 2 on growing fractions of TRAIN.

These are real retraining runs (not approximations). Evaluation uses the fixed
test split; hyper-parameters for the classical model are the validation-selected
ones from train_classical.py. Output: results/learning_curve.csv.

Usage: python scripts/learning_curve.py [--seeds 42,43,44] [--skip-distilbert]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import f1_score

from src.models.classical_ml import build_pipeline, decision_scores, default_threshold
from src.models.distilbert_classifier import pick_device, predict_scores, train_distilbert
from src.utils.config import load_config, resolve, set_seed
from src.utils.logging_utils import get_logger

log = get_logger("learning_curve")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--skip-distilbert", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = [int(s) for s in args.seeds.split(",")]
    proc, res = resolve(cfg, "processed_dir"), resolve(cfg, "results_dir")
    train, val, test = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))
    spec = json.loads((res / "classical_meta.json").read_text())["selected"]
    out_path = res / "learning_curve.csv"
    # Resume: keep finished (model, fraction, seed) points so an interrupted run can continue.
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {(r["model"], r["fraction"], r["seed"]) for r in rows}
    for frac in cfg["evaluation"]["learning_curve_fractions"]:
        for seed in seeds:
            set_seed(seed)
            sub = train.sample(frac=frac, random_state=seed) if frac < 1 else train
            pipe = build_pipeline(cfg, spec["model"], spec["C"], seed).fit(sub["text"].tolist(), sub["label"].values)
            pred = (decision_scores(pipe, test["text"].tolist()) >= default_threshold(spec["model"])).astype(int)
            if ("classical", frac, seed) not in done:
                rows.append({"model": "classical", "fraction": frac, "n_train": len(sub), "seed": seed,
                             "test_f1": f1_score(test["label"], pred, zero_division=0)})
            if not args.skip_distilbert and ("distilbert", frac, seed) not in done:
                m, tok, _ = train_distilbert(cfg, sub, val, "full", seed)
                sc = predict_scores(m, tok, test["text"].tolist(), cfg["distilbert"]["max_seq_len"], pick_device())
                rows.append({"model": "distilbert", "fraction": frac, "n_train": len(sub), "seed": seed,
                             "test_f1": f1_score(test["label"], (sc >= 0.5).astype(int), zero_division=0)})
                del m
            log.info("frac=%s seed=%s done", frac, seed)
            pd.DataFrame(rows).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
