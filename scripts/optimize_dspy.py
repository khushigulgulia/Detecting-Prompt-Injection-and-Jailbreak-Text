"""Evaluate and optimise the DSPy classifier (Solution 3).

Requires an LLM: local Ollama (no key) or OpenAI/Anthropic via env vars.
Without one it writes results/dspy_status.json = pending and exits with code 2.

Protocol (no test leakage):
  * demonstrations / optimizer training examples come from the TRAIN split only
  * optimizer validation (MIPROv2) and budget selection use the VAL split only
  * the TEST split is used once, at the end, for baseline and final program

Usage: python scripts/optimize_dspy.py [--provider ollama] [--optimizer bootstrap|mipro]
                                       [--n-train 32] [--eval-max 100] [--budgets 0,4,8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dspy
import pandas as pd

from src.evaluation.metrics import pred_path, save_predictions
from src.models.dspy_classifier import (
    InjectionClassifier, evaluate_program, lm_status, make_lm, make_metric, stratified_sample, to_examples,
)
from src.utils.config import environment_info, load_config, resolve, save_json, set_seed
from src.utils.logging_utils import get_logger
from src.utils.timing import timed

log = get_logger("optimize_dspy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--optimizer", choices=["bootstrap", "mipro"], default=None)
    ap.add_argument("--n-train", type=int, default=32, help="labelled train examples given to the optimizer")
    ap.add_argument("--eval-max", type=int, default=None, help="cap on val AND test examples evaluated (default: all)")
    ap.add_argument("--val-max", type=int, default=None, help="cap on validation examples (overrides --eval-max)")
    ap.add_argument("--test-max", type=int, default=None, help="cap on test examples (overrides --eval-max)")
    ap.add_argument("--budgets", default=None, help="comma-separated few-shot budgets, e.g. 0,4,8")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["dspy"]
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    provider = args.provider or d["provider"]
    optimizer = args.optimizer or d["optimizer"]
    proc, art, res = resolve(cfg, "processed_dir"), resolve(cfg, "artifacts_dir"), resolve(cfg, "results_dir")

    ok, why = lm_status(cfg, provider)
    if not ok:
        save_json({"status": "pending", "provider": provider, "reason": why}, res / "dspy_status.json")
        log.error("DSPy evaluation PENDING: %s", why)
        return 2

    train, val, test = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))
    cap = args.eval_max or d["eval_max_samples"]
    vcap, tcap = args.val_max or cap, args.test_max or cap
    val_e = stratified_sample(val, vcap, seed) if vcap else val
    test_e = stratified_sample(test, tcap, seed) if tcap else test
    mc = d.get("max_text_chars")
    lm = make_lm(cfg, provider)
    dspy.configure(lm=lm)
    metric = make_metric(d["injection_recall_weight"])
    opt_train = stratified_sample(train, args.n_train, seed)
    budgets = [int(b) for b in (args.budgets.split(",") if args.budgets else d["few_shot_budgets"])]
    summary: dict = {"provider": provider, "model": d["models"][provider], "optimizer": optimizer,
                     "n_opt_train": len(opt_train), "max_text_chars": mc, "n_val_eval": len(val_e), "n_test_eval": len(test_e), "seed": seed}

    # 1) few-shot budget sweep on VAL (labelled demos from TRAIN, no LLM compile cost)
    sweep = []
    for k in budgets:
        prog = InjectionClassifier(mc)
        if k > 0:
            demos = to_examples(stratified_sample(train, k, seed), mc)
            prog = dspy.LabeledFewShot(k=k).compile(prog, trainset=demos)
        preds, stats = evaluate_program(prog, lm, val_e)
        acc = float((preds["pred"] == preds["label"]).mean())
        sweep.append({"k_demos": k, "val_accuracy": acc, **stats})
        log.info("budget k=%d val_acc=%.3f", k, acc)
    summary["budget_sweep_val"] = sweep

    # 2) baseline (zero-shot, un-optimised) on VAL + TEST
    base = InjectionClassifier(mc)
    for split, df in (("val", val_e), ("test", test_e)):
        p, st = evaluate_program(base, lm, df)
        save_predictions(df, p["score"].values, p["pred"].values, pred_path(res, "dspy_baseline", split))
        summary[f"baseline_{split}_stats"] = st

    # 3) optimisation (TRAIN examples; VAL for MIPROv2 selection)
    trainset, valset = to_examples(opt_train, mc), to_examples(val_e, mc)
    with timed() as t_opt:
        if optimizer == "mipro":
            opt = dspy.MIPROv2(metric=metric, auto=d["mipro_auto"], num_threads=d["num_threads"], seed=seed)
            tuned = opt.compile(InjectionClassifier(mc), trainset=trainset, valset=valset)
        else:
            opt = dspy.BootstrapFewShot(
                metric=metric, max_bootstrapped_demos=d["max_bootstrapped_demos"], max_labeled_demos=d["max_labeled_demos"]
            )
            tuned = opt.compile(InjectionClassifier(mc), trainset=trainset)
    summary["optimization_time_s"] = t_opt.elapsed
    tuned.save(str(art / "dspy_optimized.json"))
    for split, df in (("val", val_e), ("test", test_e)):
        p, st = evaluate_program(tuned, lm, df)
        save_predictions(df, p["score"].values, p["pred"].values, pred_path(res, "dspy_optimized", split))
        summary[f"optimized_{split}_stats"] = st
    summary["environment"] = environment_info()
    save_json(summary, res / "dspy_meta.json")
    save_json({"status": "complete", "provider": provider}, res / "dspy_status.json")
    log.info("DSPy experiment complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
