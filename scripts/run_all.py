"""Run the full pipeline in order. DSPy is skipped (reported pending) without an LLM.

Usage: python scripts/run_all.py [--distilbert-mode quick|full] [--with-learning-curve]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, allow_codes: tuple[int, ...] = (0,)) -> None:
    print(f"\n$ python {' '.join(args)}")
    code = subprocess.call([sys.executable, *args], cwd=ROOT)
    if code not in allow_codes:
        raise SystemExit(f"Step failed ({code}): {' '.join(args)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilbert-mode", choices=["quick", "full"], default="full")
    ap.add_argument("--with-learning-curve", action="store_true")
    a = ap.parse_args()
    run("scripts/download_data.py")
    run("scripts/run_eda.py")
    run("scripts/train_classical.py")
    run("scripts/train_distilbert.py", "--mode", a.distilbert_mode)
    run("scripts/optimize_dspy.py", allow_codes=(0, 2))  # 2 = pending (no LLM available)
    if a.with_learning_curve:
        run("scripts/learning_curve.py")
    run("scripts/evaluate_models.py")
    run("scripts/make_figures.py")


if __name__ == "__main__":
    main()
