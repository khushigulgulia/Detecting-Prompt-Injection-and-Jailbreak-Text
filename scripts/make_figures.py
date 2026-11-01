"""Generate every figure/table that has real recorded data behind it."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import error_analysis, plots
from src.utils.config import load_config, resolve
from src.utils.logging_utils import get_logger

log = get_logger("make_figures")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res, figs, tables, proc = (resolve(cfg, k) for k in ("results_dir", "figures_dir", "tables_dir", "processed_dir"))
    error_analysis.run(res, tables)
    made = [
        plots.dataset_overview(proc, figs), plots.metrics_comparison(res, figs), plots.confusion_matrices(res, figs),
        plots.roc_curves(res, figs), plots.precision_recall_curves(res, figs), plots.learning_curve(res, figs),
        plots.dspy_optimization(res, figs), plots.inference_time(res, figs), plots.cost_comparison(res, figs),
        plots.error_analysis_plot(tables, figs),
    ]
    for m in made:
        if m is None:
            log.warning("A figure was skipped because its experiment data does not exist yet (pending).")
        else:
            log.info("wrote %s", m)


if __name__ == "__main__":
    main()
