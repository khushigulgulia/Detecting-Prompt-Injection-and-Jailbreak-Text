"""Plotting from recorded experiment outputs only.

Every function reads saved prediction/metric/meta files. If the data for a plot
does not exist (e.g. DSPy not yet run) the model/panel is omitted or the figure is
skipped with a message; nothing is invented.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.metrics import LABEL_NAMES, compute_metrics, load_predictions, pred_path

MODELS = {
    "classical": "TF-IDF + SVM",
    "distilbert": "DistilBERT",
    "dspy_baseline": "DSPy (baseline)",
    "dspy_optimized": "DSPy (optimized)",
}
COLORS = {"classical": "#4C72B0", "distilbert": "#DD8452", "dspy_baseline": "#8FBF8F", "dspy_optimized": "#2E8B57"}
sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)


def available(results: Path) -> dict[str, tuple[pd.DataFrame, str]]:
    """Return {model: (test_predictions, display)} for models with saved predictions."""
    out = {}
    for name, disp in MODELS.items():
        p = pred_path(results, name, "test")
        if p.exists():
            out[name] = (load_predictions(p), disp)
    return out


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def metrics_comparison(results: Path, figs: Path) -> Path:
    av = available(results)
    rows = []
    for n, (df, d) in av.items():
        m = compute_metrics(df["label"].values, df["pred"].values)
        for k in ("accuracy", "precision", "recall", "f1"):
            rows.append({"model": d, "metric": k.capitalize(), "value": m[k], "key": n})
    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    order = [MODELS[n] for n in av]
    sns.barplot(data=data, x="metric", y="value", hue="model", hue_order=order, ax=ax,
                palette=[COLORS[n] for n in av])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Test-set metrics (n={len(next(iter(av.values()))[0])})")
    ax.set_xlabel("")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, figs / "metrics_comparison.png")


def confusion_matrices(results: Path, figs: Path) -> Path:
    av = available(results)
    fig, axes = plt.subplots(1, len(av), figsize=(4.2 * len(av), 4), squeeze=False)
    for ax, (n, (df, d)) in zip(axes[0], av.items()):
        cm = np.array(compute_metrics(df["label"].values, df["pred"].values)["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        ax.set_title(d)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    return _save(fig, figs / "confusion_matrices.png")


def roc_curves(results: Path, figs: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n, (df, d) in available(results).items():
        fpr, tpr, _ = roc_curve(df["label"], df["score"])
        m = compute_metrics(df["label"].values, df["pred"].values, df["score"].values)
        lbl = f"{d} (AUC={m['roc_auc']:.3f})" + (" [hard labels]" if n.startswith("dspy") else "")
        ax.plot(fpr, tpr, label=lbl, color=COLORS[n], marker="o" if n.startswith("dspy") else None)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test)")
    ax.legend(fontsize=8)
    return _save(fig, figs / "roc_curves.png")


def precision_recall_curves(results: Path, figs: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n, (df, d) in available(results).items():
        p, r, _ = precision_recall_curve(df["label"], df["score"])
        m = compute_metrics(df["label"].values, df["pred"].values, df["score"].values)
        ax.plot(r, p, label=f"{d} (AP={m['average_precision']:.3f})", color=COLORS[n])
    ax.axhline(df["label"].mean(), color="k", ls="--", lw=1, label="prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (test)")
    ax.legend(fontsize=8)
    return _save(fig, figs / "precision_recall_curves.png")


def learning_curve(results: Path, figs: Path) -> Path | None:
    """Left: real retraining curves (test F1). Right: DSPy few-shot sweep (validation accuracy).

    The panels use different metrics/splits on purpose and must not be compared point-for-point.
    """
    p = results / "learning_curve.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    meta_p = results / "dspy_meta.json"
    sweep = json.loads(meta_p.read_text()).get("budget_sweep_val", []) if meta_p.exists() else []
    fig, axes = plt.subplots(1, 2 if sweep else 1, figsize=(12 if sweep else 7, 5), squeeze=False)
    ax = axes[0][0]
    for model, disp in (("classical", "TF-IDF + SVM"), ("distilbert", "DistilBERT")):
        d = df[df["model"] == model]
        if d.empty:
            continue
        g = d.groupby("n_train")["test_f1"].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(g["n_train"], g["mean"], yerr=g["std"].fillna(0), marker="o", capsize=3, label=disp, color=COLORS[model])
    ax.set_xlabel("Labelled training examples")
    ax.set_ylabel("Test F1 (mean ± std over seeds)")
    ax.set_title("Trained models: real retraining per point")
    ax.legend(fontsize=8)
    if sweep:
        ax2 = axes[0][1]
        ax2.plot([s["k_demos"] for s in sweep], [s["val_accuracy"] for s in sweep], "s-", color=COLORS["dspy_optimized"])
        ax2.set_xlabel("Few-shot demonstrations in the prompt (k)")
        ax2.set_ylabel("Validation accuracy")
        ax2.set_ylim(0.5, 1.0)
        ax2.set_title("DSPy few-shot sweep (single run, n=99 val)")
    return _save(fig, figs / "learning_curve.png")


def dspy_optimization(results: Path, figs: Path) -> Path | None:
    bp, op = pred_path(results, "dspy_baseline", "test"), pred_path(results, "dspy_optimized", "test")
    if not (bp.exists() and op.exists()):
        return None
    rows = []
    for n, d in (("dspy_baseline", "Baseline"), ("dspy_optimized", "Optimized")):
        df = load_predictions(pred_path(results, n, "test"))
        m = compute_metrics(df["label"].values, df["pred"].values)
        for k in ("accuracy", "precision", "recall", "f1"):
            rows.append({"version": d, "metric": k.capitalize(), "value": m[k]})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=pd.DataFrame(rows), x="metric", y="value", hue="version", ax=ax,
                palette=[COLORS["dspy_baseline"], COLORS["dspy_optimized"]])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("DSPy: baseline vs optimized (test)")
    return _save(fig, figs / "dspy_optimization.png")


def _meta(results: Path, name: str) -> dict[str, Any]:
    p = results / f"{name}_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def inference_time(results: Path, figs: Path) -> Path:
    rows = []
    for n in ("classical", "distilbert"):
        m = _meta(results, n)
        if m:
            rows.append({"model": MODELS[n], "ms": m["single_sample_latency_ms"], "tp": m["throughput_samples_per_s"], "key": n})
    d = _meta(results, "dspy")
    if d.get("optimized_test_stats"):
        st = d["optimized_test_stats"]
        rows.append({"model": "DSPy (optimized)", "ms": 1000 * st["latency_s_median"], "tp": st["n"] / st["total_time_s"], "key": "dspy_optimized"})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in ((axes[0], "ms", "Single-sample latency (ms, log)"), (axes[1], "tp", "Throughput (samples/s, log)")):
        ax.bar(df["model"], df[col], color=[COLORS[k] for k in df["key"]])
        ax.set_yscale("log")
        ax.set_title(title)
        for i, v in enumerate(df[col]):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
    return _save(fig, figs / "inference_time.png")


def cost_comparison(results: Path, figs: Path) -> Path | None:
    """Measured API cost per 1,000 samples (local compute is reported as time, not dollars)."""
    d = _meta(results, "dspy")
    if not d:
        return None
    st = d.get("optimized_test_stats", {})
    rows = [{"model": "TF-IDF + SVM", "cost": 0.0}, {"model": "DistilBERT", "cost": 0.0}]
    rows.append({"model": "DSPy (optimized)", "cost": st.get("cost_usd_per_1000") or 0.0})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df["model"], df["cost"], color=[COLORS["classical"], COLORS["distilbert"], COLORS["dspy_optimized"]])
    ax.set_ylabel("USD per 1,000 predictions (API only)")
    ax.set_title("Measured API cost (local inference = $0 API; see inference_time.png)")
    return _save(fig, figs / "cost_comparison.png")


def error_analysis_plot(tables: Path, figs: Path) -> Path | None:
    p = tables / "error_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    pv = df.pivot_table(index="model", columns="error_type", values="count", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pv.plot(kind="bar", ax=ax, color={"false_positive": "#E1A15A", "false_negative": "#C44E52"})
    ax.set_ylabel("Count (test)")
    ax.set_title("Errors by type")
    ax.tick_params(axis="x", rotation=0)
    return _save(fig, figs / "error_analysis.png")


def dataset_overview(processed: Path, figs: Path) -> Path:
    frames = [pd.read_parquet(processed / f"{s}.parquet").assign(split=s) for s in ("train", "val", "test")]
    df = pd.concat(frames)
    df["class"] = df["label"].map(dict(enumerate(LABEL_NAMES)))
    df["chars"] = df["text"].str.len()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.countplot(data=df, x="split", hue="class", ax=axes[0], order=["train", "val", "test"])
    axes[0].set_title("Class distribution per split")
    sns.histplot(data=df, x="chars", hue="class", log_scale=True, bins=30, ax=axes[1], element="step")
    axes[1].set_title("Text length (chars, log scale)")
    return _save(fig, figs / "dataset_overview.png")
