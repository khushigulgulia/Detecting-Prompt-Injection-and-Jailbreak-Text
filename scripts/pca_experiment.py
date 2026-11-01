"""Dimensionality-reduction experiment for Solution 1: PCA and TruncatedSVD (LSA) before a LinearSVC.

Pipeline A (true PCA)  : TF-IDF (100k) -> chi2 top-5,000 -> dense -> PCA(n) -> LinearSVC
Pipeline B (LSA / SVD) : TF-IDF (100k) -> TruncatedSVD(n) -> LinearSVC
n_components and C are chosen on VALIDATION F1 (ties -> fewer components); the test split is evaluated once for the
chosen pipelines, which are refit end-to-end as sklearn Pipelines on the training split only.
Existing results/ and artifacts/ files are not modified.

Usage: python scripts/pca_experiment.py
"""
from __future__ import annotations

import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

from src.evaluation.metrics import pred_path, save_predictions
from src.models.classical_ml import build_pipeline
from src.models.classical_reduced import build_reduced_pipeline
from src.utils.config import load_config, resolve, save_json, set_seed
from src.utils.logging_utils import get_logger

log = get_logger("pca_experiment")
K_PRE = 5000
N_GRID = [10, 25, 50, 100, 200, 400, 800, 1200]
C_GRID = [1.0, 10.0, 100.0]


def f1v(y, p) -> float:
    return float(f1_score(y, p, zero_division=0))


def summarize(y, p) -> dict:
    return dict(accuracy=float(accuracy_score(y, p)), precision=float(precision_score(y, p, zero_division=0)),
                recall=float(recall_score(y, p, zero_division=0)), f1=f1v(y, p),
                fp=int(((p == 1) & (y == 0)).sum()), fn=int(((p == 0) & (y == 1)).sum()))


def mcnemar_exact(y, pa, pb):
    ca, cb = pa == y, pb == y
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    m, k = a_only + b_only, min(a_only, b_only)
    return a_only, b_only, (1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m))


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    proc, res, tables, art = (resolve(cfg, k) for k in ("processed_dir", "results_dir", "tables_dir", "artifacts_dir"))
    train, val, test = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))
    ytr, yva, yte = (d["label"].values for d in (train, val, test))

    # ---- shared matrices for the sweep (fit on train only)
    feats = build_pipeline(cfg, "svm", 10.0, seed).named_steps["tfidf"]
    Xtr = feats.fit_transform(train["text"].tolist(), ytr)
    Xva, Xte = feats.transform(val["text"].tolist()), feats.transform(test["text"].tolist())
    sc, _ = chi2(Xtr, ytr)
    idx = np.argsort(-np.nan_to_num(sc))[:K_PRE]
    Dtr, Dva = Xtr[:, idx].toarray(), Xva[:, idx].toarray()

    # ---- full PCA once; slice leading components (PCA components are nested)
    t0 = time.perf_counter()
    pca_full = PCA(n_components=min(Dtr.shape), svd_solver="full", random_state=seed).fit(Dtr)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n90, n95 = int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.95) + 1)
    log.info("PCA on %d selected features: %d comps for 90%% var, %d for 95%% (%.1fs)", K_PRE, n90, n95,
             time.perf_counter() - t0)
    Ztr_p, Zva_p = pca_full.transform(Dtr), pca_full.transform(Dva)

    svd = TruncatedSVD(n_components=max(N_GRID), random_state=seed).fit(Xtr)
    Ztr_s, Zva_s = svd.transform(Xtr), svd.transform(Xva)

    rows = []
    for fam, Ztr, Zva, grid in (("pca", Ztr_p, Zva_p, sorted(set(N_GRID + [n90, n95]))),
                                ("svd", Ztr_s, Zva_s, N_GRID)):
        for n in grid:
            for C in C_GRID:
                clf = LinearSVC(C=C, class_weight="balanced", random_state=seed, max_iter=20000).fit(Ztr[:, :n], ytr)
                rows.append(dict(family=fam, n_components=n, C=C,
                                 val_f1=f1v(yva, (clf.decision_function(Zva[:, :n]) >= 0).astype(int))))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(tables / "pca_validation_sweep.csv", index=False)
    print(sweep.pivot_table(index=["family", "n_components"], columns="C", values="val_f1").round(4).to_string())

    def best(fam):
        d = sweep[sweep.family == fam].sort_values(["val_f1", "n_components"], ascending=[False, True])
        return d.iloc[0]

    bp, bs = best("pca"), best("svd")
    log.info("best PCA: n=%d C=%g val_f1=%.4f | best SVD: n=%d C=%g val_f1=%.4f", bp.n_components, bp.C, bp.val_f1,
             bs.n_components, bs.C, bs.val_f1)

    # ---- refit chosen configs end-to-end as sklearn Pipelines (train only); test evaluated once
    pipes = {
        "pca": build_reduced_pipeline(cfg, "pca", int(bp.n_components), float(bp.C), seed, K_PRE),
        "svd": build_reduced_pipeline(cfg, "svd", int(bs.n_components), float(bs.C), seed),
    }
    preds = {}
    for name, pipe in pipes.items():
        pipe.fit(train["text"].tolist(), ytr)
        for split, df in (("val", val), ("test", test)):
            s = pipe.decision_function(df["text"].tolist())
            save_predictions(df, s, (s >= 0).astype(int), pred_path(res, f"classical_{name}", split))
        preds[name] = (pipe.decision_function(test["text"].tolist()) >= 0).astype(int)
        joblib.dump(pipe, art / f"classical_{name}.joblib")

    base = pd.read_csv(pred_path(res, "classical", "test"))
    assert (base["label"].values == yte).all()
    pb = base["pred"].values
    entries = [("TF-IDF + SVM (main, no reduction)", Xtr.shape[1], pb),
               (f"chi2 {K_PRE} + PCA({int(bp.n_components)}) + SVM", int(bp.n_components), preds["pca"]),
               (f"TruncatedSVD({int(bs.n_components)}) + SVM", int(bs.n_components), preds["svd"])]
    out = []
    for i, (name, nfeat, p) in enumerate(entries):
        a, b, pv = (0, 0, 1.0) if i == 0 else mcnemar_exact(yte, p, pb)
        out.append(dict(model=name, n_dims=nfeat, **summarize(yte, p), only_this_correct=a, only_main_correct=b,
                        mcnemar_p=pv))
    tab = pd.DataFrame(out)
    tab.to_csv(tables / "pca_test_results.csv", index=False)
    print("\n", tab.round(4).to_string(index=False))

    pca_pipe = pipes["pca"]
    ev = float(pca_pipe.named_steps["pca"].explained_variance_ratio_.sum())
    save_json({"k_pre": K_PRE, "n_for_90pct_variance": n90, "n_for_95pct_variance": n95,
               "pca_best": {"n_components": int(bp.n_components), "C": float(bp.C), "val_f1": float(bp.val_f1),
                            "explained_variance": ev},
               "svd_best": {"n_components": int(bs.n_components), "C": float(bs.C), "val_f1": float(bs.val_f1)},
               "pc1_var": float(pca_full.explained_variance_ratio_[0]),
               "pc2_var": float(pca_full.explained_variance_ratio_[1])}, res / "classical_pca_meta.json")

    # ---- figure
    GRID, INK2 = "#e3e2dd", "#52514e"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    ax = axes[0]
    x = np.arange(1, len(cum) + 1)
    ax.plot(x, 100 * cum, color="#2a78d6", lw=2.2)
    for thr, n, col in ((90, n90, "#6b7f99"), (95, n95, "#b3392f")):
        ax.scatter([n], [thr], s=55, color=col, zorder=4)
        ax.annotate(f"{thr}% variance:\n{n} components", (n, thr), xytext=(-125, 6) if thr == 90 else (24, -44),
                    textcoords="offset points", fontsize=8, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    ax.set_xscale("log"); ax.set_xlabel("Principal components kept (log scale)"); ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title(f"PCA on {K_PRE:,} chi2-selected features", loc="left", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 102)
    ax = axes[1]
    for fam, col, lab in (("pca", "#2a78d6", f"chi2 {K_PRE} + PCA"), ("svd", "#eb6834", "TruncatedSVD (full TF-IDF)")):
        d = sweep[sweep.family == fam].groupby("n_components").val_f1.max().reset_index()
        ax.plot(d.n_components, d.val_f1, "-o", color=col, lw=2, ms=5, label=lab)
    base_val = pd.read_csv(pred_path(res, "classical", "val"))
    bf = f1v(base_val["label"].values, base_val["pred"].values)
    ax.axhline(bf, color=INK2, ls="--", lw=1)
    ax.text(N_GRID[0], bf + 0.003, "main model (no reduction)", fontsize=8, color=INK2)
    ax.set_ylim(0.81, 0.968)
    ax.scatter([bp.n_components], [bp.val_f1], s=170, facecolor="none", edgecolor="#b3392f", lw=2, zorder=5)
    ax.set_xscale("log"); ax.set_xlabel("Components kept (log scale)"); ax.set_ylabel("Validation F1 (best C)")
    ax.set_title("Validation F1 vs dimensions", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[2]
    Zte2 = pca_pipe.named_steps["pca"].transform(pca_pipe.named_steps["dense"].transform(
        pca_pipe.named_steps["select"].transform(pca_pipe.named_steps["tfidf"].transform(test["text"].tolist()))))[:, :2]
    for lab, col, name in ((0, "#6b7f99", "benign"), (1, "#d1495b", "injection")):
        m = yte == lab
        ax.scatter(Zte2[m, 0], Zte2[m, 1], s=16, alpha=0.65, color=col, label=name, edgecolor="none")
    v1, v2 = pca_full.explained_variance_ratio_[:2]
    ax.set_xlabel(f"PC1 ({100 * v1:.1f}% of variance)"); ax.set_ylabel(f"PC2 ({100 * v2:.1f}% of variance)")
    ax.set_title("Test prompts in the first two PCs", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="best")
    for a in axes:
        a.grid(color=GRID, lw=0.6); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(resolve(cfg, "figures_dir") / "fig_pca.png", dpi=250, bbox_inches="tight")
    log.info("wrote figure")


if __name__ == "__main__":
    main()
