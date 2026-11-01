"""Feature-selection experiment for Solution 1 (TF-IDF + LinearSVC).

Compares, on the SAME split as the main run:
  * baseline : all TF-IDF features, LinearSVC (the existing model, read from saved predictions)
  * chi2     : SelectKBest(chi2, k) after TF-IDF, then LinearSVC. k and C chosen on VALIDATION F1.
  * l1       : L1-penalised LinearSVC (embedded selection). C chosen on VALIDATION F1.

Selectors are fit on the training split only. The test split is evaluated once, only for the configurations
that were chosen on validation. Existing results/ and artifacts/ files are NOT modified.

Usage: python scripts/feature_selection_experiment.py
Outputs: results/classical_fs_*.csv|json, reports/tables/feature_selection_*.csv, reports/figures/fig_feature_selection.png
"""
from __future__ import annotations

import sys
import time
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

from src.evaluation.metrics import pred_path, save_predictions
from src.models.classical_ml import build_pipeline
from src.utils.config import load_config, resolve, save_json, set_seed
from src.utils.logging_utils import get_logger

log = get_logger("feature_selection")
K_GRID = [250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
C_GRID = [1.0, 10.0]
L1_C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]


def f1v(y, p) -> float:
    return float(f1_score(y, p, zero_division=0))


def summarize(y, p) -> dict:
    tn = int(((p == 0) & (y == 0)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tp = int(((p == 1) & (y == 1)).sum())
    return dict(accuracy=float(accuracy_score(y, p)), precision=float(precision_score(y, p, zero_division=0)),
                recall=float(recall_score(y, p, zero_division=0)), f1=f1v(y, p), tn=tn, fp=fp, fn=fn, tp=tp)


def mcnemar_exact(y, pa, pb) -> tuple[int, int, float]:
    ca, cb = pa == y, pb == y
    a_only, b_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
    m, k = a_only + b_only, min(a_only, b_only)
    p = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
    return a_only, b_only, p


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    proc, res = resolve(cfg, "processed_dir"), resolve(cfg, "results_dir")
    tables, art = resolve(cfg, "tables_dir"), resolve(cfg, "artifacts_dir")
    train, val, test = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))
    ytr, yva, yte = (d["label"].values for d in (train, val, test))

    # Vectorise once (fit on train only). Selection on top of these matrices is equivalent to a pipeline
    # because every selector below is also fit on the training matrix only.
    feats = build_pipeline(cfg, "svm", 10.0, seed).named_steps["tfidf"]
    t0 = time.perf_counter()
    Xtr = feats.fit_transform(train["text"].tolist(), ytr)
    Xva, Xte = feats.transform(val["text"].tolist()), feats.transform(test["text"].tolist())
    names = np.array(feats.get_feature_names_out())
    n_all = Xtr.shape[1]
    log.info("TF-IDF space: %d features (fit %.1fs)", n_all, time.perf_counter() - t0)

    chi_scores, _ = chi2(Xtr, ytr)
    chi_scores = np.nan_to_num(chi_scores)
    order = np.argsort(-chi_scores)

    # ------------------------------------------------ validation sweep (chi2)
    rows = []
    for k in K_GRID + [n_all]:
        k = min(k, n_all)
        idx = order[:k]
        for C in C_GRID:
            clf = LinearSVC(C=C, class_weight="balanced", random_state=seed).fit(Xtr[:, idx], ytr)
            rows.append(dict(method="chi2" if k < n_all else "none", k=k, C=C,
                             val_f1=f1v(yva, (clf.decision_function(Xva[:, idx]) >= 0).astype(int))))
    # ------------------------------------------------ validation sweep (L1)
    for C in L1_C_GRID:
        clf = LinearSVC(penalty="l1", dual=False, C=C, class_weight="balanced", random_state=seed, max_iter=5000)
        clf.fit(Xtr, ytr)
        rows.append(dict(method="l1", k=int((clf.coef_ != 0).sum()), C=C,
                         val_f1=f1v(yva, (clf.decision_function(Xva) >= 0).astype(int))))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(tables / "feature_selection_validation_sweep.csv", index=False)
    print(sweep.to_string(index=False))

    def best_row(method: str) -> pd.Series:
        d = sweep[sweep.method == method].sort_values(["val_f1", "k"], ascending=[False, True])  # tie -> fewer features
        return d.iloc[0]

    bc, bl = best_row("chi2"), best_row("l1")
    log.info("best chi2: k=%d C=%s val_f1=%.4f | best l1: C=%s nnz=%d val_f1=%.4f",
             bc.k, bc.C, bc.val_f1, bl.C, bl.k, bl.val_f1)

    # ------------------------------------------------ refit chosen configs, evaluate test ONCE
    idx = order[: int(bc.k)]
    chi_clf = LinearSVC(C=float(bc.C), class_weight="balanced", random_state=seed).fit(Xtr[:, idx], ytr)
    l1_clf = LinearSVC(penalty="l1", dual=False, C=float(bl.C), class_weight="balanced", random_state=seed,
                       max_iter=5000).fit(Xtr, ytr)
    out = {}
    for name, score_fn in (("chi2", lambda X: chi_clf.decision_function(X[:, idx])),
                           ("l1", lambda X: l1_clf.decision_function(X))):
        for split, df, X in (("val", val, Xva), ("test", test, Xte)):
            s = score_fn(X)
            save_predictions(df, s, (s >= 0).astype(int), pred_path(res, f"classical_fs_{name}", split))
        out[name] = (score_fn(Xte) >= 0).astype(int)

    base = pd.read_csv(pred_path(res, "classical", "test"))
    assert (base["label"].values == yte).all()
    pb = base["pred"].values
    results = {"baseline (all features)": (pb, n_all), f"chi2 top-{int(bc.k)}": (out["chi2"], int(bc.k)),
               "L1 SVM": (out["l1"], int((l1_clf.coef_ != 0).sum()))}
    table = []
    for name, (p, nfeat) in results.items():
        s = summarize(yte, p)
        a_only, b_only, pv = mcnemar_exact(yte, p, pb) if name != "baseline (all features)" else (0, 0, 1.0)
        table.append(dict(model=name, n_features=nfeat, **s, mcnemar_vs_baseline_p=pv,
                          only_this_correct=a_only, only_baseline_correct=b_only))
    tab = pd.DataFrame(table)
    tab.to_csv(tables / "feature_selection_test_results.csv", index=False)
    print("\n", tab.round(4).to_string(index=False))

    joblib.dump({"features": feats, "idx": idx, "clf": chi_clf}, art / "classical_fs_chi2.joblib")

    # ------------------------------------------------ most informative n-grams (chi2 model weights)
    w = chi_clf.coef_[0]
    sel_names = names[idx]
    top = pd.DataFrame({"feature": sel_names, "weight": w, "chi2": chi_scores[idx]})
    top["feature"] = top.feature.str.replace(r"^(word|char)__", lambda m: m.group(1) + ": ", regex=True)
    top_inj = top.sort_values("weight", ascending=False).head(15)
    top_ben = top.sort_values("weight").head(15)
    pd.concat([top_inj.assign(direction="injection"), top_ben.assign(direction="benign")]).to_csv(
        tables / "feature_selection_top_features.csv", index=False)
    print("\nTop injection n-grams:\n", top_inj.to_string(index=False))
    print("\nTop benign n-grams:\n", top_ben.to_string(index=False))

    kind = pd.Series(sel_names).str.split("__").str[0].value_counts().to_dict()
    save_json({"n_all_features": int(n_all), "chi2_best": {"k": int(bc.k), "C": float(bc.C), "val_f1": float(bc.val_f1),
               "selected_by_vectorizer": {k: int(v) for k, v in kind.items()}},
               "l1_best": {"C": float(bl.C), "nonzero": int((l1_clf.coef_ != 0).sum()), "val_f1": float(bl.val_f1)}},
              res / "classical_fs_meta.json")

    # ------------------------------------------------ figure
    COLOR = {"classical": "#2a78d6"}
    GRID, INK2 = "#e3e2dd", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for C, ls in zip(C_GRID, ("-", "--")):
        d = sweep[(sweep.method.isin(["chi2", "none"])) & (sweep.C == C)].sort_values("k")
        ax.plot(d.k, d.val_f1, ls, marker="o", ms=5, lw=2, color=COLOR["classical"], label=f"chi2 + SVM, C={C:g}")
    ax.axvline(n_all, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(0.884, 0.968)
    ax.text(n_all * 0.9, 0.8865, f"all {n_all:,} features", ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax.scatter([bc.k], [bc.val_f1], s=150, facecolor="none", edgecolor="#b3392f", lw=2, zorder=5)
    ax.text(bc.k * 1.14, bc.val_f1 + 0.0035, "chosen on validation", color="#b3392f", fontsize=8, va="bottom", ha="left")
    ax.set_xscale("log"); ax.set_xlabel("Features kept by chi-squared (log scale)"); ax.set_ylabel("Validation F1")
    ax.set_title("Validation F1 vs number of features", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax = axes[1]
    both = pd.concat([top_ben.head(10).iloc[::-1], top_inj.head(10).iloc[::-1]])
    ax.barh(range(len(both)), both.weight, color=["#2a78d6" if x < 0 else "#eb6834" for x in both.weight], zorder=3)
    ax.set_yticks(range(len(both)), [repr(f)[1:-1] for f in both.feature], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("SVM weight  (left: benign, right: injection)")
    ax.set_title("Most informative selected n-grams", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(resolve(cfg, "figures_dir") / "fig_feature_selection.png", dpi=250,
                bbox_inches="tight")
    log.info("wrote figure")


if __name__ == "__main__":
    main()
