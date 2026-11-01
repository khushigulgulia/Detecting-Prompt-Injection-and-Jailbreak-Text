"""Streamlit dashboard. Run: streamlit run src/app.py

Shows only recorded experiment outputs. Missing results are displayed as pending.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.utils.config import load_config

cfg = load_config()
FIG, TAB, RES, ART = (ROOT / cfg["paths"][k] for k in ("figures_dir", "tables_dir", "results_dir", "artifacts_dir"))
st.set_page_config(page_title="Prompt Injection Detection", page_icon="🛡️", layout="wide")

DEFENSE_NOTE = (
    "⚠️ A prediction is a **risk classification, not a security guarantee**. Model-based detectors can be "
    "bypassed. Combine them with instruction/data separation, least-privilege tool access, tool-call "
    "validation, output filtering and monitoring (defense in depth)."
)


def show_fig(name: str, caption: str | None = None) -> None:
    p = FIG / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"`{name}` is pending: the experiment that produces it has not been run yet.")


def read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def metrics_table() -> pd.DataFrame | None:
    p = TAB / "metrics_test.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_classical():
    import joblib

    return joblib.load(ART / "classical_model.joblib")


@st.cache_resource
def load_distil():
    from src.models.distilbert_classifier import load_checkpoint

    return load_checkpoint(ART / "distilbert")


def page_overview() -> None:
    st.title("🛡️ Prompt Injection & Jailbreak Detection")
    st.write("Compares three paradigms on one binary task (**0 = benign, 1 = injection/jailbreak**): "
             "**TF-IDF + SVM** (classical), **DistilBERT** (deep learning) and a **DSPy** LLM program.")
    insp = read_json(TAB / "dataset_inspection.json")
    if insp:
        c = st.columns(3)
        sizes = insp.get("split_sizes", {})
        for col, k in zip(c, ("train", "val", "test")):
            col.metric(f"{k} examples", sizes.get(k, "?"), help=str(insp["split_class_counts"].get(k)))
        ng = insp.get("near_duplicate_clusters")
        if ng:
            st.caption(f"Group-aware split: {ng['texts_in_multi_member_groups']} texts sit in near-duplicate clusters "
                       f"(cosine ≥ {ng['threshold']}) kept within a single split to avoid leakage.")
    show_fig("dataset_overview.png")
    st.markdown("**Dataset:** `deepset/prompt-injections` (CC-BY-4.0). Contains English and German text.")
    st.warning(DEFENSE_NOTE)


def page_compare() -> None:
    st.title("Compare Models")
    df = metrics_table()
    if df is None:
        st.info("No metrics yet. Run `python scripts/evaluate_models.py`.")
        return
    status = read_json(TAB / "evaluation_status.json")
    if status.get("pending"):
        st.warning(f"Pending (no real results yet): {', '.join(status['pending'])}")
    variant = st.radio("Threshold variant", sorted(df["variant"].unique()), horizontal=True,
                       help="Tuned thresholds were chosen on validation data only.")
    cols = ["display", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision", "fp", "fn"]
    st.dataframe(df[df["variant"] == variant][cols].round(4), hide_index=True, width="stretch")
    st.caption("fp = benign flagged as attack. fn = missed attack (the costlier error in a security setting).")
    for f in ("metrics_comparison.png", "confusion_matrices.png", "roc_curves.png", "precision_recall_curves.png",
              "learning_curve.png", "inference_time.png", "cost_comparison.png"):
        show_fig(f)


def page_test() -> None:
    st.title("Test a Prompt")
    st.warning(DEFENSE_NOTE)
    options = ["TF-IDF + SVM", "DistilBERT", "DSPy (LLM, sends text to configured provider)"]
    choice = st.selectbox("Model", options)
    text = st.text_area("Text to classify (treated purely as data, never executed)", height=140)
    consent = True
    if choice.startswith("DSPy"):
        consent = st.checkbox("I understand the configured LLM provider will receive this text.")
    if not st.button("Classify") or not text.strip():
        return
    if not consent:
        st.error("Please confirm the provider notice first.")
        return
    try:
        if choice == "TF-IDF + SVM":
            from src.models.classical_ml import decision_scores

            pipe = load_classical()
            meta = read_json(RES / "classical_meta.json")
            s = float(decision_scores(pipe, [text])[0])
            pred = int(s >= meta.get("default_threshold", 0.0))
            st.metric("Prediction", "injection" if pred else "benign")
            st.caption(f"Score {s:.3f} ({meta.get('score_type', 'score')}; an SVM margin is not a probability).")
        elif choice == "DistilBERT":
            from src.models.distilbert_classifier import pick_device, predict_scores

            model, tok = load_distil()
            model.to(pick_device())
            p = float(predict_scores(model, tok, [text], cfg["distilbert"]["max_seq_len"], pick_device())[0])
            st.metric("Prediction", "injection" if p >= 0.5 else "benign")
            st.progress(min(max(p, 0.0), 1.0), text=f"P(injection) = {p:.3f}")
        else:
            import dspy

            from src.models.dspy_classifier import InjectionClassifier, lm_status, load_program, make_lm

            ok, why = lm_status(cfg)
            if not ok:
                st.error(f"DSPy model unavailable: {why}")
                return
            dspy.configure(lm=make_lm(cfg))
            opt = ART / "dspy_optimized.json"
            prog = load_program(str(opt)) if opt.exists() else InjectionClassifier()
            out = prog(text=text)
            st.metric("Prediction", out.label)
            st.caption(("Optimized" if opt.exists() else "Baseline (no optimized program saved)") + " DSPy program.")
            if getattr(out, "reasoning", None):
                st.write("Reasoning:", out.reasoning)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - surface any model error to the user
        st.error(f"Prediction failed: {exc}")


def page_errors() -> None:
    st.title("Error Analysis")
    st.caption("Tags are heuristic aids for reviewing examples, not ground truth.")
    show_fig("error_analysis.png")
    summ = TAB / "error_summary.csv"
    if not summ.exists():
        st.info("Run `python scripts/make_figures.py` first.")
        return
    models = sorted(pd.read_csv(summ)["model"].unique())
    m = st.selectbox("Model", models)
    kind = st.radio("Error type", ["false_negatives", "false_positives"], horizontal=True)
    f = TAB / f"errors_{m}_{kind}.csv"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    if df.empty:
        st.success("No errors of this type for this model on the test split.")
        return
    all_tags = sorted({t for s in df["tags"] for t in s.split("|")})
    pick = st.multiselect("Filter by category tag", all_tags)
    if pick:
        df = df[df["tags"].apply(lambda s: all(t in s.split("|") for t in pick))]
    st.dataframe(df[["text", "length", "tags"]], hide_index=True, width="stretch")


def page_dspy() -> None:
    st.title("DSPy Optimization")
    status = read_json(RES / "dspy_status.json")
    meta = read_json(RES / "dspy_meta.json")
    if status.get("status") != "complete" or not meta:
        st.warning("**Pending.** The DSPy experiment has not been run, so no DSPy results exist.")
        st.code(status.get("reason", "No status file yet."))
        st.markdown("Run with a local Ollama model (no key) or an API key: `python scripts/optimize_dspy.py`")
        return
    st.write(f"Model: `{meta['model']}` · optimizer: `{meta['optimizer']}` · optimizer training examples: "
             f"{meta['n_opt_train']} · eval sizes val/test: {meta['n_val_eval']}/{meta['n_test_eval']} · "
             f"optimization time: {meta['optimization_time_s']:.0f}s")
    show_fig("dspy_optimization.png")
    sweep = pd.DataFrame(meta.get("budget_sweep_val", []))
    if not sweep.empty:
        st.subheader("Few-shot budget sweep (validation)")
        st.dataframe(sweep[["k_demos", "val_accuracy", "n_errors", "latency_s_median"]].round(3), hide_index=True)


PAGES = {"Overview": page_overview, "Compare Models": page_compare, "Test a Prompt": page_test,
         "Error Analysis": page_errors, "DSPy Optimization": page_dspy}
st.sidebar.title("Navigation")
PAGES[st.sidebar.radio("Page", list(PAGES))]()
