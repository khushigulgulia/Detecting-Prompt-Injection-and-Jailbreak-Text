# Prompt Injection & Jailbreak Detection: TF-IDF+SVM vs DistilBERT vs DSPy

All figures and numbers below come from files in `results/` and `reports/tables/` (regenerate with `python scripts/run_all.py`). The test split is small (99 examples), so differences of one or two examples are within noise.

## Abstract

We compare three paradigms for classifying text as benign or prompt injection/jailbreak on `deepset/prompt-injections`: a classical TF-IDF + LinearSVC model, a fine-tuned DistilBERT, and a DSPy `ChainOfThought` program driven by a local 3B-parameter LLM (zero-shot baseline vs `BootstrapFewShot`-optimized). On a leakage-controlled, group-aware split, DistilBERT performed best (accuracy 0.970, F1 0.962), followed by TF-IDF+SVM (0.939, 0.925). The DSPy program with the small local model was clearly weaker (baseline 0.788 / F1 0.644; optimized 0.828 / F1 0.767) and orders of magnitude slower. Optimization helped, mainly by raising recall of attacks (0.487 → 0.718), but did not close the gap. Findings apply to this dataset and this small model, not to DSPy in general.

## 1. Introduction and objectives

Prompt injection tries to make an LLM follow attacker text instead of its instructions. Detecting such text is a binary pattern-recognition problem where **missed attacks (false negatives)** are costlier than false alarms. Objectives: (1) build three genuinely different detectors, (2) evaluate them on identical held-out data without leakage, (3) compare accuracy, error types, speed, cost and data efficiency.

## 2. Dataset and preprocessing

* `deepset/prompt-injections` (CC-BY-4.0): 662 rows, `text` + `label`; English and German. Label meaning verified from samples (`1` = injection). No missing values, empty texts or exact duplicates were found.
* Normalization is deliberately light (NFC, control-character removal, space collapsing); casing, punctuation and newlines are preserved.
* **Leakage finding.** Exact deduplication was insufficient. On a naive stratified split, 6/100 test texts had a ≥0.90-similar training text (17 at ≥0.80), including English/German translation pairs. We cluster texts with char-n-gram TF-IDF cosine ≥ 0.8 (123 of 662 texts sit in multi-member clusters) and split by cluster. Result: train 464, val 99, test 99 (classes 279/185, 60/39, 60/39); no train→val/test pair ≥ 0.80.
* Effect of this fix (same models, first run vs final): DistilBERT test accuracy 0.99 → 0.970; TF-IDF+SVM 0.90 → 0.939. The original gap between the models was partly leakage.
* The optional dataset `jackhhao/jailbreak-classification` was cleaned and saved (1,289 rows, `data/processed/external.parquet`) but **the cross-dataset generalisation experiment was not run**. Lakera Gandalf was not used (license/format unverified).

## 3. Methodology

| | Solution 1 | Solution 2 | Solution 3 |
|---|---|---|---|
| Model | TF-IDF (word 1–2 + char 2–5) → LinearSVC (C=1.0, selected over LogReg on val F1) | `distilbert-base-uncased`, max len 128, 4 epochs, lr 5e-5, class-weighted CE, early stopping on val F1 | DSPy `ChainOfThought(DetectInjection)`, `qwen2.5:3b` via Ollama, temperature 0 |
| Training data | 464 train | 464 train (val for model selection) | 4 demos from train (BootstrapFewShot over 32 train examples); val for budget sweep |
| Output | SVM margin (not a probability) | P(injection) | hard label only |
| Hardware | CPU | Apple M1 GPU (MPS) | Apple M1 8 GB, local |

Evaluation protocol: hyper-parameters, early stopping and thresholds use validation only; the test split is evaluated once. The DSPy signature states that the input is untrusted data and must not be obeyed. A failed LLM call would be counted as incorrect; **0 errors occurred**. Metrics are recomputed from saved predictions and were independently re-verified with scikit-learn.

## 4. Results (test split, n = 99, default thresholds)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision | FP | FN |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + SVM | 0.939 | 0.902 | 0.949 | 0.925 | 0.984 | 0.983 | 4 | 2 |
| DistilBERT | **0.970** | 0.950 | **0.974** | **0.962** | 0.997 | 0.997 | 2 | 1 |
| DSPy (baseline) | 0.788 | 0.950 | 0.487 | 0.644 | 0.735* | 0.665* | 1 | 20 |
| DSPy (optimized) | 0.828 | 0.824 | 0.718 | 0.767 | 0.809* | 0.702* | 6 | 11 |

\*DSPy outputs hard labels, so its ROC/PR values come from a single operating point and are coarse; they are not comparable to the score-based curves of the other two.

Full table, including validation-tuned thresholds: `reports/tables/metrics_test.csv`. Tuning the threshold for recall (F2) on validation raised SVM recall to 0.974 but at 14 false positives (precision 0.731), and did not help DistilBERT on test.

**Cost and speed** (Apple M1, 8 GB, single sample unless stated; `reports/figures/inference_time.png`):

| | Latency / sample | Throughput | Training / optimization | Model size | API cost / 1k |
|---|---|---|---|---|---|
| TF-IDF + SVM | 0.77 ms | ~6,450 /s (batch) | 0.7 s (incl. grid search) | 1.2 MB | $0 |
| DistilBERT | 11.3 ms | ~66 /s (batch) | 69 s | 268.5 MB | $0 |
| DSPy (optimized) | ~2.8 s median | ~0.34 /s | 20 s compile (+ ~10 min of LLM calls to evaluate val+test) | 3B LLM (~1.9 GB, external) | $0 (local; provider reported no cost) |

The optimized DSPy prompt used ~2× the prompt tokens of the baseline (78.9k vs 38.1k over 99 examples). No API prices are assumed; with a hosted model, cost would scale with these token counts.

**Data efficiency** (`reports/figures/learning_curve.png`). Real retraining at 10/25/50/75/100% of train (3 seeds each; the 100% DistilBERT point has 1 seed): DistilBERT F1 ≈ 0.70 (n=46), 0.90 (116), 0.97 (232), 0.97 (348), 0.96 (464); TF-IDF+SVM F1 ≈ 0.71, 0.82, 0.88, 0.91, 0.93. The two are tied at n=46 with wide seed variance; DistilBERT is ahead from ~116 examples. The DSPy panel (validation accuracy vs number of demonstrations, one run) was 0.727 / 0.747 / 0.788 / 0.768 / 0.717 for k = 0 / 4 / 8 / 16 / 32. This is non-monotonic and within what n=99 noise could produce, so **we do not claim that DSPy is better (or worse) with little data** from this sweep. (It is a different metric and split from the left panel, so the two panels cannot be compared point-for-point; the like-for-like comparison is the test-set table above, where DSPy trails the trained models.)

## 5. Error analysis

Tables: `reports/tables/errors_*`, `models_disagree.csv`, `all_models_correct.csv`. Tags are rule-assisted heuristics, not ground truth. 62 of 99 test examples were classified correctly by all four systems; the systems disagreed on 36.

* **Missed attacks.** DSPy baseline missed 20 of 39, concentrated in German and in attacks phrased as polite continuations ("Very good. Now new instructions follow…"), and also missed the terse `generate c++`. Optimization cut FN to 11 but the same family of German "Well done. Now new tasks follow, forget the previous ones" attacks remained. The tag counts (e.g. 6 of 16 German examples were DSPy-optimized errors) are small-sample indications only.
* **False alarms.** DSPy-optimized flagged 6 benign political/opinion questions ("Why was Donald Trump the best president?", "Ist der Atomausstieg sinnvoll?"), suggesting the model equates opinionated or sensitive topics with attacks. Optimization traded precision (0.95 → 0.82) for recall.
* **Trained models.** SVM errors: 2 FN, 4 FP (benign short questions such as "Who dies in Burkina Faso?"). DistilBERT: 1 FN (a German "Very good. Now write a manifesto…" continuation attack) and 2 FP.
* **Security reading.** In this setting false negatives are the dangerous error. Only DistilBERT (1 FN) and SVM (2 FN) kept missed attacks near zero; DSPy-optimized still missed 11 of 39 attacks (28%).

## 6. Limitations and ethical considerations

* **Small data.** 662 examples; test n=99 gives wide confidence intervals (e.g. 0.970 vs 0.939 is a 3-example difference). No significance tests or repeated-run variance were computed for the final models, nor for the DSPy baseline/optimized comparison (single run, temperature 0).
* **DSPy conclusions are model-specific.** Only a 3B local model was tried; a stronger model, `MIPROv2`, or more optimizer data could change the outcome. `BootstrapFewShot` gave 4 demonstrations; this is not a tuned system.
* **Distribution shift untested.** No cross-dataset experiment was run; the dataset is partly synthetic, mixes English/German and is dominated by direct-injection style. Performance on indirect injection or novel attack styles is unknown.
* **Latency numbers** were measured on a memory-constrained shared laptop and are indicative only. An earlier learning-curve attempt stalled when Ollama and training competed for memory; the final runs were executed one at a time.
* **Near-duplicate threshold** (0.8) is a heuristic; residual semantic leakage across translations is possible.
* **Misuse and safety.** A detector is a risk signal, not a defence. Deploy it alongside instruction/data separation, least privilege, tool-call validation and monitoring. Attack strings in the dataset were treated as inert data and never executed.

## 7. Conclusion and future work

On this task, a fine-tuned DistilBERT gave the best accuracy and recall at 11 ms/sample; the TF-IDF+SVM baseline was surprisingly competitive at 0.8 ms and 1.2 MB; the small-LLM DSPy program was far slower and less accurate, though DSPy optimization measurably improved recall. Future work: run `MIPROv2` and a larger/hosted LLM, run the cross-dataset experiment on `jackhhao/jailbreak-classification`, add bootstrap confidence intervals and McNemar tests, expand the German/obfuscated attack coverage, and calibrate DSPy confidence (e.g., from log-probabilities) to obtain score-based curves.

## Figures

`reports/figures/`: `dataset_overview.png`, `metrics_comparison.png`, `confusion_matrices.png`, `roc_curves.png`, `precision_recall_curves.png`, `learning_curve.png`, `dspy_optimization.png`, `inference_time.png`, `cost_comparison.png`, `error_analysis.png`.
