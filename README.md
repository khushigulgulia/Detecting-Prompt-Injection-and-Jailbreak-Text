# Detecting-Prompt-Injection-and-Jailbreak-Text

Detecting Prompt Injection and Jailbreak Text: A Leakage-Controlled Comparison of TF-IDF+SVM, DistilBERT and a DSPy Program on a Small Local LLM

## Prompt Injection & Jailbreak Detection: three approaches compared

A pattern-recognition case study that classifies text as **benign (0)** or **prompt injection / jailbreak (1)** using three different paradigms on identical, leakage-controlled data:

| # | Paradigm | Approach |
|---|----------|----------|
| 1 | Classical ML | TF-IDF (word + char n-grams) → LinearSVC / Logistic Regression (validation-selected) |
| 2 | Deep learning | Fine-tuned `distilbert-base-uncased` |
| 3 | DSPy | `ChainOfThought` LLM program, un-optimized baseline vs. `BootstrapFewShot` / `MIPROv2`-optimized |

All numbers in `reports/` are recomputed from saved prediction files; nothing is hard-coded. Experiments that have not been run are shown as **pending**, never filled in.

## Layout

```
configs/config.yaml     all parameters (seed, splits, TF-IDF, DistilBERT, DSPy, thresholds)
src/data/               load_dataset, preprocess, split_data (group-aware)
src/models/             classical_ml, distilbert_classifier, dspy_classifier
src/evaluation/         metrics, plots, error_analysis
src/app.py              Streamlit dashboard
scripts/                download_data, run_eda, train_classical, train_distilbert,
                        optimize_dspy, learning_curve, evaluate_models, make_figures, run_all
tests/                  pytest suite (no API key / GPU needed)
notebooks/              01_eda.ipynb, 02_colab_distilbert.ipynb
reports/                figures/, tables/, final_report.md
results/                predictions + run metadata (JSON/CSV)
artifacts/              fitted models (git-ignored)
```

## Dataset

* **Primary:** [`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections), 662 rows, columns `text`, `label` (CC-BY-4.0). Contains **English and German** text. The label mapping was verified from real samples: `1` = injection (e.g. "Forget all previous tasks…"), `0` = benign.
* **Optional external:** [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification) (Apache-2.0; columns `prompt`, `type` ∈ {benign, jailbreak}). It is cleaned and saved as `data/processed/external.parquet` as a generalisation set.
* **Lakera Gandalf:** not used; its license/format was not verified, so it is intentionally excluded.

### Preprocessing and leakage control
* Unicode NFC, control-character removal, space collapsing. **Casing, punctuation and newlines are kept** (they can be attack signals); no stop-word or token removal.
* Missing rows, empty text, unmappable labels, exact duplicates and *conflicting* duplicates (same text, different label) are dropped and counted in `reports/tables/dataset_inspection.json`.
* **Group-aware split (70/15/15):** exact-dedup is not enough. The dataset contains near-duplicates and English/German translation pairs. Texts with char-n-gram TF-IDF cosine ≥ 0.8 are clustered and kept in a single split (`StratifiedGroupKFold`). On the naive stratified split, 6 of 100 test texts had a ≥0.90-similar training text; after grouping, none are ≥0.80.
* TF-IDF is fit on training data only; hyper-parameters, early stopping and thresholds use validation only; the test split is touched once for final evaluation.

## Setup

Requires Python 3.10+ (developed on 3.11).

```bash
# macOS / Linux
python3.11 -m venv .venv && source .venv/bin/activate
# Windows (PowerShell)
#   py -3.11 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

* **CPU:** everything runs; DistilBERT fine-tuning on CPU is slow (tens of minutes).
* **GPU:** CUDA (NVIDIA) or Apple MPS is auto-detected. This project's full run was done on an Apple M1 (MPS).
* **Google Colab:** open `notebooks/02_colab_distilbert.ipynb` (T4 GPU) for the full DistilBERT experiment.

## Run

```bash
python scripts/download_data.py          # download, inspect, clean, dedupe, group-split
python scripts/run_eda.py                # dataset summary + plot
python scripts/train_classical.py        # Solution 1
python scripts/train_distilbert.py --mode quick   # 1-minute smoke test (saves nothing)
python scripts/train_distilbert.py --mode full    # Solution 2
python scripts/optimize_dspy.py          # Solution 3 (needs an LLM, see below)
python scripts/learning_curve.py         # real retraining at 10/25/50/75/100% of train (3 seeds)
python scripts/evaluate_models.py        # metric tables from saved predictions
python scripts/make_figures.py           # all figures + error analysis tables
streamlit run src/app.py                 # dashboard
pytest                                   # tests
python scripts/run_all.py                # everything, in order
```

Common overrides: `--seed`, `--epochs`, `--mode`, `--provider`, `--optimizer`, `--n-train`, `--eval-max`, `--budgets 0,4,8`.

## DSPy setup (no API key required)

**Local, free (default): Ollama**
```bash
brew install ollama            # or https://ollama.com/download
ollama serve &                 # starts the local server
ollama pull qwen2.5:3b         # ~2 GB; fits an 8 GB laptop
python scripts/optimize_dspy.py --provider ollama
```
**Hosted:** copy `.env.example` to `.env`, set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (never commit it), then `--provider openai|anthropic`. Without a reachable LLM the script writes `results/dspy_status.json` = `pending` and exits with code 2.

Notes for this DSPy version (3.4.x): Ollama is called through its OpenAI-compatible `/v1` endpoint and an empty API key is rejected, so the code passes a placeholder (Ollama ignores it).

**DSPy evaluation protocol:** demonstrations / optimizer examples come from **train** only; MIPROv2 selection uses **val**; **test** is used once for the final baseline and optimized numbers. DSPy emits hard labels, so its ROC/PR curves have few points (flagged in the metric notes); it exposes no calibrated probability.

## Using the dashboard safely

*Test a Prompt* never sends text to an external API unless you pick the DSPy model **and** tick the provider-notice checkbox. A classifier is a risk signal, not a defense: combine it with instruction/data separation, least privilege, tool-call validation and monitoring.

## Reproducibility

Seeds, configuration, package versions, timestamps and dataset sizes are stored in `results/*_meta.json` and `reports/tables/dataset_inspection.json`. Datasets and models are cached (`data/raw/`, Hugging Face cache).

## Troubleshooting

| Problem | Fix |
|---|---|
| `DSPy evaluation PENDING` | Start Ollama (`ollama serve`) and pull the model, or set an API key |
| `404 page not found` from Ollama | Use the OpenAI-compatible `/v1` base URL (handled in `make_lm`) |
| `empty credential under 'ollama'` | Pass any non-empty `api_key` (handled in `make_lm`) |
| First DSPy call takes > 1 min | Ollama is loading the model into memory; later calls take seconds |
| DistilBERT very slow | Use Colab GPU or reduce `--epochs` |
| Dashboard says "No checkpoint" | Run `train_distilbert.py --mode full` first |
