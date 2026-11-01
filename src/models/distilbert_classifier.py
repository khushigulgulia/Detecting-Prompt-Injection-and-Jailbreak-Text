"""Solution 2: fine-tuned DistilBERT binary classifier (plain PyTorch loop)."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def pick_device() -> torch.device:
    """CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _encode(tok: Any, texts: list[str], max_len: int) -> dict[str, torch.Tensor]:
    return tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")


def _batches(df: pd.DataFrame, bs: int, shuffle: bool, seed: int) -> list[pd.DataFrame]:
    idx = np.arange(len(df))
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    return [df.iloc[idx[i : i + bs]] for i in range(0, len(df), bs)]


@torch.no_grad()
def predict_scores(model: Any, tok: Any, texts: list[str], max_len: int, device: torch.device, bs: int = 32) -> np.ndarray:
    """Return P(injection) for each text."""
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(texts), bs):
        enc = {k: v.to(device) for k, v in _encode(tok, texts[i : i + bs], max_len).items()}
        probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1]
        out.append(probs.float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_distilbert(
    cfg: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, mode: str, seed: int, epochs: int | None = None
) -> tuple[Any, Any, dict[str, Any]]:
    """Fine-tune and return (best_model, tokenizer, history).

    The best epoch is chosen by validation F1 with early stopping. The test set is
    never passed in here.
    """
    dcfg = cfg["distilbert"]
    p = dcfg["quick"] if mode == "quick" else dcfg["full"]
    if mode == "quick":
        train = train.sample(min(len(train), p["max_train_samples"]), random_state=seed)
        val = val.sample(min(len(val), p["max_eval_samples"]), random_state=seed)
    lr = p.get("learning_rate", 5e-5)
    wd = p.get("weight_decay", 0.01)
    n_epochs = epochs if epochs is not None else p["epochs"]
    patience = p.get("early_stopping_patience", 1)
    max_len = dcfg["max_seq_len"]
    device = pick_device()
    log.info("Training on %s (%s mode): %d train / %d val, %d epochs", device, mode, len(train), len(val), n_epochs)

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(dcfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(dcfg["model_name"], num_labels=2).to(device)

    if dcfg["class_weighting"] == "balanced":
        counts = np.bincount(train["label"].values, minlength=2).astype(float)
        w = torch.tensor(len(train) / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    else:
        w = None
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    total_steps = n_epochs * int(np.ceil(len(train) / p["batch_size"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, total_steps)))

    history: dict[str, Any] = {"epoch": [], "train_loss": [], "val_f1": [], "device": str(device)}
    best_f1, best_state, bad = -1.0, None, 0
    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for b in _batches(train, p["batch_size"], True, seed + epoch):
            enc = {k: v.to(device) for k, v in _encode(tok, b["text"].tolist(), max_len).items()}
            y = torch.tensor(b["label"].values, dtype=torch.long, device=device)
            loss = loss_fn(model(**enc).logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        vs = predict_scores(model, tok, val["text"].tolist(), max_len, device, bs=16)
        f1 = float(f1_score(val["label"].values, (vs >= 0.5).astype(int), zero_division=0))
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_f1"].append(f1)
        log.info("epoch %d loss=%.4f val_f1=%.4f", epoch + 1, np.mean(losses), f1)
        if device.type == "mps":
            torch.mps.empty_cache()  # return cached blocks to the OS; unified memory is scarce
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break
    history["train_time_s"] = time.perf_counter() - t0
    history["best_val_f1"] = best_f1
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tok, history


def save_checkpoint(model: Any, tok: Any, path: Path) -> float:
    """Save model+tokenizer; return on-disk size in MB."""
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def load_checkpoint(path: Path) -> tuple[Any, Any]:
    """Load a saved checkpoint, with a helpful error if it is missing."""
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"No DistilBERT checkpoint at {path}. Run scripts/train_distilbert.py --mode full first.")
    return AutoModelForSequenceClassification.from_pretrained(path), AutoTokenizer.from_pretrained(path)
