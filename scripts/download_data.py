"""Download datasets, inspect schemas, clean, deduplicate and split.

Usage: python scripts/download_data.py [--config configs/config.yaml] [--seed 42]
Outputs go to data/processed/ plus reports/tables/dataset_inspection.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.load_dataset import inspect_dataset, load_hf_dataset
from src.data.preprocess import clean_dataframe
from src.data.split_data import group_stratified_split, near_duplicate_groups, stratified_split
from src.utils.config import environment_info, load_config, resolve, save_json, set_seed
from src.utils.logging_utils import get_logger

log = get_logger("download_data")

# Raw column names, verified against the real schemas (see README, "Dataset").
PRIMARY_COLS = ("text", "label")
SECONDARY_COLS = {"jackhhao/jailbreak-classification": ("prompt", "type")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    raw_dir, out_dir, tables = resolve(cfg, "raw_dir"), resolve(cfg, "processed_dir"), resolve(cfg, "tables_dir")

    inspection: dict = {"environment": environment_info(), "datasets": {}}

    name = cfg["data"]["primary_dataset"]
    raw = load_hf_dataset(name, raw_dir)
    tcol, lcol = PRIMARY_COLS
    inspection["datasets"][name] = {"raw": inspect_dataset(raw, tcol, lcol)}
    # Verify label meaning from real samples instead of assuming it.
    for lab, samples in inspection["datasets"][name]["raw"]["samples"].items():
        log.info("[%s] label=%s samples: %s", name, lab, samples)
    clean, report = clean_dataframe(raw, tcol, lcol, source=name, min_chars=cfg["data"]["min_text_chars"])
    inspection["datasets"][name]["cleaning"] = report
    log.info("Cleaning report: %s", report)

    # Optional secondary datasets: merged into the pool (merge: true) or kept as an external set.
    external, merged = [], [clean]
    for sec in cfg["data"]["secondary_datasets"]:
        if not sec.get("enabled"):
            continue
        sname = sec["name"]
        try:
            sraw = load_hf_dataset(sname, raw_dir)
            stc, slc = SECONDARY_COLS[sname]
            inspection["datasets"][sname] = {"raw": inspect_dataset(sraw, stc, slc)}
            sclean, srep = clean_dataframe(sraw, stc, slc, source=sname)
            inspection["datasets"][sname]["cleaning"] = srep
            log.info("%s cleaned: %s", sname, srep)
            (merged if sec.get("merge") else external).append(sclean)
        except Exception as exc:  # noqa: BLE001 - optional dataset; document and continue
            log.warning("Optional dataset %s unavailable (%s); continuing without it.", sname, exc)
            inspection["datasets"][sname] = {"error": str(exc)}
    if len(merged) > 1:
        before = sum(len(m) for m in merged)
        pool, mrep = clean_dataframe(pd.concat(merged, ignore_index=True), "text", "label", source="merged")
        # clean_dataframe overwrote 'source'; restore it from the first occurrence of each text.
        src = pd.concat(merged, ignore_index=True).drop_duplicates("text").set_index("text")["source"]
        pool["source"] = pool["text"].map(src)
        mrep["rows_before_merge_dedup"] = before
        inspection["merge"] = mrep
        log.info("Merged pool: %s | per-source: %s", mrep, pool["source"].value_counts().to_dict())
        clean = pool
    inspection["pool_class_counts"] = clean["label"].value_counts().to_dict()
    inspection["pool_source_counts"] = clean["source"].value_counts().to_dict()

    sp = cfg["data"]["split"]
    thr = cfg["data"].get("group_similarity_threshold")
    if thr:
        groups = near_duplicate_groups(clean["text"].tolist(), thr)
        inspection["near_duplicate_clusters"] = {
            "threshold": thr, "n_groups": int(len(set(groups))), "n_texts": len(clean),
            "texts_in_multi_member_groups": int((pd.Series(groups).map(pd.Series(groups).value_counts()) > 1).sum()),
        }
        log.info("Near-duplicate grouping: %s", inspection["near_duplicate_clusters"])
        splits = group_stratified_split(clean, sp["train"], sp["val"], sp["test"], seed, thr)
    else:
        splits = stratified_split(clean, sp["train"], sp["val"], sp["test"], seed)
    for k, v in splits.items():
        v.to_parquet(out_dir / f"{k}.parquet")
        v.to_csv(out_dir / f"{k}.csv", index=False)
        log.info("%s: %d rows, label counts %s", k, len(v), v["label"].value_counts().to_dict())
    inspection["split_source_counts"] = {k: v["source"].value_counts().to_dict() for k, v in splits.items()}
    inspection["split_sizes"] = {k: len(v) for k, v in splits.items()}
    inspection["split_class_counts"] = {k: v["label"].value_counts().to_dict() for k, v in splits.items()}

    if external:
        ext = pd.concat(external, ignore_index=True)
        ext = ext[~ext["text"].isin(set(clean["text"]))]
        ext.to_parquet(out_dir / "external.parquet")
        inspection["external_size"] = len(ext)

    save_json(inspection, tables / "dataset_inspection.json")
    log.info("Done. Inspection saved to %s", tables / "dataset_inspection.json")


if __name__ == "__main__":
    main()
