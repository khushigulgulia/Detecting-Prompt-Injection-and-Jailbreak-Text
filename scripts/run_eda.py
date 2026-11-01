"""Print dataset inspection results and write the overview plot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.plots import dataset_overview
from src.utils.config import load_config, resolve


def main() -> None:
    cfg = load_config()
    tables = resolve(cfg, "tables_dir")
    insp_path = tables / "dataset_inspection.json"
    if not insp_path.exists():
        raise SystemExit("Run scripts/download_data.py first.")
    insp = json.loads(insp_path.read_text())
    for name, d in insp["datasets"].items():
        raw = d.get("raw")
        if not raw:
            print(f"{name}: unavailable ({d.get('error')})")
            continue
        print(f"\n== {name}: {raw['rows']} rows, classes {raw['class_distribution']}, dup texts {raw['duplicate_texts']}")
        print("   missing:", raw["missing"], "| cleaning:", d.get("cleaning"))
    print("\nSplit sizes:", insp["split_sizes"], insp["split_class_counts"])
    print("Wrote", dataset_overview(resolve(cfg, "processed_dir"), resolve(cfg, "figures_dir")))


if __name__ == "__main__":
    main()
