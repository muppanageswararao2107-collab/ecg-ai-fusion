"""Step 1 — read PTB-XL once, build superclass labels, check the official folds, and fit
normalization on the training folds only.

    python scripts/prepare_data.py                    # full dataset
    python scripts/prepare_data.py --limit 500        # quick test on 500 records
    python scripts/prepare_data.py --ptbxl-dir /path/to/ptb-xl/1.0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecgai.config import load_config  # noqa: E402
from ecgai.data import prepare_cache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ptbxl-dir", default=None, help="overrides paths.ptbxl_dir in config.yaml")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N labelled records (testing)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.ptbxl_dir:
        cfg["paths"]["ptbxl_dir"] = args.ptbxl_dir
    prepare_cache(cfg, limit=args.limit)
    print("\nDone. Next: python scripts/train_all.py --time-test")


if __name__ == "__main__":
    main()
