"""Step 2 — train CNN, LSTM, Transformer and the parallel fusion model with the same protocol.

    python scripts/train_all.py --time-test                    # 1 epoch per model, prints a time estimate
    python scripts/train_all.py                                # all models, all seeds in config.yaml
    python scripts/train_all.py --models cnn fusion --seeds 0  # a subset
    python scripts/train_all.py --smoke                        # tiny run to check that everything works

Only the training folds (1-8) and the validation fold (9) are loaded here. The test fold
is not read by this script at all.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecgai.config import load_config, path  # noqa: E402
from ecgai.data import labels_from_meta, load_cache, split_indices  # noqa: E402
from ecgai.models import MODEL_NAMES  # noqa: E402
from ecgai.preprocess import load_normalization  # noqa: E402
from ecgai.train import get_device, standardize_split, train_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    ap.add_argument("--time-test", action="store_true", help="train 1 epoch per model and estimate total time")
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 batches per model (pipeline check only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    classes = cfg["data"]["classes"]
    X, meta = load_cache(cfg)
    splits = split_indices(meta["strat_fold"].to_numpy(), cfg["data"])
    norm = load_normalization(path(cfg, "artifacts_dir") / "norm.json")

    print(f"Device: {get_device(cfg['training'].get('device', 'auto'))}")
    print("Standardizing train/val with the saved TRAINING statistics ...")
    Xtr = standardize_split(X, splits["train"], norm["mean"], norm["std"])
    Xva = standardize_split(X, splits["val"], norm["mean"], norm["std"])
    Y = labels_from_meta(meta, classes)
    Ytr, Yva = Y[splits["train"]], Y[splits["val"]]
    print(f"train {Xtr.shape}  val {Xva.shape}")

    seeds = args.seeds if args.seeds is not None else cfg["training"]["seeds"]
    models_dir = path(cfg, "artifacts_dir") / "models"
    results_dir = path(cfg, "results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.time_test:
        timing = {}
        for name in args.models:
            t0 = time.time()
            train_model(name, cfg, Xtr, Ytr, Xva, Yva, seed=0,
                        out_path=models_dir / "timing" / f"{name}_timing.pt", max_epochs=1)
            timing[name] = time.time() - t0
        n_ep, n_seeds = cfg["training"]["max_epochs"], len(seeds)
        print("\nTime per epoch (incl. validation):")
        for k, v in timing.items():
            print(f"  {k:12s} {v:7.1f} s  -> worst case {v * n_ep * n_seeds / 3600:5.1f} h "
                  f"for {n_ep} epochs x {n_seeds} seeds (early stopping usually ends sooner)")
        print(f"  TOTAL worst case: {sum(timing.values()) * n_ep * n_seeds / 3600:.1f} h")
        (results_dir / "timing_test.json").write_text(json.dumps(timing, indent=2))
        return

    runs = []
    for seed in seeds:
        for name in args.models:
            out = models_dir / ("smoke" if args.smoke else "") / f"{name}_seed{seed}.pt"
            run = train_model(name, cfg, Xtr, Ytr, Xva, Yva, seed=seed, out_path=out,
                              max_epochs=2 if args.smoke else args.epochs,
                              max_batches=5 if args.smoke else None)
            runs.append({k: v for k, v in run.items() if k != "history"})
            print(f"==> {name} seed {seed}: best val macro-AUROC {run['best_val_macro_auroc']:.4f} "
                  f"(epoch {run['best_epoch']}, {run['train_time_s']:.0f}s, {run['n_params']:,} params)")

    log_path = results_dir / ("training_runs_smoke.json" if args.smoke else "training_runs.json")
    previous = json.loads(log_path.read_text()) if log_path.exists() else []
    keyed = {(r["model"], r["seed"]): r for r in previous}
    keyed.update({(r["model"], r["seed"]): r for r in runs})
    log_path.write_text(json.dumps(sorted(keyed.values(), key=lambda r: (r["model"], r["seed"])), indent=2))
    print(f"\nSaved run summaries to {log_path}")
    if not args.smoke:
        print("Next: python scripts/evaluate_all.py")
    else:
        print("Smoke test finished. Smoke checkpoints are in artifacts/models/smoke/ and are NOT used by the app.")


if __name__ == "__main__":
    main()
