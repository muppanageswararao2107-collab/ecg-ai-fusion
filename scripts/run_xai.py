"""Step 4 — explainability on the deployed models.

    python scripts/run_xai.py
    python scripts/run_xai.py --per-class 20      # faster

For every deployed model and every class it:
  * computes Integrated Gradients on test ECGs that ALL models correctly flagged for that class
    (the same ECGs for every model, so the comparison is fair);
  * runs the deletion test (mask top-10 % most-attributed time vs random time);
  * saves one example figure per class and model (plus Grad-CAM for the CNN).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecgai.config import load_config, path  # noqa: E402
from ecgai.data import labels_from_meta, load_cache, split_indices  # noqa: E402
from ecgai.inference import resolve_checkpoint  # noqa: E402
from ecgai.models import DISPLAY_NAMES  # noqa: E402
from ecgai.plotting import fig_to_png, plot_12lead  # noqa: E402
from ecgai.preprocess import STANDARD_LEADS, load_normalization  # noqa: E402
from ecgai.train import get_device, load_checkpoint, predict_proba, standardize_split  # noqa: E402
from ecgai.xai import deletion_test, grad_cam_1d, integrated_gradients, summarize_deletion  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-class", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    xcfg = cfg["xai"]
    n_per = args.per_class or int(xcfg["deletion_samples_per_class"])
    classes = cfg["data"]["classes"]
    fs = cfg["data"]["sampling_rate"]
    out = path(cfg, "results_dir") / "xai"
    out.mkdir(parents=True, exist_ok=True)

    dep_file = path(cfg, "artifacts_dir") / "deployment.json"
    if not dep_file.exists():
        sys.exit("deployment.json not found. Run scripts/evaluate_all.py first.")
    dep = json.loads(dep_file.read_text())
    device = get_device(cfg["training"].get("device", "auto"))

    X, meta = load_cache(cfg)
    test_idx = split_indices(meta["strat_fold"].to_numpy(), cfg["data"])["test"]
    norm = load_normalization(path(cfg, "artifacts_dir") / "norm.json")
    Xte = standardize_split(X, test_idx, norm["mean"], norm["std"])
    Yte = labels_from_meta(meta, classes)[test_idx]

    models, flags = {}, {}
    for name, info in dep["models"].items():
        model, _ = load_checkpoint(resolve_checkpoint(info["checkpoint"]), device)
        models[name] = model
        thr = np.array([info["thresholds"][c] for c in classes])
        flags[name] = predict_proba(model, Xte, device) >= thr

    rng = np.random.default_rng(0)
    all_rows, summary = [], {}
    for j, c in enumerate(classes):
        common = (Yte[:, j] == 1) & np.all([flags[m][:, j] for m in models], axis=0)
        cand = np.where(common)[0]
        if len(cand) == 0:
            print(f"[{c}] no test ECG was flagged correctly by every model; skipped.")
            continue
        chosen = rng.choice(cand, size=min(n_per, len(cand)), replace=False)
        print(f"[{c}] {len(cand)} ECGs flagged correctly by all models; using {len(chosen)}.")
        for name, model in models.items():
            res = []
            for k, i in enumerate(chosen):
                ig = integrated_gradients(model, Xte[i], j, steps=int(xcfg["ig_steps"]))
                d = deletion_test(model, Xte[i], j, ig["attr"], fraction=float(xcfg["deletion_fraction"]),
                                  segment=int(xcfg["deletion_segment"]), random_repeats=int(xcfg["random_repeats"]),
                                  rng=np.random.default_rng(int(i)))
                d.update(model=name, cls=c, ecg_id=int(meta.iloc[test_idx[i]]["ecg_id"]),
                         ig_completeness_error=ig["delta"])
                res.append(d)
                if k == 0:
                    raw = np.asarray(X[test_idx[i]], dtype=np.float32)
                    png = fig_to_png(plot_12lead(raw, fs, STANDARD_LEADS, attribution=ig["attr"],
                                                 title=f"{DISPLAY_NAMES[name]} — Integrated Gradients — {c} "
                                                       f"(PTB-XL ecg_id {d['ecg_id']})"))
                    (out / f"ig_{name}_{c}.png").write_bytes(png)
                    if name == "cnn":
                        cam = grad_cam_1d(model, Xte[i], j)
                        png = fig_to_png(plot_12lead(raw, fs, STANDARD_LEADS, attribution=np.repeat(cam[:, None], 12, 1),
                                                     title=f"CNN — Grad-CAM — {c} (PTB-XL ecg_id {d['ecg_id']})"))
                        (out / f"gradcam_cnn_{c}.png").write_bytes(png)
            all_rows += res
            summary.setdefault(name, {})[c] = summarize_deletion(res)
            s = summary[name][c]
            print(f"   {name:12s} drop(top) {s['mean_drop_top']:.3f} vs drop(random) {s['mean_drop_random']:.3f}"
                  f"  -> {s['verdict']}")

    for name in summary:
        rows = [r for r in all_rows if r["model"] == name]
        summary[name]["ALL"] = summarize_deletion(rows)
    pd.DataFrame(all_rows).to_csv(out / "deletion_test_samples.csv", index=False)
    (out / "deletion_test_summary.json").write_text(json.dumps(summary, indent=2))
    table = [{"model": m, "class": c, **{k: v for k, v in s.items()}} for m, d in summary.items() for c, s in d.items()]
    pd.DataFrame(table).to_csv(out / "deletion_test_summary.csv", index=False)
    print(f"\nSaved XAI figures and deletion-test results to {out}")


if __name__ == "__main__":
    main()
