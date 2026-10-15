"""Step 3 — evaluate every trained model, build the late-fusion ensemble, write the comparison
tables/figures and the deployment file used by the website.

    python scripts/evaluate_all.py

Protocol (no test-set leakage):
  * decision thresholds and ensemble weights are tuned on the validation fold (9) only;
  * the deployed seed of each model is chosen by VALIDATION macro-AUROC;
  * the test fold (10) is only used to report the final numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from ecgai.config import ROOT, load_config, path  # noqa: E402
from ecgai.data import labels_from_meta, load_cache, split_indices  # noqa: E402
from ecgai.metrics import combine, full_evaluation, tune_ensemble_weights  # noqa: E402
from ecgai.models import DISPLAY_NAMES, MODEL_NAMES  # noqa: E402
from ecgai.preprocess import load_normalization  # noqa: E402
from ecgai.train import get_device, load_checkpoint, predict_proba, standardize_split  # noqa: E402

STANDALONE = ["cnn", "lstm", "transformer"]
ORDER = ["cnn", "lstm", "transformer", "ensemble", "fusion"]
COLORS = {"cnn": "#0f766e", "lstm": "#2563eb", "transformer": "#7c3aed", "ensemble": "#b45309", "fusion": "#be123c"}


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


@torch.no_grad()
def inference_ms(model, X: np.ndarray, device, n: int = 100) -> float:
    model.eval()
    xs = torch.from_numpy(np.ascontiguousarray(X[:n])).to(device)
    model(xs[:1])  # warm-up
    t0 = time.perf_counter()
    for i in range(len(xs)):
        model(xs[i:i + 1])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / len(xs) * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    classes = cfg["data"]["classes"]
    n_boot = int(cfg["evaluation"]["bootstrap_samples"])
    res = path(cfg, "results_dir")
    (res / "preds").mkdir(parents=True, exist_ok=True)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    device = get_device(cfg["training"].get("device", "auto"))

    X, meta = load_cache(cfg)
    splits = split_indices(meta["strat_fold"].to_numpy(), cfg["data"])
    norm = load_normalization(path(cfg, "artifacts_dir") / "norm.json")
    Xva = standardize_split(X, splits["val"], norm["mean"], norm["std"])
    Xte = standardize_split(X, splits["test"], norm["mean"], norm["std"])
    Y = labels_from_meta(meta, classes)
    Yva, Yte = Y[splits["val"]], Y[splits["test"]]

    ckpts = sorted((path(cfg, "artifacts_dir") / "models").glob("*_seed*.pt"))
    if not ckpts:
        sys.exit("No checkpoints found in artifacts/models/. Run scripts/train_all.py first.")

    # ---- predictions + metrics for every (model, seed)
    evals, preds, run_meta = {}, {}, {}
    for ck in ckpts:
        m = re.match(r"(\w+?)_seed(\d+)\.pt$", ck.name)
        if not m or m.group(1) not in MODEL_NAMES:
            continue
        name, seed = m.group(1), int(m.group(2))
        model, meta_ck = load_checkpoint(ck, device)
        if name == "fusion":
            p_va = predict_proba(model, Xva, device)
            p_te, alpha = predict_proba(model, Xte, device, return_alpha=True)
        else:
            p_va, p_te, alpha = predict_proba(model, Xva, device), predict_proba(model, Xte, device), None
        np.savez_compressed(res / "preds" / f"{name}_seed{seed}.npz", p_val=p_va, p_test=p_te,
                            alpha_test=alpha if alpha is not None else np.zeros(0))
        ev = full_evaluation(Yva, p_va, Yte, p_te, classes, n_boot=n_boot)
        ev["n_params"] = meta_ck["n_params"]
        ev["inference_ms_per_ecg"] = round(inference_ms(model, Xte, device), 2)
        run_json = ck.with_suffix(".json")
        if run_json.exists():
            rj = json.loads(run_json.read_text())
            ev["train_time_s"], ev["best_epoch"] = rj.get("train_time_s"), rj.get("best_epoch")
            run_meta[(name, seed)] = rj
        evals[(name, seed)] = ev
        preds[(name, seed)] = {"val": p_va, "test": p_te, "alpha": alpha, "ckpt": ck}
        print(f"{name:12s} seed {seed}: val {ev['val_macro_auroc']:.4f}  test {ev['test_macro_auroc']:.4f}  "
              f"F1 {ev['test_macro_f1']:.4f}")

    # ---- late-fusion ensemble for every seed with all three standalone models
    seeds = sorted({s for (_, s) in preds})
    for s in seeds:
        if all((m, s) in preds for m in STANDALONE):
            pv = [preds[(m, s)]["val"] for m in STANDALONE]
            pt = [preds[(m, s)]["test"] for m in STANDALONE]
            w, _ = tune_ensemble_weights(Yva, pv, cfg["evaluation"]["ensemble_weight_step"])
            ev = full_evaluation(Yva, combine(pv, w), Yte, combine(pt, w), classes, n_boot=n_boot)
            ev["weights"] = {m: float(x) for m, x in zip(STANDALONE, w)}
            ev["n_params"] = int(sum(evals[(m, s)]["n_params"] for m in STANDALONE))
            evals[("ensemble", s)] = ev
            preds[("ensemble", s)] = {"val": combine(pv, w), "test": combine(pt, w)}
            print(f"ensemble     seed {s}: weights {ev['weights']}  test {ev['test_macro_auroc']:.4f}")

    # ---- aggregate across seeds
    rows, per_class_rows = [], []
    for m in ORDER:
        ks = sorted(k for k in evals if k[0] == m)
        if not ks:
            continue
        tv = np.array([evals[k]["test_macro_auroc"] for k in ks])
        tf = np.array([evals[k]["test_macro_f1"] for k in ks])
        vv = np.array([evals[k]["val_macro_auroc"] for k in ks])
        rows.append({
            "model": m, "display_name": DISPLAY_NAMES[m], "n_seeds": len(ks),
            "test_macro_auroc_mean": tv.mean(), "test_macro_auroc_std": tv.std(ddof=1) if len(ks) > 1 else np.nan,
            "test_macro_f1_mean": tf.mean(), "test_macro_f1_std": tf.std(ddof=1) if len(ks) > 1 else np.nan,
            "val_macro_auroc_mean": vv.mean(),
            "n_params": evals[ks[0]]["n_params"],
            "train_time_s_mean": np.mean([evals[k].get("train_time_s") or np.nan for k in ks]),
            "inference_ms_per_ecg": np.mean([evals[k].get("inference_ms_per_ecg", np.nan) for k in ks]),
        })
        for c in classes:
            per_class_rows.append({"model": m, "class": c,
                                   "test_auroc_mean": np.mean([evals[k]["test_auroc_per_class"][c] for k in ks]),
                                   "test_f1_mean": np.mean([evals[k]["test_per_class"][c]["f1"] for k in ks]),
                                   "test_precision_mean": np.mean([evals[k]["test_per_class"][c]["precision"] for k in ks]),
                                   "test_recall_mean": np.mean([evals[k]["test_per_class"][c]["recall"] for k in ks])})
    comp = pd.DataFrame(rows)
    comp.to_csv(res / "comparison.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(res / "per_class.csv", index=False)
    with open(res / "comparison.md", "w") as f:
        f.write("| Model | Seeds | Test macro-AUROC | Test macro-F1 | Val macro-AUROC | Params | ms/ECG |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            sd = f" ± {r['test_macro_auroc_std']:.4f}" if np.isfinite(r["test_macro_auroc_std"]) else ""
            sdf = f" ± {r['test_macro_f1_std']:.4f}" if np.isfinite(r["test_macro_f1_std"]) else ""
            f.write(f"| {r['display_name']} | {r['n_seeds']} | {r['test_macro_auroc_mean']:.4f}{sd} | "
                    f"{r['test_macro_f1_mean']:.4f}{sdf} | {r['val_macro_auroc_mean']:.4f} | {r['n_params']:,} | "
                    f"{r['inference_ms_per_ecg']:.1f} |\n")
        f.write("\nReference only: Strodthoff et al. (2021, IEEE JBHI) report macro-AUROC around 0.93 on the same "
                "superdiagnostic task. Those are their numbers, not ours.\n")
    with open(res / "metrics_all.json", "w") as f:
        json.dump({f"{k[0]}_seed{k[1]}": v for k, v in evals.items()}, f, indent=2, default=float)

    # ---- choose deployed models by VALIDATION macro-AUROC
    deployment = {"classes": classes, "selection": "best validation macro-AUROC per model type",
                  "thresholds_tuned_on": "validation fold 9", "models": {}}
    for m in MODEL_NAMES:
        ks = [k for k in evals if k[0] == m]
        if not ks:
            continue
        best = max(ks, key=lambda k: evals[k]["val_macro_auroc"])
        ev = evals[best]
        deployment["models"][m] = {"checkpoint": rel(preds[best]["ckpt"]), "seed": best[1],
                                   "thresholds": ev["thresholds_from_val"], "val_macro_auroc": ev["val_macro_auroc"],
                                   "test_macro_auroc": ev["test_macro_auroc"],
                                   "test_macro_auroc_ci95": ev["test_macro_auroc_ci95"],
                                   "test_macro_f1": ev["test_macro_f1"], "n_params": ev["n_params"],
                                   "display_name": DISPLAY_NAMES[m]}
    if all(m in deployment["models"] for m in STANDALONE):
        sel = {m: (m, deployment["models"][m]["seed"]) for m in STANDALONE}
        pv = [preds[sel[m]]["val"] for m in STANDALONE]
        pt = [preds[sel[m]]["test"] for m in STANDALONE]
        w, _ = tune_ensemble_weights(Yva, pv, cfg["evaluation"]["ensemble_weight_step"])
        ev = full_evaluation(Yva, combine(pv, w), Yte, combine(pt, w), classes, n_boot=n_boot)
        deployment["ensemble"] = {"members": STANDALONE, "weights": [float(x) for x in w],
                                  "thresholds": ev["thresholds_from_val"], "val_macro_auroc": ev["val_macro_auroc"],
                                  "test_macro_auroc": ev["test_macro_auroc"], "test_macro_f1": ev["test_macro_f1"],
                                  "display_name": DISPLAY_NAMES["ensemble"]}
    dep_path = path(cfg, "artifacts_dir") / "deployment.json"
    dep_path.write_text(json.dumps(deployment, indent=2))

    # ---- figures
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=140)
    for i, r in enumerate(rows):
        err = r["test_macro_auroc_std"] if np.isfinite(r["test_macro_auroc_std"]) else 0
        ax.bar(i, r["test_macro_auroc_mean"], yerr=err, color=COLORS[r["model"]], capsize=4)
        ax.text(i, r["test_macro_auroc_mean"] + err + 0.003, f"{r['test_macro_auroc_mean']:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(rows)), [DISPLAY_NAMES[r["model"]].split(" (")[0] for r in rows], fontsize=8)
    lo = min(r["test_macro_auroc_mean"] for r in rows)
    ax.set_ylim(max(0.5, lo - 0.05), 1.0)
    ax.set_ylabel("Test macro-AUROC (mean ± SD over seeds)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(res / "figures" / "auroc_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(classes), figsize=(3.0 * len(classes), 3.2), dpi=130, sharey=True)
    dep_keys = {m: (m, d["seed"]) for m, d in deployment["models"].items()}
    if "ensemble" in deployment:
        dep_keys["ensemble"] = ("ensemble", None)
    for j, c in enumerate(classes):
        ax = axes[j]
        for m in ORDER:
            if m not in dep_keys:
                continue
            if m == "ensemble":
                p = combine([preds[(s, deployment['models'][s]['seed'])]["test"] for s in STANDALONE],
                            np.array(deployment["ensemble"]["weights"]))
            else:
                p = preds[dep_keys[m]]["test"]
            fpr, tpr, _ = roc_curve(Yte[:, j], p[:, j])
            ax.plot(fpr, tpr, lw=1.2, color=COLORS[m], label=DISPLAY_NAMES[m].split(" (")[0])
        ax.plot([0, 1], [0, 1], ls="--", color="#a1a1aa", lw=0.8)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel("False positive rate", fontsize=8)
        if j == 0:
            ax.set_ylabel("True positive rate", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(res / "figures" / "roc_curves.png")
    plt.close(fig)

    for m, info in deployment["models"].items():
        ev = evals[(m, info["seed"])]
        fig, axes = plt.subplots(1, len(classes), figsize=(2.3 * len(classes), 2.4), dpi=130)
        for j, c in enumerate(classes):
            cm = ev["test_confusion"][c]
            mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
            axes[j].imshow(mat, cmap="Blues")
            for a in range(2):
                for b in range(2):
                    axes[j].text(b, a, mat[a, b], ha="center", va="center", fontsize=8,
                                 color="white" if mat[a, b] > mat.max() / 2 else "black")
            axes[j].set_xticks([0, 1], ["pred 0", "pred 1"], fontsize=7)
            axes[j].set_yticks([0, 1], ["true 0", "true 1"], fontsize=7)
            axes[j].set_title(c, fontsize=9)
        fig.suptitle(f"{DISPLAY_NAMES[m]} — test fold confusion matrices (val-tuned thresholds)", fontsize=9)
        fig.tight_layout()
        fig.savefig(res / "figures" / f"confusion_{m}.png")
        plt.close(fig)

    if run_meta:
        fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=140)
        for (m, s), rj in sorted(run_meta.items()):
            h = rj.get("history", [])
            ax.plot([e["epoch"] for e in h], [e["val_macro_auroc"] for e in h], color=COLORS[m], alpha=0.8,
                    lw=1, label=DISPLAY_NAMES[m].split(" (")[0] if s == min(x[1] for x in run_meta if x[0] == m) else None)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation macro-AUROC")
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(res / "figures" / "training_curves.png")
        plt.close(fig)

    if "fusion" in deployment["models"]:
        alpha = preds[("fusion", deployment["models"]["fusion"]["seed"])]["alpha"]
        if alpha is not None and len(alpha):
            rows_a = []
            for j, c in enumerate(classes):
                sel = Yte[:, j] == 1
                rows_a.append({"true_class": c, "n": int(sel.sum()),
                               **{b: float(alpha[sel, i].mean()) for i, b in enumerate(STANDALONE)}})
            rows_a.append({"true_class": "ALL", "n": int(len(alpha)),
                           **{b: float(alpha[:, i].mean()) for i, b in enumerate(STANDALONE)}})
            pd.DataFrame(rows_a).to_csv(res / "fusion_branch_weights.csv", index=False)
            fig, ax = plt.subplots(figsize=(7, 3.2), dpi=140)
            dfa = pd.DataFrame(rows_a).set_index("true_class")[STANDALONE]
            bottom = np.zeros(len(dfa))
            for b in STANDALONE:
                ax.bar(dfa.index, dfa[b], bottom=bottom, color=COLORS[b], label=b.upper())
                bottom += dfa[b].to_numpy()
            ax.set_ylabel("Mean gate weight (test fold)")
            ax.legend(fontsize=7, frameon=False, ncol=3)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            fig.savefig(res / "figures" / "fusion_branch_weights.png")
            plt.close(fig)

    print("\n" + open(res / "comparison.md").read())
    print(f"Deployment file: {dep_path}")
    print("Next: python scripts/run_xai.py   (then: streamlit run app.py)")


if __name__ == "__main__":
    main()
