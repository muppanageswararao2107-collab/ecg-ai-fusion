"""Loading the deployed models and running them in parallel on one ECG."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import ROOT, path as cfg_path
from .metrics import apply_thresholds, combine
from .preprocess import load_normalization

STANDALONE = ["cnn", "lstm", "transformer"]


class ModelsNotTrained(RuntimeError):
    """Raised when trained weights or the deployment file are missing."""


def resolve_checkpoint(p: str) -> Path:
    """Checkpoint paths are stored relative to the project root so the folder can be moved
    (e.g. trained on Colab, used on a Mac)."""
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def deployment_path(cfg: dict) -> Path:
    return cfg_path(cfg, "artifacts_dir") / "deployment.json"


def check_ready(cfg: dict) -> tuple[bool, str]:
    norm = cfg_path(cfg, "artifacts_dir") / "norm.json"
    dep = deployment_path(cfg)
    if not norm.exists():
        return False, "Model not trained yet: normalization statistics are missing (run scripts/prepare_data.py)."
    if not dep.exists():
        return False, ("Model not trained yet: no deployment file. Train and evaluate first "
                       "(scripts/train_all.py, then scripts/evaluate_all.py).")
    try:
        torch_ok = __import__("torch") is not None
    except ImportError:
        torch_ok = False
    if not torch_ok:
        return False, "PyTorch is not installed, so the models cannot run."
    info = json.loads(dep.read_text(encoding="utf-8"))
    missing = [m["checkpoint"] for m in info.get("models", {}).values()
               if not resolve_checkpoint(m["checkpoint"]).exists()]
    if missing:
        return False, f"Model not trained yet: checkpoint file(s) missing: {', '.join(missing)}"
    return True, "Models loaded."


class ModelBundle:
    """Every deployed model + its validation-tuned thresholds + the late-fusion ensemble."""

    def __init__(self, cfg: dict, device: str = "cpu"):
        import torch  # noqa: F401  (imported lazily so the rest of the app works without torch)

        from .train import load_checkpoint

        ok, msg = check_ready(cfg)
        if not ok:
            raise ModelsNotTrained(msg)
        self.cfg = cfg
        self.norm = load_normalization(cfg_path(cfg, "artifacts_dir") / "norm.json")
        self.deployment = json.loads(deployment_path(cfg).read_text(encoding="utf-8"))
        self.classes = self.deployment["classes"]
        self.models, self.meta = {}, {}
        for name, info in self.deployment["models"].items():
            model, meta = load_checkpoint(resolve_checkpoint(info["checkpoint"]), device)
            if meta["classes"] != self.classes:
                raise ModelsNotTrained(f"Checkpoint {info['checkpoint']} has different classes.")
            self.models[name], self.meta[name] = model, {**meta, **info}
        self.ensemble = self.deployment.get("ensemble")

    def predict(self, windows_norm: np.ndarray) -> dict:
        """windows_norm: [W, T, C] standardized. Runs all models concurrently (one thread each)."""
        from .train import predict_proba

        def run(name):
            model = self.models[name]
            dev = next(model.parameters()).device
            if name == "fusion":
                p, alpha = predict_proba(model, windows_norm, dev, return_alpha=True)
                return name, p, alpha
            return name, predict_proba(model, windows_norm, dev), None

        with ThreadPoolExecutor(max_workers=len(self.models)) as pool:
            outputs = list(pool.map(run, list(self.models)))

        results, per_window = {}, {}
        for name, p_win, alpha in outputs:
            thr = np.array([self.meta[name]["thresholds"][c] for c in self.classes])
            p = p_win.mean(axis=0)
            results[name] = {"probs": {c: float(p[j]) for j, c in enumerate(self.classes)},
                             "flags": {c: int(v) for c, v in zip(self.classes, apply_thresholds(p[None], thr)[0])},
                             "thresholds": {c: float(t) for c, t in zip(self.classes, thr)}}
            if alpha is not None:
                a = alpha.mean(axis=0)
                results[name]["branch_weights"] = {b: float(a[i]) for i, b in enumerate(["cnn", "lstm", "transformer"])}
            per_window[name] = p_win

        if self.ensemble and all(m in per_window for m in self.ensemble["members"]):
            w = np.array(self.ensemble["weights"])
            p_win = combine([per_window[m] for m in self.ensemble["members"]], w)
            p = p_win.mean(axis=0)
            thr = np.array([self.ensemble["thresholds"][c] for c in self.classes])
            results["ensemble"] = {
                "probs": {c: float(p[j]) for j, c in enumerate(self.classes)},
                "flags": {c: int(v) for c, v in zip(self.classes, apply_thresholds(p[None], thr)[0])},
                "thresholds": {c: float(t) for c, t in zip(self.classes, thr)},
                "weights": {m: float(x) for m, x in zip(self.ensemble["members"], w)}}
            per_window["ensemble"] = p_win

        return {"models": results,
                "per_window": {k: v.tolist() for k, v in per_window.items()},
                "n_windows": int(windows_norm.shape[0]),
                "classes": self.classes}

    def performance(self) -> dict:
        """Test metrics recorded at evaluation time (read from deployment.json, never recomputed here)."""
        out = {name: {"test_macro_auroc": info.get("test_macro_auroc"), "val_macro_auroc": info.get("val_macro_auroc"),
                      "seed": info.get("seed"), "n_params": info.get("n_params")}
               for name, info in self.deployment["models"].items()}
        if self.ensemble:
            out["ensemble"] = {"test_macro_auroc": self.ensemble.get("test_macro_auroc"),
                               "val_macro_auroc": self.ensemble.get("val_macro_auroc")}
        return out
