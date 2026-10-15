"""Evaluation metrics for multi-label ECG classification (NumPy / scikit-learn only)."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, roc_auc_score

THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        if 0 < y[:, j].sum() < len(y):
            out[j] = roc_auc_score(y[:, j], p[:, j])
    return out


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    vals = per_class_auroc(y, p)
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def _f1(y: np.ndarray, yhat: np.ndarray) -> float:
    tp = float((y * yhat).sum())
    fp = float(((1 - y) * yhat).sum())
    fn = float((y * (1 - yhat)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def tune_thresholds(y_val: np.ndarray, p_val: np.ndarray, grid: np.ndarray = THRESHOLD_GRID) -> np.ndarray:
    """Per-class threshold that maximizes F1 on the VALIDATION set."""
    thr = np.full(y_val.shape[1], 0.5)
    for j in range(y_val.shape[1]):
        scores = [_f1(y_val[:, j], (p_val[:, j] >= t).astype(float)) for t in grid]
        thr[j] = float(grid[int(np.argmax(scores))])
    return thr


def apply_thresholds(p: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (p >= np.asarray(thr)[None, :]).astype(np.int64)


def classification_table(y: np.ndarray, yhat: np.ndarray, classes: list[str]) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(y, yhat, average=None, zero_division=0)
    per = {c: {"precision": float(prec[j]), "recall": float(rec[j]), "f1": float(f1[j]),
               "support": int(sup[j])} for j, c in enumerate(classes)}
    return {"per_class": per, "macro_f1": float(np.mean(f1)),
            "macro_precision": float(np.mean(prec)), "macro_recall": float(np.mean(rec))}


def confusion_matrices(y: np.ndarray, yhat: np.ndarray, classes: list[str]) -> dict:
    mcm = multilabel_confusion_matrix(y, yhat)
    return {c: {"tn": int(m[0, 0]), "fp": int(m[0, 1]), "fn": int(m[1, 0]), "tp": int(m[1, 1])}
            for c, m in zip(classes, mcm)}


def bootstrap_macro_auroc_ci(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 0,
                             alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if np.any(yb.sum(axis=0) == 0) or np.any(yb.sum(axis=0) == len(yb)):
            continue
        vals.append(macro_auroc(yb, p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def full_evaluation(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray,
                    classes: list[str], n_boot: int = 1000, thresholds: np.ndarray | None = None) -> dict:
    """Thresholds are tuned on validation only, then applied once to the test set."""
    thr = tune_thresholds(y_val, p_val) if thresholds is None else np.asarray(thresholds)
    yhat = apply_thresholds(p_test, thr)
    auc = per_class_auroc(y_test, p_test)
    lo, hi = bootstrap_macro_auroc_ci(y_test, p_test, n=n_boot)
    table = classification_table(y_test, yhat, classes)
    return {
        "val_macro_auroc": macro_auroc(y_val, p_val),
        "test_macro_auroc": float(np.nanmean(auc)),
        "test_macro_auroc_ci95": [lo, hi],
        "test_auroc_per_class": {c: float(auc[j]) for j, c in enumerate(classes)},
        "thresholds_from_val": {c: float(thr[j]) for j, c in enumerate(classes)},
        "test_macro_f1": table["macro_f1"],
        "test_macro_precision": table["macro_precision"],
        "test_macro_recall": table["macro_recall"],
        "test_per_class": table["per_class"],
        "test_confusion": confusion_matrices(y_test, yhat, classes),
        "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    }


def simplex_grid(n_models: int, step: float) -> list[np.ndarray]:
    k = int(round(1 / step))
    pts = []
    for combo in itertools.product(range(k + 1), repeat=n_models - 1):
        if sum(combo) <= k:
            w = np.array(list(combo) + [k - sum(combo)], dtype=float) / k
            pts.append(w)
    return pts


def tune_ensemble_weights(y_val: np.ndarray, probs_val: list[np.ndarray], step: float = 0.1
                          ) -> tuple[np.ndarray, float]:
    """Late fusion: weighted average of model probabilities; weights chosen on VALIDATION macro-AUROC."""
    best_w, best = None, -np.inf
    for w in simplex_grid(len(probs_val), step):
        p = sum(wi * pi for wi, pi in zip(w, probs_val))
        score = macro_auroc(y_val, p)
        if score > best + 1e-12:
            best, best_w = score, w
    return best_w, float(best)


def combine(probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(float(w) * p for w, p in zip(weights, probs))
