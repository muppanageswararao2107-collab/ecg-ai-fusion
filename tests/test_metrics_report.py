import numpy as np

from ecgai.metrics import (apply_thresholds, bootstrap_macro_auroc_ci, full_evaluation, macro_auroc, simplex_grid,
                           tune_ensemble_weights, tune_thresholds)
from ecgai.plotting import fig_to_png, plot_12lead
from ecgai.preprocess import STANDARD_LEADS
from ecgai.report import build_pdf
from helpers import synthetic_ecg

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def _toy(n=400, seed=0, signal=2.0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 5)) < 0.3).astype(int)
    y[:5] = np.eye(5, dtype=int)
    logits = signal * (y - 0.5) + rng.standard_normal((n, 5))
    return y, 1 / (1 + np.exp(-logits))


def test_perfect_and_random_auroc():
    y, _ = _toy()
    assert macro_auroc(y, y.astype(float)) == 1.0
    rng = np.random.default_rng(3)
    assert 0.4 < macro_auroc(y, rng.random(y.shape)) < 0.6


def test_thresholds_are_in_grid_and_applied():
    y, p = _toy()
    thr = tune_thresholds(y, p)
    assert thr.shape == (5,) and np.all((thr > 0) & (thr < 1))
    assert apply_thresholds(p, thr).shape == p.shape


def test_bootstrap_ci_contains_point_estimate():
    y, p = _toy()
    lo, hi = bootstrap_macro_auroc_ci(y, p, n=200)
    assert lo <= macro_auroc(y, p) <= hi


def test_full_evaluation_uses_validation_thresholds():
    yv, pv = _toy(seed=1)
    yt, pt = _toy(seed=2)
    ev = full_evaluation(yv, pv, yt, pt, CLASSES, n_boot=50)
    assert list(ev["thresholds_from_val"].values()) == tune_thresholds(yv, pv).tolist()
    assert set(ev["test_confusion"]["MI"]) == {"tn", "fp", "fn", "tp"}
    assert 0.5 < ev["test_macro_auroc"] <= 1.0


def test_ensemble_weights_on_simplex_and_prefer_better_model():
    y, good = _toy(seed=4, signal=3.0)
    rng = np.random.default_rng(5)
    bad = rng.random(y.shape)
    w, score = tune_ensemble_weights(y, [good, bad, bad], step=0.25)
    assert abs(w.sum() - 1) < 1e-9 and w[0] >= 0.5
    assert all(abs(p.sum() - 1) < 1e-9 for p in simplex_grid(3, 0.1))
    assert len(simplex_grid(3, 0.1)) == 66


def test_pdf_report_is_generated():
    x = synthetic_ecg(10, 100)
    wave = fig_to_png(plot_12lead(x, 100, STANDARD_LEADS, title="unit test"))
    attr = np.abs(np.random.default_rng(0).standard_normal(x.shape))
    xai_png = fig_to_png(plot_12lead(x, 100, STANDARD_LEADS, attribution=attr))
    preds = {m: {"probs": {c: 0.1 * (i + 1) for i, c in enumerate(CLASSES)},
                 "flags": {c: int(i > 2) for i, c in enumerate(CLASSES)}, "thresholds": {c: 0.5 for c in CLASSES}}
             for m in ["cnn", "lstm", "transformer", "ensemble", "fusion"]}
    preds["fusion"]["branch_weights"] = {"cnn": 0.5, "lstm": 0.2, "transformer": 0.3}
    preds["ensemble"]["weights"] = {"cnn": 0.4, "lstm": 0.3, "transformer": 0.3}
    pdf = build_pdf({
        "report_id": "ECG-TEST", "version": "1.0.0",
        "input": {"filename": "unit-test.csv", "source": "Uploaded CSV", "fs": 100.0, "n_leads": 12,
                  "duration_s": 10.0, "ecg_id": None, "true_labels": None, "notes": ["note"]},
        "quality": [{"name": "Missing values", "status": "PASS", "message": "ok"}], "quality_status": "PASS",
        "heart_rate": {"available": False, "bpm": None, "reason": "unit test", "lead": "II"},
        "classes": CLASSES, "model_names": {m: m for m in preds}, "predictions": preds,
        "performance": {}, "waveform_png": wave,
        "xai": {"png": xai_png, "model": "fusion", "class": "MI", "method": "Integrated Gradients"},
        "n_windows": 1})
    assert pdf[:4] == b"%PDF" and len(pdf) > 10_000
