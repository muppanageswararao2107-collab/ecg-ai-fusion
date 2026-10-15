"""Explainability: Integrated Gradients (all models), 1D Grad-CAM (CNN) and a deletion test
that checks whether the explanations are better than random."""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn

from .models import CNN1D


def _no_cudnn():
    # cuDNN refuses RNN backward passes in eval mode; disabling it is harmless here.
    if torch.cuda.is_available():
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def integrated_gradients(model: nn.Module, x: np.ndarray, target: int, steps: int = 50,
                         baseline: np.ndarray | None = None, batch: int = 25) -> dict:
    """Integrated Gradients for one standardized ECG x [T, C] and one class.

    Baseline: all zeros, i.e. the per-lead TRAINING mean after standardization.
    Returns {"attr": [T, C], "delta": completeness error, "logit": f(x), "baseline_logit": f(b)}."""
    model.eval()
    dev = model_device(model)
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev)
    bt = torch.zeros_like(xt) if baseline is None else torch.from_numpy(np.asarray(baseline, np.float32)).to(dev)
    alphas = (torch.arange(steps, dtype=torch.float32, device=dev) + 0.5) / steps
    total = torch.zeros_like(xt)
    with _no_cudnn():
        for s in range(0, steps, batch):
            a = alphas[s:s + batch].view(-1, 1, 1)
            xs = (bt.unsqueeze(0) + a * (xt - bt).unsqueeze(0)).detach().requires_grad_(True)
            out = model(xs)[:, target]
            grads = torch.autograd.grad(out.sum(), xs)[0]
            total += grads.sum(dim=0)
        with torch.no_grad():
            f_x = float(model(xt.unsqueeze(0))[0, target])
            f_b = float(model(bt.unsqueeze(0))[0, target])
    attr = ((xt - bt) * total / steps).detach().cpu().numpy()
    return {"attr": attr, "delta": float(attr.sum() - (f_x - f_b)), "logit": f_x, "baseline_logit": f_b}


def grad_cam_1d(model: CNN1D, x: np.ndarray, target: int) -> np.ndarray:
    """Grad-CAM on the last residual block of the CNN. Returns [T] in [0, 1]."""
    if not isinstance(model, CNN1D):
        raise TypeError("Grad-CAM is implemented for the CNN model only.")
    model.eval()
    dev = model_device(model)
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(dev).unsqueeze(0)
    fmap = model.feature_map(xt)                       # [1, C, T']
    fmap.retain_grad()
    logit = model.head(model.dropout(fmap.mean(dim=-1)))[0, target]
    model.zero_grad(set_to_none=True)
    logit.backward()
    weights = fmap.grad.mean(dim=-1, keepdim=True)     # [1, C, 1]
    cam = torch.relu((weights * fmap).sum(dim=1))[0].detach().cpu().numpy()
    t_in = x.shape[0]
    cam_full = np.interp(np.linspace(0, len(cam) - 1, t_in), np.arange(len(cam)), cam)
    return cam_full / cam_full.max() if cam_full.max() > 0 else cam_full


def time_importance(attr: np.ndarray, segment: int) -> np.ndarray:
    """Sum |attribution| over leads and inside fixed-length time segments -> [n_segments]."""
    imp = np.abs(attr).sum(axis=1)
    n_seg = len(imp) // segment
    return imp[: n_seg * segment].reshape(n_seg, segment).sum(axis=1)


@torch.no_grad()
def _prob(model: nn.Module, xs: np.ndarray, target: int) -> np.ndarray:
    dev = model_device(model)
    out = model(torch.from_numpy(np.asarray(xs, dtype=np.float32)).to(dev))[:, target]
    return torch.sigmoid(out).cpu().numpy()


def deletion_test(model: nn.Module, x: np.ndarray, target: int, attr: np.ndarray, fraction: float = 0.1,
                  segment: int = 10, random_repeats: int = 5, rng: np.random.Generator | None = None) -> dict:
    """Mask the most-attributed `fraction` of time (all leads, set to the training mean = 0)
    and compare the probability drop with masking random segments of the same total size."""
    rng = rng or np.random.default_rng(0)
    seg_imp = time_importance(attr, segment)
    n_seg = len(seg_imp)
    k = max(1, int(round(fraction * n_seg)))

    def masked(segs):
        xm = np.array(x, dtype=np.float32, copy=True)
        for s in segs:
            xm[s * segment:(s + 1) * segment] = 0.0
        return xm

    top = np.argsort(seg_imp)[::-1][:k]
    batch = [x, masked(top)] + [masked(rng.choice(n_seg, size=k, replace=False)) for _ in range(random_repeats)]
    p = _prob(model, np.stack(batch), target)
    return {"p_original": float(p[0]), "drop_top": float(p[0] - p[1]),
            "drop_random": float(np.mean(p[0] - p[2:])), "k_segments": int(k), "n_segments": int(n_seg)}


def summarize_deletion(results: list[dict]) -> dict:
    if not results:
        return {"n": 0}
    top = np.array([r["drop_top"] for r in results])
    rnd = np.array([r["drop_random"] for r in results])
    summary = {"n": int(len(results)), "mean_drop_top": float(top.mean()), "mean_drop_random": float(rnd.mean()),
               "fraction_top_beats_random": float((top > rnd).mean())}
    try:
        from scipy.stats import wilcoxon
        if np.any(top - rnd != 0):
            summary["wilcoxon_p"] = float(wilcoxon(top, rnd, alternative="greater").pvalue)
    except Exception:  # pragma: no cover
        pass
    summary["verdict"] = ("explanations beat random masking"
                          if summary["mean_drop_top"] > summary["mean_drop_random"]
                          and summary.get("wilcoxon_p", 1.0) < 0.05
                          else "explanations did NOT clearly beat random masking")
    return summary
