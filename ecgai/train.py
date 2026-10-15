"""Training loop — identical protocol for every model (same data, loss, optimizer,
schedule, augmentation, early stopping and checkpoint selection)."""
from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .metrics import macro_auroc
from .models import ParallelFusionNet, build_model, compute_loss, count_parameters


def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return "not a git repository"


def standardize_split(X: np.ndarray, idx: np.ndarray, mean: np.ndarray, std: np.ndarray,
                      chunk: int = 2048) -> np.ndarray:
    """Copy the rows `idx` of X (may be memory-mapped) into RAM, standardized with TRAINING stats."""
    idx = np.asarray(idx)
    out = np.empty((len(idx),) + tuple(X.shape[1:]), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        order = np.argsort(part)
        block = np.asarray(X[part[order]], dtype=np.float32)
        out[s:s + chunk][order] = (block - mean) / std
    return out


def _augment(x: torch.Tensor, scale: float, noise_std: float) -> torch.Tensor:
    if scale > 0:
        factors = 1.0 + (torch.rand(x.shape[0], 1, 1, device=x.device) * 2 - 1) * scale
        x = x * factors
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std
    return x


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 256,
                  return_alpha: bool = False):
    """X: standardized [N, T, C]. Returns probabilities [N, K] (and fusion branch weights if asked)."""
    model.eval()
    probs, alphas = [], []
    for s in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size], dtype=np.float32)).to(device)
        if isinstance(model, ParallelFusionNet):
            logits, _, alpha = model.forward_all(xb)
            alphas.append(alpha.float().cpu().numpy())
        else:
            logits = model(xb)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0, 0), dtype=np.float32)
    if return_alpha:
        return p, (np.concatenate(alphas) if alphas else None)
    return p


def train_model(name: str, cfg: dict, Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray, Yva: np.ndarray,
                seed: int, out_path: Path, log: Callable[[str], None] = print,
                max_epochs: int | None = None, max_batches: int | None = None) -> dict:
    """Train one model with one seed. Xtr/Xva are already standardized with training statistics.
    Selects the checkpoint with the best VALIDATION macro-AUROC. The test set is never touched here."""
    tcfg, dcfg = cfg["training"], cfg["data"]
    set_seed(seed)
    device = get_device(tcfg.get("device", "auto"))
    model = build_model(name, cfg["models"], n_leads=Xtr.shape[2], n_classes=Ytr.shape[1],
                        seq_len=Xtr.shape[1]).to(device)
    n_params = count_parameters(model)

    pos_weight = None
    if tcfg.get("use_pos_weight", False):
        pos = Ytr.sum(axis=0)
        pos_weight = torch.tensor((len(Ytr) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    epochs = int(max_epochs or tcfg["max_epochs"])
    bs = int(tcfg["batch_size"])
    rng = np.random.default_rng(seed)
    best_auc, best_epoch, bad_epochs = -np.inf, -1, 0
    history = []
    t0 = time.time()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(Xtr))
        n_batches = int(np.ceil(len(order) / bs))
        if max_batches:
            n_batches = min(n_batches, max_batches)
        losses, te = [], time.time()
        for b in range(n_batches):
            idx = order[b * bs:(b + 1) * bs]
            if len(idx) < 2:   # BatchNorm needs more than one sample
                continue
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(Ytr[idx].astype(np.float32)).to(device)
            if tcfg.get("augment", False):
                xb = _augment(xb, tcfg.get("aug_scale", 0.0), tcfg.get("aug_noise_std", 0.0))
            loss, _ = compute_loss(model, xb, yb, criterion)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        p_val = predict_proba(model, Xva, device)
        val_auc = macro_auroc(Yva, p_val)
        scheduler.step(val_auc if np.isfinite(val_auc) else 0.0)
        epoch_time = time.time() - te
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
                        "val_macro_auroc": float(val_auc), "lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch_time_s": round(epoch_time, 2)})
        log(f"[{name} seed={seed}] epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  "
            f"val macro-AUROC {val_auc:.4f}  ({epoch_time:.1f}s)")

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, best_epoch, bad_epochs = float(val_auc), epoch, 0
            torch.save({
                "model_name": name,
                "models_cfg": cfg["models"],
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "seed": int(seed),
                "epoch": int(epoch),
                "val_macro_auroc": float(val_auc),
                "classes": list(dcfg["classes"]),
                "leads": list(dcfg["leads"]),
                "seq_len": int(Xtr.shape[1]),
                "n_leads": int(Xtr.shape[2]),
                "n_params": int(n_params),
                "version": str(cfg.get("version", "")),
            }, out_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(tcfg["patience"]):
                log(f"[{name} seed={seed}] early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    total_time = time.time() - t0
    run = {
        "model": name, "seed": int(seed), "checkpoint": str(out_path), "best_epoch": int(best_epoch),
        "best_val_macro_auroc": float(best_auc), "n_params": int(n_params), "train_time_s": round(total_time, 1),
        "epochs_run": len(history), "device": str(device), "torch_version": torch.__version__,
        "python": platform.python_version(), "git_commit": git_commit(), "history": history,
    }
    out_path.with_suffix(".json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_name"], ckpt["models_cfg"], n_leads=ckpt["n_leads"],
                        n_classes=len(ckpt["classes"]), seq_len=ckpt["seq_len"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
