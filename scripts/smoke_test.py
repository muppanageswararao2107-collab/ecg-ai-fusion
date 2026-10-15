"""Run this FIRST after installing: checks that every model builds, trains for a few steps,
saves/loads, explains and predicts — on random tensors, in about a minute. No dataset needed.

    python scripts/smoke_test.py

Random tensors are used only to check the code runs. Nothing produced here is a result.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ecgai.config import load_config  # noqa: E402
from ecgai.models import MODEL_NAMES, build_model, count_parameters  # noqa: E402
from ecgai.train import get_device, load_checkpoint, predict_proba, train_model  # noqa: E402
from ecgai.xai import deletion_test, grad_cam_1d, integrated_gradients  # noqa: E402


def main() -> None:
    cfg = load_config()
    dev = get_device(cfg["training"].get("device", "auto"))
    print(f"PyTorch {torch.__version__}  device: {dev}")
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((64, 1000, 12)).astype(np.float32)
    Ytr = (rng.random((64, 5)) < 0.3).astype(np.float32)
    Ytr[:5] = np.eye(5)  # every class present
    Xva = rng.standard_normal((32, 1000, 12)).astype(np.float32)
    Yva = (rng.random((32, 5)) < 0.3).astype(np.float32)
    Yva[:5] = np.eye(5)

    with tempfile.TemporaryDirectory() as tmp:
        for name in MODEL_NAMES:
            t0 = time.time()
            model = build_model(name, cfg["models"])
            out = model(torch.from_numpy(Xtr[:4]))
            assert out.shape == (4, 5), out.shape
            run = train_model(name, cfg, Xtr, Ytr, Xva, Yva, seed=0, out_path=Path(tmp) / f"{name}.pt",
                              max_epochs=1, max_batches=2, log=lambda *_: None)
            m2, meta = load_checkpoint(Path(tmp) / f"{name}.pt", dev)
            p = predict_proba(m2, Xva[:3], dev)
            assert p.shape == (3, 5) and np.all((p >= 0) & (p <= 1))
            ig = integrated_gradients(m2, Xva[0], 0, steps=8, batch=4)
            assert ig["attr"].shape == (1000, 12)
            d = deletion_test(m2, Xva[0], 0, ig["attr"], random_repeats=2)
            if name == "cnn":
                cam = grad_cam_1d(m2, Xva[0], 0)
                assert cam.shape == (1000,)
            if name == "fusion":
                _, alpha = predict_proba(m2, Xva[:3], dev, return_alpha=True)
                assert np.allclose(alpha.sum(1), 1, atol=1e-4)
            print(f"  OK  {name:12s} params {count_parameters(model):>9,}  ckpt epoch {run['best_epoch']}  "
                  f"IG completeness err {ig['delta']:+.3f}  deletion drop {d['drop_top']:+.3f}  "
                  f"({time.time() - t0:.1f}s)")
    print("\nAll model code paths work. Next: download PTB-XL and run scripts/prepare_data.py")


if __name__ == "__main__":
    main()
