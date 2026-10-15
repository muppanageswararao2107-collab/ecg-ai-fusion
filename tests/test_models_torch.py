"""Model tests — need PyTorch (skipped automatically when it is not installed)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecgai.config import load_config  # noqa: E402
from ecgai.inference import ModelBundle, ModelsNotTrained, check_ready  # noqa: E402
from ecgai.models import MODEL_NAMES, ParallelFusionNet, build_model, count_parameters  # noqa: E402
from ecgai.preprocess import STANDARD_LEADS, save_normalization  # noqa: E402
from ecgai.train import load_checkpoint, predict_proba, train_model  # noqa: E402
from ecgai.xai import deletion_test, grad_cam_1d, integrated_gradients  # noqa: E402

CFG = load_config()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_shape_and_backward(name):
    model = build_model(name, CFG["models"])
    x = torch.randn(3, 1000, 12)
    out = model(x)
    assert out.shape == (3, 5)
    out.sum().backward()
    assert count_parameters(model) > 10_000


def test_fusion_alpha_sums_to_one_and_branch_dropout_only_in_training():
    model = build_model("fusion", CFG["models"])
    model.eval()
    _, aux, alpha = model.forward_all(torch.randn(4, 1000, 12))
    assert torch.allclose(alpha.sum(1), torch.ones(4), atol=1e-5)
    assert set(aux) == {"cnn", "lstm", "transformer"}
    model.train()
    model.branch_dropout = 1.0
    _, _, alpha = model.forward_all(torch.randn(8, 1000, 12))
    assert (alpha < 1e-3).sum(1).eq(1).all()  # exactly one branch hidden per sample


def _toy_data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1000, 12)).astype(np.float32)
    Y = (rng.random((n, 5)) < 0.3).astype(np.float32)
    Y[:5] = np.eye(5)
    return X, Y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_train_save_load_predict_explain(tmp_path, name):
    Xtr, Ytr = _toy_data(48, 0)
    Xva, Yva = _toy_data(24, 1)
    run = train_model(name, CFG, Xtr, Ytr, Xva, Yva, seed=0, out_path=tmp_path / f"{name}.pt",
                      max_epochs=1, max_batches=1, log=lambda *_: None)
    assert run["best_epoch"] == 1 and (tmp_path / f"{name}.json").exists()
    model, meta = load_checkpoint(tmp_path / f"{name}.pt")
    p = predict_proba(model, Xva, torch.device("cpu"))
    assert p.shape == (24, 5) and np.all((p >= 0) & (p <= 1))
    ig = integrated_gradients(model, Xva[0], 1, steps=16, batch=8)
    assert ig["attr"].shape == (1000, 12)
    assert abs(ig["delta"]) < 0.5 * max(1.0, abs(ig["logit"] - ig["baseline_logit"]))  # completeness, coarse
    d = deletion_test(model, Xva[0], 1, ig["attr"], random_repeats=2)
    assert set(d) >= {"drop_top", "drop_random"}
    if name == "cnn":
        cam = grad_cam_1d(model, Xva[0], 1)
        assert cam.shape == (1000,) and cam.max() <= 1.0


def test_same_seed_is_reproducible(tmp_path):
    Xtr, Ytr = _toy_data(32, 0)
    Xva, Yva = _toy_data(16, 1)
    ps = []
    for k in range(2):
        train_model("cnn", CFG, Xtr, Ytr, Xva, Yva, seed=7, out_path=tmp_path / f"c{k}.pt",
                    max_epochs=1, max_batches=2, log=lambda *_: None)
        m, _ = load_checkpoint(tmp_path / f"c{k}.pt")
        ps.append(predict_proba(m, Xva, torch.device("cpu")))
    assert np.allclose(ps[0], ps[1], atol=1e-5)


def test_bundle_reports_not_trained_and_then_predicts(tmp_path):
    cfg = json.loads(json.dumps({k: v for k, v in CFG.items() if not k.startswith("_")}))
    cfg["paths"] = {"ptbxl_dir": str(tmp_path), "cache_dir": str(tmp_path / "cache"),
                    "artifacts_dir": str(tmp_path / "art"), "results_dir": str(tmp_path / "res")}
    ok, msg = check_ready(cfg)
    assert not ok and "not trained" in msg
    with pytest.raises(ModelsNotTrained):
        ModelBundle(cfg)

    save_normalization(tmp_path / "art" / "norm.json", np.zeros(12), np.ones(12), STANDARD_LEADS, 100, 1000, "t", 1)
    Xtr, Ytr = _toy_data(32, 0)
    Xva, Yva = _toy_data(16, 1)
    dep = {"classes": CFG["data"]["classes"], "models": {}}
    for name in MODEL_NAMES:
        ck = tmp_path / "art" / "models" / f"{name}_seed0.pt"
        train_model(name, cfg, Xtr, Ytr, Xva, Yva, seed=0, out_path=ck, max_epochs=1, max_batches=1,
                    log=lambda *_: None)
        dep["models"][name] = {"checkpoint": str(ck), "seed": 0, "thresholds": {c: 0.5 for c in dep["classes"]},
                               "val_macro_auroc": 0.5, "test_macro_auroc": 0.5, "n_params": 1}
    dep["ensemble"] = {"members": ["cnn", "lstm", "transformer"], "weights": [0.4, 0.3, 0.3],
                       "thresholds": {c: 0.5 for c in dep["classes"]}, "val_macro_auroc": 0.5, "test_macro_auroc": 0.5}
    (tmp_path / "art" / "deployment.json").write_text(json.dumps(dep))
    bundle = ModelBundle(cfg)
    out = bundle.predict(Xva[:2])
    assert set(out["models"]) == {"cnn", "lstm", "transformer", "fusion", "ensemble"}
    assert abs(sum(out["models"]["fusion"]["branch_weights"].values()) - 1) < 1e-4
    assert out["n_windows"] == 2
    assert isinstance(bundle.models["fusion"], ParallelFusionNet)
