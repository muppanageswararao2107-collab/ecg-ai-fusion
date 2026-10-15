"""PTB-XL loading: metadata, superclass labels, official folds, and the prepared array cache."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import path as cfg_path
from .preprocess import STANDARD_LEADS, fit_normalization, normalize_lead_name, save_normalization
from .wfdb_io import read_record

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ----------------------------------------------------------------------------- labels
def load_metadata(ptbxl_dir: str | Path) -> pd.DataFrame:
    p = Path(ptbxl_dir) / "ptbxl_database.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download PTB-XL v1.0.3 and set paths.ptbxl_dir in config.yaml.")
    df = pd.read_csv(p, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_scp_map(ptbxl_dir: str | Path) -> dict[str, str]:
    """SCP code -> diagnostic superclass, using only rows with diagnostic == 1."""
    p = Path(ptbxl_dir) / "scp_statements.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found.")
    agg = pd.read_csv(p, index_col=0)
    agg = agg[agg["diagnostic"] == 1]
    return {code: cls for code, cls in agg["diagnostic_class"].items() if isinstance(cls, str)}


def codes_to_superclasses(codes: dict, scp_map: dict[str, str], min_likelihood: float = 0,
                          classes: list[str] = SUPERCLASSES) -> list[str]:
    found = {scp_map[c] for c, lik in codes.items()
             if c in scp_map and float(lik) >= min_likelihood and scp_map[c] in classes}
    return [c for c in classes if c in found]


def build_label_matrix(df: pd.DataFrame, scp_map: dict[str, str], classes: list[str] = SUPERCLASSES,
                       min_likelihood: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Y [N, n_classes] uint8, keep_mask [N] bool). Rows without any superclass are dropped."""
    labels = df["scp_codes"].apply(lambda d: codes_to_superclasses(d, scp_map, min_likelihood, classes))
    Y = np.zeros((len(df), len(classes)), dtype=np.uint8)
    for i, labs in enumerate(labels):
        for lab in labs:
            Y[i, classes.index(lab)] = 1
    keep = Y.sum(axis=1) > 0
    return Y, keep


# ----------------------------------------------------------------------------- splits
def split_indices(folds: np.ndarray, data_cfg: dict) -> dict[str, np.ndarray]:
    folds = np.asarray(folds)
    return {
        "train": np.where(np.isin(folds, data_cfg["train_folds"]))[0],
        "val": np.where(np.isin(folds, data_cfg["val_folds"]))[0],
        "test": np.where(np.isin(folds, data_cfg["test_folds"]))[0],
    }


def assert_no_patient_overlap(patient_ids: np.ndarray, splits: dict[str, np.ndarray]) -> None:
    """Raise AssertionError if any patient appears in more than one split."""
    patient_ids = np.asarray(patient_ids)
    sets = {k: set(patient_ids[v].tolist()) for k, v in splits.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                raise AssertionError(
                    f"{len(overlap)} patient(s) appear in both '{names[i]}' and '{names[j]}' "
                    f"(e.g. {sorted(overlap)[:5]}). This would leak data.")


def assert_splits_disjoint(splits: dict[str, np.ndarray]) -> None:
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.intersect1d(splits[names[i]], splits[names[j]]).size:
                raise AssertionError(f"Splits '{names[i]}' and '{names[j]}' share records.")


# ----------------------------------------------------------------------------- signals
def load_ptbxl_signal(ptbxl_dir: str | Path, rel_path: str, expected_len: int) -> np.ndarray:
    """Read one PTB-XL record (relative path without extension) as [T, 12] mV in standard lead order."""
    sig, header = read_record(Path(ptbxl_dir) / f"{rel_path}.hea")
    names = [normalize_lead_name(s.description) for s in header.signals]
    if names != STANDARD_LEADS:
        order = [names.index(lead) for lead in STANDARD_LEADS]  # raises if a lead is missing
        sig = sig[:, order]
    if sig.shape[0] != expected_len:
        raise ValueError(f"{rel_path}: expected {expected_len} samples, got {sig.shape[0]}")
    if not np.isfinite(sig).all():
        raise ValueError(f"{rel_path}: contains missing values")
    return sig.astype(np.float32)


def prepare_cache(cfg: dict, limit: int | None = None, log: Callable[[str], None] = print) -> dict:
    """Read PTB-XL once and write:
      data/cache/X.npy        float32 [N, 1000, 12] raw millivolts (memory-mappable)
      data/cache/meta.csv     ecg_id, patient_id, strat_fold, split, label columns
      artifacts/norm.json     per-lead mean/std fitted on the TRAINING folds only
      results/data_summary.json
    """
    dcfg = cfg["data"]
    ptbxl_dir = cfg_path(cfg, "ptbxl_dir")
    cache_dir = cfg_path(cfg, "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    classes = dcfg["classes"]

    df = load_metadata(ptbxl_dir)
    scp_map = load_scp_map(ptbxl_dir)
    Y, keep = build_label_matrix(df, scp_map, classes, dcfg.get("min_likelihood", 0))
    n_total, n_dropped = len(df), int((~keep).sum())
    df, Y = df[keep], Y[keep]
    if limit:
        df, Y = df.iloc[:limit], Y[:limit]
        log(f"[prepare] LIMIT MODE: only the first {limit} labelled records are used (for quick tests).")

    rate = int(dcfg["sampling_rate"])
    col = "filename_lr" if rate == 100 else "filename_hr"
    length = int(dcfg["length"])
    X = np.lib.format.open_memmap(cache_dir / "X.npy", mode="w+", dtype=np.float32,
                                  shape=(len(df), length, len(STANDARD_LEADS)))
    for i, rel in enumerate(df[col].tolist()):
        X[i] = load_ptbxl_signal(ptbxl_dir, rel, length)
        if (i + 1) % 2000 == 0 or i + 1 == len(df):
            log(f"[prepare] loaded {i + 1}/{len(df)} records")
    X.flush()

    folds = df["strat_fold"].to_numpy()
    splits = split_indices(folds, dcfg)
    assert_splits_disjoint(splits)
    assert_no_patient_overlap(df["patient_id"].to_numpy(), splits)

    split_name = np.full(len(df), "unused", dtype=object)
    for name, idx in splits.items():
        split_name[idx] = name
    meta = pd.DataFrame({"ecg_id": df.index.to_numpy(), "patient_id": df["patient_id"].to_numpy(),
                         "strat_fold": folds, "split": split_name,
                         "age": df.get("age", pd.Series(np.nan, index=df.index)).to_numpy(),
                         "sex": df.get("sex", pd.Series(np.nan, index=df.index)).to_numpy()})
    for j, c in enumerate(classes):
        meta[c] = Y[:, j]
    meta.to_csv(cache_dir / "meta.csv", index=False)
    # Test-fold-only copy for the website's demo picker (small enough to move from Colab to a laptop).
    test_sorted = np.sort(splits["test"])
    np.save(cache_dir / "test_X.npy", np.asarray(X[test_sorted], dtype=np.float32))
    meta.iloc[test_sorted].to_csv(cache_dir / "test_meta.csv", index=False)

    if len(splits["train"]) == 0:
        raise ValueError("No training records found for the configured train folds.")
    mean, std = fit_normalization(_TrainView(X, splits["train"]))
    save_normalization(cfg_path(cfg, "artifacts_dir") / "norm.json", mean, std, STANDARD_LEADS, rate,
                       length, fitted_on=f"PTB-XL folds {dcfg['train_folds']} (training only)",
                       n_records=len(splits["train"]))

    summary = {
        "records_in_database": int(n_total),
        "records_without_superclass_dropped": n_dropped,
        "records_used": int(len(df)),
        "limit_mode": bool(limit),
        "sampling_rate": rate,
        "classes": classes,
        "split_sizes": {k: int(len(v)) for k, v in splits.items()},
        "split_patients": {k: int(meta.loc[v, "patient_id"].nunique()) for k, v in splits.items()},
        "label_counts": {k: {c: int(Y[v, j].sum()) for j, c in enumerate(classes)} for k, v in splits.items()},
        "patient_overlap_check": "passed",
    }
    res_dir = cfg_path(cfg, "results_dir")
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))
    return summary


class _TrainView:
    """Lazy view of X restricted to some rows, so normalization can be fitted in chunks
    without copying the whole training set into memory."""

    def __init__(self, X: np.ndarray, idx: np.ndarray):
        self.X, self.idx = X, np.sort(idx)
        self.shape = (len(self.idx),) + X.shape[1:]

    def __getitem__(self, sl: slice) -> np.ndarray:
        return self.X[self.idx[sl]]


def load_cache(cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    cache_dir = cfg_path(cfg, "cache_dir")
    x_path, m_path = cache_dir / "X.npy", cache_dir / "meta.csv"
    if not x_path.exists() or not m_path.exists():
        raise FileNotFoundError("Prepared data not found. Run: python scripts/prepare_data.py")
    X = np.load(x_path, mmap_mode="r")
    meta = pd.read_csv(m_path)
    if len(meta) != X.shape[0]:
        raise ValueError("Cache is inconsistent: meta.csv and X.npy have different lengths. Re-run prepare_data.py")
    return X, meta


def labels_from_meta(meta: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return meta[classes].to_numpy(dtype=np.float32)
