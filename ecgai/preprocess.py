"""Signal preprocessing shared by training and inference.

The same functions are used on PTB-XL training data and on uploaded ECGs, so the
model always sees inputs prepared the same way.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_LEAD_LOOKUP = {name.upper(): name for name in STANDARD_LEADS}
_ALIASES = {"LEAD I": "I", "LEAD II": "II", "LEAD III": "III", "LEADI": "I", "LEADII": "II",
            "LEADIII": "III", "MLII": None, "MLI": None}


def normalize_lead_name(name: str) -> str | None:
    """Map a lead label such as 'avr', 'AVR', 'Lead II' or ' v1 ' to the standard name.
    Returns None when the label is not one of the 12 standard leads."""
    key = str(name).strip().upper().replace("-", "").replace("_", " ")
    key = " ".join(key.split())
    if key in _ALIASES:
        return _ALIASES[key]
    key = key.replace(" ", "")
    return _LEAD_LOOKUP.get(key)


def resample(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Resample [n_samples, n_leads] from fs_in to fs_out with a polyphase anti-aliasing filter."""
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Sampling rates must be positive.")
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(signal, dtype=np.float32).copy()
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    out = resample_poly(np.asarray(signal, dtype=np.float64), ratio.numerator, ratio.denominator, axis=0)
    return out.astype(np.float32)


def make_windows(signal: np.ndarray, length: int, max_windows: int | None = None
                 ) -> tuple[np.ndarray, int, bool]:
    """Split [n, leads] into non-overlapping windows of `length` samples.

    Returns (windows [W, length, leads], samples_dropped_at_end, truncated_by_max_windows).
    Raises ValueError when the signal is shorter than one window."""
    n = signal.shape[0]
    if n < length:
        raise ValueError(f"Recording has {n} samples; at least {length} are needed.")
    n_win = n // length
    truncated = False
    if max_windows is not None and n_win > max_windows:
        n_win, truncated = max_windows, True
    used = n_win * length
    dropped = n - used if not truncated else 0
    windows = signal[:used].reshape(n_win, length, signal.shape[1]).astype(np.float32)
    return windows, dropped, truncated


def fit_normalization(X: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead mean and std over [N, T, C]. Computed in chunks (works on memory-mapped arrays).
    Call this on TRAINING data only."""
    n_leads = X.shape[2]
    s = np.zeros(n_leads, dtype=np.float64)
    ss = np.zeros(n_leads, dtype=np.float64)
    count = 0
    for start in range(0, X.shape[0], chunk):
        block = np.asarray(X[start:start + chunk], dtype=np.float64)
        s += block.sum(axis=(0, 1))
        ss += (block ** 2).sum(axis=(0, 1))
        count += block.shape[0] * block.shape[1]
    if count == 0:
        raise ValueError("Cannot fit normalization on an empty array.")
    mean = s / count
    var = np.maximum(ss / count - mean ** 2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply saved per-lead z-score normalization (never refit on the input)."""
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def save_normalization(path: str | Path, mean: np.ndarray, std: np.ndarray, leads: list[str],
                       fs: int, length: int, fitted_on: str, n_records: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {"leads": list(leads), "mean": [float(v) for v in mean], "std": [float(v) for v in std],
            "sampling_rate": int(fs), "length": int(length), "fitted_on": fitted_on,
            "n_records": int(n_records), "units": "mV"}
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_normalization(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["mean"] = np.asarray(data["mean"], dtype=np.float32)
    data["std"] = np.asarray(data["std"], dtype=np.float32)
    return data
