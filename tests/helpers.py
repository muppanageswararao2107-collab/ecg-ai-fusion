"""Synthetic ECG-like signals for UNIT TESTS ONLY.

These are sums of Gaussian bumps shaped like P-QRS-T complexes. They exist to test parsing,
resampling and quality checks. They are never shown in the app or used as results.
"""
from __future__ import annotations

import numpy as np

from ecgai.preprocess import STANDARD_LEADS


def synthetic_ecg(seconds: float = 10.0, fs: float = 500.0, bpm: float = 72.0, seed: int = 0,
                  noise: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    beat = 60.0 / bpm
    base = np.zeros_like(t)
    for onset in np.arange(0.2, seconds, beat):
        for center, width, amp in [(0.0, 0.025, 0.12), (0.16, 0.010, -0.1), (0.18, 0.012, 1.0),
                                   (0.20, 0.010, -0.25), (0.42, 0.045, 0.3)]:
            base += amp * np.exp(-0.5 * ((t - onset - center) / width) ** 2)
    leads = []
    for i in range(len(STANDARD_LEADS)):
        scale = (0.5 + 0.1 * (i % 4)) * (-1.0 if STANDARD_LEADS[i] == "aVR" else 1.0)
        leads.append(scale * base + noise * rng.standard_normal(t.size))
    return np.stack(leads, axis=1).astype(np.float32)
