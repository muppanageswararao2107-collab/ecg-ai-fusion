"""Automated heart-rate estimate.

This is a simple R-peak detector. It is NOT a validated clinical measurement; the
app and report label it as an automated estimate and show "unavailable" whenever the
detection is not self-consistent.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def estimate_heart_rate(signal_mv: np.ndarray, fs: float, leads: list[str]) -> dict:
    """Returns {"available": bool, "bpm": float|None, "n_beats": int, "lead": str, "reason": str}."""
    lead = "II" if "II" in leads else leads[0]
    x = np.asarray(signal_mv[:, leads.index(lead)], dtype=np.float64)
    out = {"available": False, "bpm": None, "n_beats": 0, "lead": lead, "reason": ""}
    if x.size < fs * 4 or not np.isfinite(x).all():
        out["reason"] = "Recording too short or contains missing values."
        return out
    hi = min(15.0, 0.45 * fs)
    if hi <= 5.0:
        out["reason"] = "Sampling rate too low for QRS detection."
        return out
    b, a = butter(2, [5.0 / (fs / 2), hi / (fs / 2)], btype="band")
    y = filtfilt(b, a, x)
    energy = np.convolve(np.gradient(y) ** 2, np.ones(max(1, int(0.08 * fs))) / max(1, int(0.08 * fs)), "same")
    thr = 0.3 * np.percentile(energy, 99)
    peaks, _ = find_peaks(energy, height=thr, distance=int(0.3 * fs))
    if len(peaks) < 4:
        out["reason"] = "Too few beats detected."
        return out
    rr = np.diff(peaks) / fs
    med = float(np.median(rr))
    cv = float(np.std(rr) / med) if med > 0 else np.inf
    bpm = 60.0 / med if med > 0 else np.inf
    out["n_beats"] = int(len(peaks))
    if not 30 <= bpm <= 220:
        out["reason"] = f"Detected rate ({bpm:.0f} bpm) is outside the plausible range."
        return out
    if cv > 0.35:
        out["reason"] = "Beat-to-beat intervals are too irregular for a reliable estimate."
        return out
    out.update(available=True, bpm=round(bpm, 1), reason="Automated estimate from R-peak detection.")
    return out
