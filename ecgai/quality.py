"""Signal-quality checks run on every ECG before analysis.

Each check returns PASS / WARN / FAIL with a plain-language message. Any FAIL blocks
the analysis; WARNs are shown to the user and printed in the report.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class QualityCheck:
    name: str
    status: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_quality_checks(signal_mv: np.ndarray, fs: float, leads: list[str]) -> list[QualityCheck]:
    """signal_mv: [n_samples, n_leads] in millivolts."""
    x = np.asarray(signal_mv, dtype=np.float64)
    checks: list[QualityCheck] = []

    # 1. Missing values
    n_bad = int((~np.isfinite(x)).sum())
    if n_bad:
        checks.append(QualityCheck("Missing values", FAIL,
                                   f"{n_bad} missing or non-numeric samples. Gaps are never filled in automatically."))
        return checks  # the other checks are meaningless with NaNs
    checks.append(QualityCheck("Missing values", PASS, "No missing samples."))

    # 2. Flat-line leads
    ptp = x.max(axis=0) - x.min(axis=0)
    diffs = np.abs(np.diff(x, axis=0))
    still_frac = (diffs < 1e-6).mean(axis=0) if x.shape[0] > 1 else np.ones(x.shape[1])
    flat = [leads[i] for i in range(x.shape[1]) if ptp[i] < 0.05 or still_frac[i] > 0.5]
    if len(flat) >= 3:
        checks.append(QualityCheck("Flat-line leads", FAIL,
                                   f"{len(flat)} leads carry almost no signal ({', '.join(flat)}). "
                                   "Electrodes may be disconnected."))
    elif flat:
        checks.append(QualityCheck("Flat-line leads", WARN, f"Little or no signal in: {', '.join(flat)}."))
    else:
        checks.append(QualityCheck("Flat-line leads", PASS, "All leads carry signal."))

    # 3. Clipping / saturation: many samples sitting exactly at the lead's extreme value
    clipped = []
    for i in range(x.shape[1]):
        col = x[:, i]
        if ptp[i] < 0.05:
            continue
        at_max = np.isclose(col, col.max(), atol=1e-6).mean()
        at_min = np.isclose(col, col.min(), atol=1e-6).mean()
        if max(at_max, at_min) > 0.01:
            clipped.append(leads[i])
    if clipped:
        checks.append(QualityCheck("Clipping / saturation", WARN,
                                   f"Signal appears clipped in: {', '.join(clipped)}."))
    else:
        checks.append(QualityCheck("Clipping / saturation", PASS, "No clipping detected."))

    # 4. Amplitude plausibility (typical ECG amplitudes are within a few mV)
    peak = np.abs(x - np.median(x, axis=0)).max(axis=0)
    huge = [leads[i] for i in range(x.shape[1]) if peak[i] > 10.0]
    if len(huge) >= 6:
        checks.append(QualityCheck("Amplitude range", FAIL,
                                   "Most leads exceed 10 mV. The units are probably wrong "
                                   "(e.g. microvolts read as millivolts)."))
    elif huge:
        checks.append(QualityCheck("Amplitude range", WARN,
                                   f"Unusually large amplitude (>10 mV) in: {', '.join(huge)}. "
                                   "Check units or motion artefacts."))
    else:
        checks.append(QualityCheck("Amplitude range", PASS, "Amplitudes are within a plausible range."))

    # 5. Strong high-frequency noise relative to the signal (simple, conservative)
    if x.shape[0] > 3:
        hf = np.median(np.abs(np.diff(x, n=2, axis=0)), axis=0)
        ratio = hf / np.maximum(ptp, 1e-6)
        noisy = [leads[i] for i in range(x.shape[1]) if ptp[i] >= 0.2 and ratio[i] > 0.2]
        if noisy:
            checks.append(QualityCheck("High-frequency noise", WARN, f"Noisy signal in: {', '.join(noisy)}."))
        else:
            checks.append(QualityCheck("High-frequency noise", PASS, "No excessive high-frequency noise."))
    return checks


def overall_status(checks: list[QualityCheck]) -> str:
    statuses = {c.status for c in checks}
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return PASS
