"""Parsing and validation of uploaded ECGs (WFDB or CSV).

Rules:
* Only complete 12-lead recordings are accepted; missing leads are never invented.
* Uploaded files are written to a temporary folder only while being parsed and are
  deleted immediately afterwards.
* Normalization uses the statistics saved from the training folds; it is never refit.
"""
from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .measures import estimate_heart_rate
from .preprocess import STANDARD_LEADS, make_windows, normalize_lead_name, resample, standardize
from .quality import FAIL, QualityCheck, overall_status, run_quality_checks
from .wfdb_io import WFDBError, read_record

IGNORED_CSV_COLUMNS = {"time", "t", "sample", "samples", "index", "timestamp", "time_s", "time_ms", "unnamed: 0"}
SUPPORTED_HELP = (
    "Supported inputs: (1) a WFDB record — upload the .hea file together with its .dat file; "
    "(2) a CSV file with a header row naming the 12 leads "
    "(I, II, III, aVR, aVL, aVF, V1–V6), one row per sample, values in mV or µV. "
    "Recordings must be at least 10 seconds long."
)


class UploadError(ValueError):
    """A user-facing validation error. The message is shown in the app as-is."""


@dataclass
class ECGInput:
    signal: np.ndarray            # [n_samples, 12] millivolts, standard lead order
    fs: float
    leads: list[str]
    filename: str
    source: str                   # "PTB-XL test record" | "Uploaded WFDB" | "Uploaded CSV"
    notes: list[str] = field(default_factory=list)
    ecg_id: int | None = None
    true_labels: list[str] | None = None

    @property
    def duration_s(self) -> float:
        return self.signal.shape[0] / self.fs


# ----------------------------------------------------------------------------- helpers
def _check_size(name: str, data: bytes, max_bytes: int) -> None:
    if len(data) == 0:
        raise UploadError(f"'{name}' is empty.")
    if len(data) > max_bytes:
        raise UploadError(f"'{name}' is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB.")


def _safe_name(name: str) -> str:
    base = Path(name).name
    if not base or base in (".", ".."):
        raise UploadError("Invalid file name.")
    return base


def reorder_to_standard(signal: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Map arbitrary lead labels to the 12 standard leads. Raises UploadError if any is missing."""
    mapped = [normalize_lead_name(n) for n in names]
    present = [m for m in mapped if m]
    dupes = sorted({m for m in present if present.count(m) > 1})
    if dupes:
        raise UploadError(f"Duplicate leads found: {', '.join(dupes)}.")
    missing = [lead for lead in STANDARD_LEADS if lead not in present]
    if missing:
        found = ", ".join(str(n) for n in names) or "none"
        if len(present) <= 1:
            raise UploadError(
                f"This recording has {len(names)} lead(s) ({found}). The models were trained on 12-lead ECGs "
                "and need all 12 leads (I, II, III, aVR, aVL, aVF, V1–V6). Single-lead or reduced-lead "
                "recordings cannot be analysed, and missing leads are never generated artificially.")
        raise UploadError(
            f"Missing lead(s): {', '.join(missing)}. Leads found: {found}. All 12 standard leads are required; "
            "missing leads are never generated artificially.")
    order = [mapped.index(lead) for lead in STANDARD_LEADS]
    return signal[:, order], list(STANDARD_LEADS)


# ----------------------------------------------------------------------------- WFDB
def parse_wfdb_upload(files: dict[str, bytes], max_bytes: int) -> ECGInput:
    """files: {filename: content}. Needs exactly one .hea and the signal file(s) it references."""
    if not files:
        raise UploadError("No files were uploaded. " + SUPPORTED_HELP)
    clean = {}
    for name, data in files.items():
        base = _safe_name(name)
        ext = Path(base).suffix.lower()
        if ext not in (".hea", ".dat"):
            raise UploadError(f"'{base}' is not a WFDB file (.hea/.dat). " + SUPPORTED_HELP)
        _check_size(base, data, max_bytes)
        clean[base] = data
    heas = [n for n in clean if n.lower().endswith(".hea")]
    if len(heas) != 1:
        raise UploadError(f"Upload exactly one .hea header file (found {len(heas)}) plus its .dat file.")
    hea = heas[0]

    with tempfile.TemporaryDirectory(prefix="ecg_upload_") as tmp:
        for name, data in clean.items():
            (Path(tmp) / name).write_bytes(data)
        try:
            signal, header = read_record(Path(tmp) / hea)
        except WFDBError as exc:
            raise UploadError(f"Could not read the WFDB record: {exc}") from exc
        except (OSError, ValueError) as exc:
            raise UploadError(f"The WFDB files appear to be corrupted: {exc}") from exc
    # temporary folder (and the uploaded bytes written to it) is deleted here

    names = [s.description for s in header.signals]
    signal, leads = reorder_to_standard(signal, names)
    notes = [f"Sampling rate {header.fs:g} Hz read from the WFDB header."]
    units = {s.units for s in header.signals}
    notes.append(f"Units in header: {', '.join(sorted(units))} (converted to mV where needed).")
    return ECGInput(signal=signal.astype(np.float32), fs=float(header.fs), leads=leads,
                    filename=hea, source="Uploaded WFDB", notes=notes)


# ----------------------------------------------------------------------------- CSV
def parse_csv_upload(name: str, data: bytes, fs: float | None, units: str, max_bytes: int) -> ECGInput:
    base = _safe_name(name)
    if Path(base).suffix.lower() != ".csv":
        raise UploadError(f"'{base}' is not a .csv file. " + SUPPORTED_HELP)
    _check_size(base, data, max_bytes)
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise UploadError("Enter the sampling rate (Hz) of the CSV recording. It cannot be read from a CSV file.")
    if fs < 50 or fs > 10000:
        raise UploadError(f"A sampling rate of {fs:g} Hz is outside the supported range (50–10000 Hz).")
    if units not in ("mV", "µV"):
        raise UploadError("Units must be mV or µV.")
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:  # pandas raises several error types for malformed files
        raise UploadError(f"The CSV file could not be parsed: {exc}") from exc
    if df.empty:
        raise UploadError("The CSV file has no data rows.")

    keep_cols = [c for c in df.columns if str(c).strip().lower() not in IGNORED_CSV_COLUMNS]
    if not any(normalize_lead_name(c) for c in keep_cols):
        raise UploadError("The CSV needs a header row naming the leads (I, II, III, aVR, aVL, aVF, V1–V6). "
                          + SUPPORTED_HELP)
    lead_cols = [c for c in keep_cols if normalize_lead_name(c)]
    numeric = df[lead_cols].apply(pd.to_numeric, errors="coerce")
    bad = int(numeric.isna().sum().sum())
    if bad:
        worst = numeric.isna().sum().sort_values(ascending=False)
        worst = ", ".join(f"{k} ({v})" for k, v in worst[worst > 0].items())
        raise UploadError(f"The CSV contains {bad} missing or non-numeric values (per lead: {worst}). "
                          "Gaps are never filled in automatically; please export a complete recording.")
    signal = numeric.to_numpy(dtype=np.float64)
    if units == "µV":
        signal = signal / 1000.0
    signal, leads = reorder_to_standard(signal, [str(c) for c in lead_cols])
    notes = [f"Sampling rate {fs:g} Hz entered by the user.", f"Values interpreted as {units}."]
    ignored = [str(c) for c in df.columns if c not in lead_cols]
    if ignored:
        notes.append(f"Ignored non-lead columns: {', '.join(ignored)}.")
    return ECGInput(signal=signal.astype(np.float32), fs=float(fs), leads=leads, filename=base,
                    source="Uploaded CSV", notes=notes)


# ----------------------------------------------------------------------------- model input
@dataclass
class PreparedECG:
    signal_100: np.ndarray        # [n, 12] mV at the model sampling rate
    windows_norm: np.ndarray      # [W, length, 12] standardized with training statistics
    quality: list[QualityCheck]
    quality_status: str
    heart_rate: dict
    notes: list[str]


def prepare_for_model(inp: ECGInput, norm: dict, max_windows: int) -> PreparedECG:
    """Quality checks -> resample -> 10 s windows -> standardize with saved training stats."""
    fs_model, length = int(norm["sampling_rate"]), int(norm["length"])
    if inp.signal.ndim != 2 or inp.signal.shape[1] != 12:
        raise UploadError("Internal error: signal must have 12 leads.")
    min_len = length / fs_model
    if inp.duration_s < min_len - 1e-9:
        raise UploadError(f"The recording is {inp.duration_s:.1f} s long; at least {min_len:.0f} s are required.")

    quality = run_quality_checks(inp.signal, inp.fs, inp.leads)
    status = overall_status(quality)
    notes = list(inp.notes)
    sig100 = resample(inp.signal, inp.fs, fs_model) if status != FAIL else inp.signal.astype(np.float32)
    if status != FAIL and abs(inp.fs - fs_model) > 1e-9:
        notes.append(f"Resampled from {inp.fs:g} Hz to {fs_model} Hz (polyphase anti-aliasing filter).")
    hr = estimate_heart_rate(sig100, fs_model, inp.leads) if status != FAIL else {
        "available": False, "bpm": None, "n_beats": 0, "lead": "II", "reason": "Quality check failed."}

    if status == FAIL:
        windows = np.zeros((0, length, 12), dtype=np.float32)
    else:
        if sig100.shape[0] < length:  # rounding during resampling
            raise UploadError(f"The recording is shorter than {min_len:.0f} s after resampling.")
        raw_windows, dropped, truncated = make_windows(sig100, length, max_windows)
        if raw_windows.shape[0] > 1:
            notes.append(f"Split into {raw_windows.shape[0]} non-overlapping 10 s windows; "
                         "probabilities are averaged across windows.")
        if dropped:
            notes.append(f"The final {dropped / fs_model:.1f} s (shorter than one window) were not analysed.")
        if truncated:
            notes.append(f"Only the first {max_windows * length / fs_model / 60:.1f} minutes were analysed.")
        windows = standardize(raw_windows, norm["mean"], norm["std"])
    return PreparedECG(sig100, windows, quality, status, hr, notes)
