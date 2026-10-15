"""Minimal, dependency-free WFDB reader/writer.

Supports single-segment records stored in formats 16, 212 and 80, which covers
PTB-XL (format 16) and MIT-BIH style files (format 212). Anything else raises
``WFDBError`` with a clear message instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SUPPORTED_FORMATS = {16, 212, 80}
_FMT_RE = re.compile(r"^(\d+)(?:x(\d+))?(?::(\d+))?(?:\+(\d+))?$")
_GAIN_RE = re.compile(r"^([-+]?[\d.]+(?:[eE][-+]?\d+)?)(?:\((-?\d+)\))?(?:/(\S+))?$")


class WFDBError(ValueError):
    """Raised when a WFDB record cannot be parsed safely."""


@dataclass
class SignalSpec:
    file_name: str
    fmt: int
    byte_offset: int
    gain: float
    baseline: int
    units: str
    adc_zero: int
    description: str


@dataclass
class WFDBHeader:
    record_name: str
    n_sig: int
    fs: float
    n_samples: int | None
    signals: list[SignalSpec] = field(default_factory=list)
    comments: list[str] = field(default_factory=list)


def parse_header(text: str) -> WFDBHeader:
    """Parse the text of a .hea file."""
    lines = [ln.strip() for ln in text.splitlines()]
    comments = [ln[1:].strip() for ln in lines if ln.startswith("#")]
    content = [ln for ln in lines if ln and not ln.startswith("#")]
    if not content:
        raise WFDBError("Header file is empty.")

    rec = content[0].split()
    if len(rec) < 2:
        raise WFDBError("Header record line is incomplete (needs record name and number of signals).")
    if "/" in rec[0]:
        raise WFDBError("Multi-segment WFDB records are not supported. Please export a single-segment record.")
    try:
        n_sig = int(rec[1])
    except ValueError as exc:
        raise WFDBError(f"Invalid number of signals in header: {rec[1]!r}") from exc
    if n_sig < 1:
        raise WFDBError("Header declares zero signals.")

    fs = 250.0  # WFDB default when the field is absent
    if len(rec) >= 3:
        fs_token = rec[2].split("/")[0].split("(")[0]
        try:
            fs = float(fs_token)
        except ValueError as exc:
            raise WFDBError(f"Invalid sampling frequency in header: {rec[2]!r}") from exc
    if fs <= 0:
        raise WFDBError("Sampling frequency must be positive.")

    n_samples = None
    if len(rec) >= 4:
        try:
            n_samples = int(rec[3])
        except ValueError as exc:
            raise WFDBError(f"Invalid number of samples in header: {rec[3]!r}") from exc

    sig_lines = content[1:1 + n_sig]
    if len(sig_lines) != n_sig:
        raise WFDBError(f"Header declares {n_sig} signals but describes {len(sig_lines)}.")

    signals = []
    for i, ln in enumerate(sig_lines):
        parts = ln.split(maxsplit=8)
        if len(parts) < 2:
            raise WFDBError(f"Signal line {i + 1} is incomplete: {ln!r}")
        m = _FMT_RE.match(parts[1])
        if not m:
            raise WFDBError(f"Cannot parse storage format {parts[1]!r} on signal line {i + 1}.")
        fmt = int(m.group(1))
        if m.group(2) and int(m.group(2)) != 1:
            raise WFDBError("Records with multiple samples per frame are not supported.")
        if m.group(3) and int(m.group(3)) != 0:
            raise WFDBError("Records with signal skew are not supported.")
        if fmt not in SUPPORTED_FORMATS:
            raise WFDBError(f"WFDB storage format {fmt} is not supported (supported: 16, 212, 80).")
        byte_offset = int(m.group(4) or 0)

        gain, baseline, units = 200.0, None, "mV"
        if len(parts) >= 3:
            g = _GAIN_RE.match(parts[2])
            if not g:
                raise WFDBError(f"Cannot parse gain field {parts[2]!r} on signal line {i + 1}.")
            gain = float(g.group(1)) or 200.0
            if g.group(2) is not None:
                baseline = int(g.group(2))
            if g.group(3):
                units = g.group(3)
        adc_zero = int(parts[4]) if len(parts) >= 5 else 0
        if baseline is None:
            baseline = adc_zero
        description = parts[8].strip() if len(parts) >= 9 else f"sig{i}"
        signals.append(SignalSpec(parts[0], fmt, byte_offset, gain, baseline, units, adc_zero, description))

    return WFDBHeader(rec[0], n_sig, fs, n_samples, signals, comments)


def _decode(raw: bytes, fmt: int, n_values: int | None) -> tuple[np.ndarray, int]:
    """Decode raw bytes into int32 digital values; returns (values, missing_value_code)."""
    if fmt == 16:
        usable = len(raw) - (len(raw) % 2)
        vals = np.frombuffer(raw[:usable], dtype="<i2").astype(np.int32)
        missing = -32768
    elif fmt == 80:
        vals = np.frombuffer(raw, dtype=np.uint8).astype(np.int32) - 128
        missing = -128
    elif fmt == 212:
        usable = len(raw) - (len(raw) % 3)
        b = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        s0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
        s1 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
        vals = np.empty(b.shape[0] * 2, dtype=np.int32)
        vals[0::2], vals[1::2] = s0, s1
        vals[vals > 2047] -= 4096
        missing = -2048
    else:  # pragma: no cover - guarded by parse_header
        raise WFDBError(f"Unsupported format {fmt}")
    if n_values is not None:
        vals = vals[:n_values]
    return vals, missing


def read_record(hea_path: str | Path) -> tuple[np.ndarray, WFDBHeader]:
    """Read a WFDB record. Returns (signal [n_samples, n_sig] float32 in millivolts where
    units are known, header). Missing samples become NaN."""
    hea_path = Path(hea_path)
    header = parse_header(hea_path.read_text(encoding="latin-1"))
    folder = hea_path.parent

    groups: dict[str, list[int]] = {}
    for idx, s in enumerate(header.signals):
        groups.setdefault(s.file_name, []).append(idx)

    columns: dict[int, np.ndarray] = {}
    n_samples = header.n_samples
    for fname, idxs in groups.items():
        fmts = {header.signals[i].fmt for i in idxs}
        offsets = {header.signals[i].byte_offset for i in idxs}
        if len(fmts) != 1 or len(offsets) != 1:
            raise WFDBError(f"Signals stored in {fname} use mixed formats/offsets, which is not supported.")
        fpath = folder / Path(fname).name  # never follow directories named in the header
        if not fpath.exists():
            raise WFDBError(f"Signal file {Path(fname).name!r} referenced by the header was not provided.")
        raw = fpath.read_bytes()[offsets.pop():]
        n_in_file = len(idxs)
        n_values = n_samples * n_in_file if n_samples else None
        vals, missing = _decode(raw, fmts.pop(), n_values)
        if n_samples is None:
            n_samples = len(vals) // n_in_file
        if len(vals) < n_samples * n_in_file:
            raise WFDBError(
                f"Signal file {Path(fname).name!r} is truncated: expected {n_samples * n_in_file} values, "
                f"found {len(vals)}.")
        frame = vals[: n_samples * n_in_file].reshape(n_samples, n_in_file)
        for col, sig_idx in enumerate(idxs):
            columns[sig_idx] = np.where(frame[:, col] == missing, np.nan, frame[:, col]).astype(np.float64)

    if not n_samples:
        raise WFDBError("The record contains no samples.")
    out = np.empty((n_samples, header.n_sig), dtype=np.float32)
    for i, spec in enumerate(header.signals):
        phys = (columns[i] - spec.baseline) / spec.gain
        unit = spec.units.lower().replace("µ", "u")
        if unit in ("uv", "microvolt", "microvolts"):
            phys = phys / 1000.0
        elif unit in ("v", "volt", "volts"):
            phys = phys * 1000.0
        out[:, i] = phys
    return out, header


def write_record_fmt16(folder: str | Path, record_name: str, signal_mv: np.ndarray, fs: float,
                       lead_names: list[str], gain: float = 1000.0) -> Path:
    """Write a record in WFDB format 16 (used by the tests and for exporting processed signals)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sig = np.asarray(signal_mv, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != len(lead_names):
        raise ValueError("signal_mv must be [n_samples, n_leads] matching lead_names")
    digital = np.round(np.nan_to_num(sig, nan=0.0) * gain)
    digital = np.clip(digital, -32767, 32767).astype("<i2")
    digital[np.isnan(sig)] = -32768
    (folder / f"{record_name}.dat").write_bytes(digital.tobytes())
    fs_txt = f"{fs:g}"
    lines = [f"{record_name} {sig.shape[1]} {fs_txt} {sig.shape[0]}"]
    for j, name in enumerate(lead_names):
        lines.append(f"{record_name}.dat 16 {gain:g}(0)/mV 16 0 {int(digital[0, j])} 0 0 {name}")
    hea = folder / f"{record_name}.hea"
    hea.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return hea
