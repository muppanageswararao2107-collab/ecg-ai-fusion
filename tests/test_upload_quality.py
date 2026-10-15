import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ecgai.measures import estimate_heart_rate
from ecgai.preprocess import STANDARD_LEADS
from ecgai.quality import FAIL, PASS, WARN, overall_status, run_quality_checks
from ecgai.upload import UploadError, parse_csv_upload, parse_wfdb_upload, prepare_for_model
from ecgai.wfdb_io import write_record_fmt16
from helpers import synthetic_ecg

MAX = 20 * 1024 * 1024
NORM = {"sampling_rate": 100, "length": 1000, "mean": np.zeros(12, np.float32), "std": np.ones(12, np.float32)}


def _wfdb_bytes(tmp_path, x, fs, leads, name="rec"):
    hea = write_record_fmt16(tmp_path, name, x, fs, leads)
    return {f"{name}.hea": hea.read_bytes(), f"{name}.dat": (tmp_path / f"{name}.dat").read_bytes()}


def _csv_bytes(x, leads):
    buf = io.StringIO()
    pd.DataFrame(x, columns=leads).to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------------------------------------------- WFDB
def test_valid_12_lead_wfdb(tmp_path):
    inp = parse_wfdb_upload(_wfdb_bytes(tmp_path, synthetic_ecg(12, 500), 500, STANDARD_LEADS), MAX)
    assert inp.signal.shape == (6000, 12) and inp.fs == 500 and inp.leads == STANDARD_LEADS
    prep = prepare_for_model(inp, NORM, 30)
    assert prep.windows_norm.shape == (1, 1000, 12)
    assert prep.quality_status in (PASS, WARN)
    assert any("Resampled" in n for n in prep.notes)


def test_wfdb_leads_in_other_order_are_reordered(tmp_path):
    x = synthetic_ecg(10, 100)
    order = list(reversed(range(12)))
    inp = parse_wfdb_upload(_wfdb_bytes(tmp_path, x[:, order], 100, [STANDARD_LEADS[i] for i in order]), MAX)
    assert np.allclose(inp.signal, x, atol=1e-3)


def test_single_lead_wfdb_is_rejected(tmp_path):
    files = _wfdb_bytes(tmp_path, synthetic_ecg(10, 360)[:, :1], 360, ["MLII"])
    with pytest.raises(UploadError, match="12"):
        parse_wfdb_upload(files, MAX)


def test_missing_leads_are_named(tmp_path):
    files = _wfdb_bytes(tmp_path, synthetic_ecg(10, 500)[:, :8], 500, STANDARD_LEADS[:8])
    with pytest.raises(UploadError, match="V3"):
        parse_wfdb_upload(files, MAX)


def test_wfdb_needs_header(tmp_path):
    files = _wfdb_bytes(tmp_path, synthetic_ecg(10, 100), 100, STANDARD_LEADS)
    files.pop("rec.hea")
    with pytest.raises(UploadError):
        parse_wfdb_upload(files, MAX)


def test_wfdb_missing_dat(tmp_path):
    files = _wfdb_bytes(tmp_path, synthetic_ecg(10, 100), 100, STANDARD_LEADS)
    files.pop("rec.dat")
    with pytest.raises(UploadError, match="not provided"):
        parse_wfdb_upload(files, MAX)


def test_corrupted_header(tmp_path):
    with pytest.raises(UploadError):
        parse_wfdb_upload({"x.hea": b"\x00\x01garbage", "x.dat": b"\x00\x00"}, MAX)


def test_unsupported_extension_and_empty_and_oversize():
    with pytest.raises(UploadError):
        parse_wfdb_upload({"scan.pdf": b"%PDF"}, MAX)
    with pytest.raises(UploadError, match="empty"):
        parse_wfdb_upload({"r.hea": b""}, MAX)
    with pytest.raises(UploadError, match="limit"):
        parse_wfdb_upload({"r.hea": b"x" * 2000}, 1000)
    with pytest.raises(UploadError):
        parse_wfdb_upload({}, MAX)


def test_temp_files_are_deleted(tmp_path):
    files = _wfdb_bytes(tmp_path, synthetic_ecg(10, 100), 100, STANDARD_LEADS)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    old = tempfile.tempdir
    tempfile.tempdir = str(scratch)
    try:
        parse_wfdb_upload(files, MAX)
        with pytest.raises(UploadError):
            parse_wfdb_upload({"bad.hea": b"rubbish", "bad.dat": b"\x00"}, MAX)
    finally:
        tempfile.tempdir = old
    assert os.listdir(scratch) == []


def test_path_traversal_names_are_neutralised(tmp_path):
    files = _wfdb_bytes(tmp_path, synthetic_ecg(10, 100), 100, STANDARD_LEADS)
    files = {"../../" + k: v for k, v in files.items()}
    inp = parse_wfdb_upload(files, MAX)
    assert inp.filename == "rec.hea"


# ---------------------------------------------------------------- CSV
def test_valid_csv_with_time_column():
    x = synthetic_ecg(10, 250)
    df = pd.DataFrame(x, columns=STANDARD_LEADS)
    df.insert(0, "time", np.arange(len(df)) / 250)
    inp = parse_csv_upload("e.csv", df.to_csv(index=False).encode(), 250, "mV", MAX)
    assert inp.signal.shape == (2500, 12)
    assert any("Ignored" in n for n in inp.notes)


def test_csv_requires_sampling_rate():
    with pytest.raises(UploadError, match="sampling rate"):
        parse_csv_upload("e.csv", _csv_bytes(synthetic_ecg(10, 100), STANDARD_LEADS), None, "mV", MAX)


def test_csv_microvolts_are_converted():
    x = synthetic_ecg(10, 100)
    inp = parse_csv_upload("e.csv", _csv_bytes(x * 1000, STANDARD_LEADS), 100, "µV", MAX)
    assert np.allclose(inp.signal, x, atol=1e-4)


def test_csv_without_header_is_rejected():
    raw = "\n".join(",".join("0.1" for _ in range(12)) for _ in range(1000)).encode()
    with pytest.raises(UploadError, match="header"):
        parse_csv_upload("e.csv", raw, 100, "mV", MAX)


def test_csv_with_missing_values_is_rejected():
    df = pd.DataFrame(synthetic_ecg(10, 100), columns=STANDARD_LEADS).astype(object)
    df.loc[5, "V2"] = np.nan
    df.loc[7, "I"] = "abc"
    with pytest.raises(UploadError, match="missing or non-numeric"):
        parse_csv_upload("e.csv", df.to_csv(index=False).encode(), 100, "mV", MAX)


def test_csv_single_lead_is_rejected():
    raw = pd.DataFrame({"II": synthetic_ecg(10, 100)[:, 1]}).to_csv(index=False).encode()
    with pytest.raises(UploadError, match="12-lead"):
        parse_csv_upload("e.csv", raw, 100, "mV", MAX)


def test_csv_wrong_extension_and_garbage():
    with pytest.raises(UploadError):
        parse_csv_upload("e.txt", b"I,II\n1,2\n", 100, "mV", MAX)
    with pytest.raises(UploadError):
        parse_csv_upload("e.csv", b"\x00\x01\x02", 100, "mV", MAX)


def test_short_recording_is_rejected():
    inp = parse_csv_upload("e.csv", _csv_bytes(synthetic_ecg(6, 100), STANDARD_LEADS), 100, "mV", MAX)
    with pytest.raises(UploadError, match="at least 10"):
        prepare_for_model(inp, NORM, 30)


def test_long_recording_is_windowed():
    inp = parse_csv_upload("e.csv", _csv_bytes(synthetic_ecg(35, 100), STANDARD_LEADS), 100, "mV", MAX)
    prep = prepare_for_model(inp, NORM, 30)
    assert prep.windows_norm.shape == (3, 1000, 12)
    assert any("windows" in n for n in prep.notes) and any("not analysed" in n for n in prep.notes)


def test_prepare_applies_saved_normalisation():
    inp = parse_csv_upload("e.csv", _csv_bytes(synthetic_ecg(10, 100), STANDARD_LEADS), 100, "mV", MAX)
    norm = dict(NORM, mean=np.full(12, 1.0, np.float32), std=np.full(12, 2.0, np.float32))
    prep = prepare_for_model(inp, norm, 30)
    assert np.allclose(prep.windows_norm[0], (prep.signal_100[:1000] - 1.0) / 2.0, atol=1e-5)


# ---------------------------------------------------------------- quality and heart rate
def test_clean_signal_passes_quality():
    checks = run_quality_checks(synthetic_ecg(10, 100), 100, STANDARD_LEADS)
    assert overall_status(checks) == PASS, [c.to_dict() for c in checks]


def test_flat_lead_warns_and_many_flat_fail():
    x = synthetic_ecg(10, 100)
    x[:, 3] = 0
    assert overall_status(run_quality_checks(x, 100, STANDARD_LEADS)) == WARN
    x[:, :4] = 0
    assert overall_status(run_quality_checks(x, 100, STANDARD_LEADS)) == FAIL


def test_nan_fails_quality():
    x = synthetic_ecg(10, 100)
    x[10, 2] = np.nan
    checks = run_quality_checks(x, 100, STANDARD_LEADS)
    assert overall_status(checks) == FAIL and checks[0].name == "Missing values"


def test_strong_noise_warns():
    x = synthetic_ecg(10, 100)
    x[:, 6] += np.random.default_rng(0).normal(0, 0.4, 1000)
    assert any(c.name == "High-frequency noise" and c.status == WARN
               for c in run_quality_checks(x, 100, STANDARD_LEADS))


def test_clipping_warns():
    x = synthetic_ecg(10, 100)
    x[:, 1] = np.clip(x[:, 1], None, 0.2)
    assert any(c.name.startswith("Clipping") and c.status == WARN for c in run_quality_checks(x, 100, STANDARD_LEADS))


def test_wrong_units_fail_amplitude_check():
    checks = run_quality_checks(synthetic_ecg(10, 100) * 1000, 100, STANDARD_LEADS)
    assert any(c.name == "Amplitude range" and c.status == FAIL for c in checks)


def test_heart_rate_estimate_on_regular_rhythm():
    for bpm in (60, 72, 110):
        hr = estimate_heart_rate(synthetic_ecg(10, 100, bpm=bpm), 100, STANDARD_LEADS)
        assert hr["available"] and abs(hr["bpm"] - bpm) < 4, (bpm, hr)


def test_heart_rate_unavailable_on_noise():
    rng = np.random.default_rng(0)
    hr = estimate_heart_rate(rng.standard_normal((1000, 12)).astype(np.float32), 100, STANDARD_LEADS)
    assert not hr["available"]
