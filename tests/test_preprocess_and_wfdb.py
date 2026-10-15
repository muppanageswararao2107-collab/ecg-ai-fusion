import numpy as np
import pytest

from ecgai.preprocess import (STANDARD_LEADS, fit_normalization, load_normalization, make_windows,
                              normalize_lead_name, resample, save_normalization, standardize)
from ecgai.wfdb_io import WFDBError, _decode, parse_header, read_record, write_record_fmt16
from helpers import synthetic_ecg


def test_lead_name_normalisation():
    assert normalize_lead_name("avr") == "aVR"
    assert normalize_lead_name(" V1 ") == "V1"
    assert normalize_lead_name("Lead II") == "II"
    assert normalize_lead_name("MLII") is None
    assert normalize_lead_name("foo") is None


def test_resample_500_to_100_keeps_duration():
    x = synthetic_ecg(10, 500)
    y = resample(x, 500, 100)
    assert y.shape == (1000, 12)


def test_resample_360_to_100():
    x = synthetic_ecg(10, 360)
    assert resample(x, 360, 100).shape == (1000, 12)


def test_windows_and_remainder():
    x = np.zeros((2550, 12), np.float32)
    w, dropped, trunc = make_windows(x, 1000, max_windows=5)
    assert w.shape == (2, 1000, 12) and dropped == 550 and not trunc
    with pytest.raises(ValueError):
        make_windows(np.zeros((999, 12)), 1000)
    w, _, trunc = make_windows(np.zeros((10000, 12)), 1000, max_windows=3)
    assert w.shape[0] == 3 and trunc


def test_normalisation_fitted_in_chunks_matches_numpy(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(2.0, 3.0, size=(50, 100, 12)).astype(np.float32)
    mean, std = fit_normalization(X, chunk=7)
    assert np.allclose(mean, X.mean(axis=(0, 1)), atol=1e-4)
    assert np.allclose(std, X.std(axis=(0, 1)), atol=1e-3)
    save_normalization(tmp_path / "n.json", mean, std, STANDARD_LEADS, 100, 1000, "train", 50)
    n = load_normalization(tmp_path / "n.json")
    z = standardize(X, n["mean"], n["std"])
    assert abs(float(z.mean())) < 1e-3


def test_normalisation_uses_given_stats_not_input_stats():
    x = np.full((1000, 12), 5.0, np.float32)
    z = standardize(x, np.zeros(12, np.float32), np.full(12, 2.0, np.float32))
    assert np.allclose(z, 2.5)  # refitting on the input would have produced zeros


def test_wfdb_fmt16_roundtrip(tmp_path):
    x = synthetic_ecg(10, 500)
    hea = write_record_fmt16(tmp_path, "rec", x, 500, STANDARD_LEADS)
    y, header = read_record(hea)
    assert header.fs == 500 and y.shape == x.shape
    assert np.abs(y - x).max() < 1e-3
    assert [s.description for s in header.signals] == STANDARD_LEADS


def test_wfdb_header_fields():
    h = parse_header("r1 2 360 650000\nr1.dat 212 200(1024)/mV 11 1024 995 21537 0 MLII\n"
                     "r1.dat 212 200/uV 11 1024 1011 20052 0 V5\n# comment\n")
    assert h.fs == 360 and h.n_samples == 650000
    assert h.signals[0].baseline == 1024 and h.signals[1].baseline == 1024 and h.signals[1].units == "uV"
    assert h.comments == ["comment"]


def test_wfdb_212_decoding():
    # two 12-bit samples: 0x123 (291) and -1 (0xFFF) packed as bytes 0x23, 0xF1, 0xFF
    vals, _ = _decode(bytes([0x23, 0xF1, 0xFF]), 212, None)
    assert vals.tolist() == [291, -1]


def test_wfdb_rejects_unsupported_format():
    with pytest.raises(WFDBError):
        parse_header("r 1 100 10\nr.dat 310 200 12 0 0 0 0 I\n")


def test_wfdb_missing_signal_file(tmp_path):
    (tmp_path / "r.hea").write_text("r 1 100 10\nr.dat 16 200 16 0 0 0 0 I\n")
    with pytest.raises(WFDBError):
        read_record(tmp_path / "r.hea")


def test_wfdb_truncated_signal_file(tmp_path):
    x = synthetic_ecg(2, 100)
    hea = write_record_fmt16(tmp_path, "r", x, 100, STANDARD_LEADS)
    dat = tmp_path / "r.dat"
    dat.write_bytes(dat.read_bytes()[:100])
    with pytest.raises(WFDBError):
        read_record(hea)
