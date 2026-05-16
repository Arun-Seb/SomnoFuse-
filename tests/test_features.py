"""
tests/test_features.py
Quick unit tests that don't require the actual dataset.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocessing import (
    SnorePreprocessor, EEGPreprocessor, ECGPreprocessor,
    RespiratoryPreprocessor, SpO2Preprocessor, extract_windows
)
from src.features.physiological import (
    bandpower, eeg_band_features, hrv_features, spo2_features
)
from src.features.respiratory import (
    estimate_respiratory_rate, airflow_waveform_features
)
from src.features.fusion import concatenate_features, LateFusion


FS = 256
DURATION = 30  # seconds
N = FS * DURATION


# ── Preprocessing ─────────────────────────────────────────────────────────────

def test_snore_preprocessor():
    x = np.random.randn(N).astype(np.float32)
    proc = SnorePreprocessor(fs=FS)
    out  = proc(x)
    assert out.shape == x.shape
    assert not np.isnan(out).any()


def test_eeg_preprocessor():
    x = np.random.randn(N).astype(np.float32) * 50   # uV scale
    proc = EEGPreprocessor(fs=FS)
    out  = proc(x)
    assert out.shape == x.shape


def test_ecg_preprocessor():
    t = np.linspace(0, DURATION, N)
    x = (np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(N)).astype(np.float32)
    proc = ECGPreprocessor(fs=FS)
    out  = proc(x)
    assert out.shape == x.shape
    assert not np.isnan(out).any()


def test_spo2_preprocessor():
    spo2 = np.array([98, 97, 200, 95, -5, 96, 94], dtype=np.float32)
    proc = SpO2Preprocessor()
    out  = proc(spo2)
    assert (out >= 50).all() and (out <= 100).all()


def test_extract_windows():
    x = np.random.randn(N).astype(np.float32)
    windows, onsets = extract_windows(x, fs=FS, window_sec=5.0, hop_sec=2.5)
    assert windows.shape[1] == FS * 5
    assert len(onsets) == windows.shape[0]


# ── EEG features ─────────────────────────────────────────────────────────────

def test_bandpower():
    t = np.linspace(0, DURATION, N)
    x = np.sin(2 * np.pi * 10 * t)   # 10 Hz pure tone → alpha band
    bp_alpha = bandpower(x, fs=FS, low=8.0, high=13.0)
    bp_delta = bandpower(x, fs=FS, low=0.5, high=4.0)
    assert bp_alpha > bp_delta


def test_eeg_band_features():
    x = np.random.randn(N)
    feats = eeg_band_features(x, fs=FS)
    assert "total_power" in feats
    assert "rel_delta"   in feats
    assert "hjorth_activity" in feats
    for v in feats.values():
        assert np.isfinite(v)


# ── HRV ──────────────────────────────────────────────────────────────────────

def test_hrv_features_synthetic():
    """Synthesise a clean ECG-like signal with R-peaks at ~1 Hz."""
    t = np.linspace(0, DURATION, N)
    ecg = np.zeros(N, dtype=np.float32)
    for pk in range(1, DURATION):
        idx = int(pk * FS)
        ecg[max(0, idx-3):idx+3] += np.array([0.1, 0.5, 1.0, 0.5, 0.1, 0.0])[:min(6, N-max(0,idx-3))]
    ecg += 0.01 * np.random.randn(N)
    feats = hrv_features(ecg, fs=FS)
    assert "nn_mean" in feats


# ── SpO₂ ─────────────────────────────────────────────────────────────────────

def test_spo2_features():
    spo2 = np.array([98, 97, 95, 93, 91, 89, 88, 90, 92, 95], dtype=np.float32)
    feats = spo2_features(spo2, fs=1.0)
    assert feats["t90"] > 0
    assert feats["spo2_min"] < 95


# ── Respiratory ───────────────────────────────────────────────────────────────

def test_respiratory_rate():
    t = np.linspace(0, DURATION, N)
    # Simulate airflow at 15 breaths/min = 0.25 Hz
    airflow = np.sin(2 * np.pi * 0.25 * t).astype(np.float32)
    rr = estimate_respiratory_rate(airflow, fs=FS)
    assert abs(rr - 15.0) < 5.0, f"RR estimate too far off: {rr}"


def test_airflow_features():
    t = np.linspace(0, DURATION, N)
    airflow = (np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.random.randn(N)).astype(np.float32)
    feats = airflow_waveform_features(airflow, fs=FS)
    assert "resp_rate_bpm" in feats
    assert "flow_std" in feats


# ── Fusion ────────────────────────────────────────────────────────────────────

def test_concatenate_features():
    d1 = {"a": 1.0, "b": 2.0}
    d2 = {"c": 3.0, "d": float("nan")}
    vec, names = concatenate_features([d1, d2])
    assert len(vec) == 4
    assert not np.isnan(vec).any()
    assert names == ["a", "b", "c", "d"]


def test_late_fusion_mean():
    p1 = np.array([0.7, 0.2, 0.1])
    p2 = np.array([0.5, 0.3, 0.2])
    fused = LateFusion("mean")([p1, p2])
    np.testing.assert_allclose(fused, [0.6, 0.25, 0.15])


def test_late_fusion_weighted():
    p1 = np.array([1.0, 0.0])
    p2 = np.array([0.0, 1.0])
    fused = LateFusion("weighted", weights=[3, 1])([p1, p2])
    assert fused[0] > fused[1]
