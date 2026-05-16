"""
src/data/preprocessing.py

Signal conditioning for all CPS modalities before feature extraction.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import zscore


# ──────────────────────────────────────────────────────────────────────────────
# Generic bandpass / notch helpers
# ──────────────────────────────────────────────────────────────────────────────

def bandpass_filter(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    sos = sp_signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, x).astype(np.float32)


def notch_filter(
    x: np.ndarray,
    fs: float,
    freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    """Notch filter (default 50 Hz mains)."""
    b, a = sp_signal.iirnotch(freq, q, fs)
    return sp_signal.filtfilt(b, a, x).astype(np.float32)


def lowpass_filter(
    x: np.ndarray,
    fs: float,
    cutoff: float,
    order: int = 4,
) -> np.ndarray:
    sos = sp_signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sp_signal.sosfiltfilt(sos, x).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Modality-specific preprocessing
# ──────────────────────────────────────────────────────────────────────────────

class SnorePreprocessor:
    """
    Condition the raw Schnarc / Druck-Snore channels.

    Pipeline:
    1. DC removal
    2. Notch at 50 Hz (mains hum)
    3. Bandpass 20–2000 Hz  (snoring energy range)
    4. Amplitude normalisation (per-recording RMS)
    """

    def __init__(
        self,
        fs: float = 256.0,
        bandpass_low: float = 20.0,
        bandpass_high: float = 2000.0,  # capped by Nyquist at 256 Hz → 128 Hz
        notch_freq: float = 50.0,
        remove_dc: bool = True,
    ):
        self.fs = fs
        # Clamp high to Nyquist
        self.bp_low = bandpass_low
        self.bp_high = min(bandpass_high, fs / 2 - 1)
        self.notch_freq = notch_freq
        self.remove_dc = remove_dc

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().astype(np.float64)
        if self.remove_dc:
            x -= x.mean()
        if self.notch_freq < self.fs / 2:
            x = notch_filter(x, self.fs, self.notch_freq)
        x = bandpass_filter(x, self.fs, self.bp_low, self.bp_high)
        rms = np.sqrt(np.mean(x ** 2))
        if rms > 1e-9:
            x /= rms
        return x.astype(np.float32)


class EEGPreprocessor:
    """
    Condition EEG channels for spectral analysis.

    Pipeline:
    1. Notch 50 Hz
    2. Bandpass 0.5–40 Hz
    3. Robust z-score (median / IQR) per channel
    4. Optional artefact masking (amplitude threshold)
    """

    def __init__(
        self,
        fs: float = 256.0,
        amp_threshold_uv: float = 500.0,
    ):
        self.fs = fs
        self.amp_threshold = amp_threshold_uv

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().astype(np.float64)
        x = notch_filter(x, self.fs, 50.0)
        x = bandpass_filter(x, self.fs, 0.5, 40.0)
        # Robust normalisation
        med = np.median(x)
        iqr = np.percentile(x, 75) - np.percentile(x, 25)
        if iqr > 1e-9:
            x = (x - med) / iqr
        # Zero-out artefact frames (very high amplitude)
        artefact_mask = np.abs(x) > self.amp_threshold
        x[artefact_mask] = 0.0
        return x.astype(np.float32)


class ECGPreprocessor:
    """
    Condition ECG for R-peak / HRV analysis.

    Pipeline:
    1. Baseline wander removal (highpass 0.5 Hz)
    2. Notch 50 Hz
    3. Lowpass 40 Hz anti-alias
    4. Z-score normalisation
    """

    def __init__(self, fs: float = 256.0):
        self.fs = fs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().astype(np.float64)
        sos_hp = sp_signal.butter(4, 0.5, btype="high", fs=self.fs, output="sos")
        x = sp_signal.sosfiltfilt(sos_hp, x)
        x = notch_filter(x, self.fs, 50.0)
        x = lowpass_filter(x, self.fs, 40.0)
        std = x.std()
        if std > 1e-9:
            x = (x - x.mean()) / std
        return x.astype(np.float32)


class RespiratoryPreprocessor:
    """
    Condition airflow / RIP channels.

    Pipeline:
    1. Highpass 0.05 Hz (removes very slow drift)
    2. Lowpass 5 Hz  (respiration < 2 Hz; keep headroom)
    3. Min-max normalisation to [-1, 1]
    """

    def __init__(self, fs: float = 256.0):
        self.fs = fs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().astype(np.float64)
        sos_hp = sp_signal.butter(2, 0.05, btype="high", fs=self.fs, output="sos")
        x = sp_signal.sosfiltfilt(sos_hp, x)
        x = lowpass_filter(x, self.fs, 5.0, order=2)
        lo, hi = x.min(), x.max()
        span = hi - lo
        if span > 1e-9:
            x = 2 * (x - lo) / span - 1
        return x.astype(np.float32)


class SpO2Preprocessor:
    """
    Clip SpO2 to physiologically valid range [50, 100] %
    and interpolate brief drop-outs.
    """

    VALID_MIN = 50.0
    VALID_MAX = 100.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.copy().astype(np.float32)
        invalid = (x < self.VALID_MIN) | (x > self.VALID_MAX)
        if invalid.any():
            indices = np.arange(len(x))
            valid_indices = indices[~invalid]
            if len(valid_indices) > 1:
                x[invalid] = np.interp(indices[invalid], valid_indices, x[valid_indices])
            else:
                x[invalid] = 95.0  # fallback
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Windowing
# ──────────────────────────────────────────────────────────────────────────────

def extract_windows(
    x: np.ndarray,
    fs: float,
    window_sec: float,
    hop_sec: float,
    start_sample: int = 0,
    end_sample: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-length window over signal *x*.

    Returns
    -------
    windows : ndarray, shape (n_windows, window_samples)
    onset_times : ndarray of float, shape (n_windows,)
        Onset of each window in seconds.
    """
    window_samples = int(window_sec * fs)
    hop_samples    = int(hop_sec * fs)
    if end_sample is None:
        end_sample = len(x)

    x_slice = x[start_sample:end_sample]
    n = len(x_slice)

    windows, onsets = [], []
    pos = 0
    while pos + window_samples <= n:
        windows.append(x_slice[pos : pos + window_samples])
        onsets.append((start_sample + pos) / fs)
        pos += hop_samples

    if not windows:
        return np.empty((0, window_samples), dtype=np.float32), np.array([])

    return np.stack(windows, axis=0).astype(np.float32), np.array(onsets, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Batch preprocessor (wraps all modalities)
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalPreprocessor:
    """
    Apply modality-specific preprocessors to a PatientRecord and return
    a dict of conditioned signal arrays.
    """

    def __init__(self, fs: float = 256.0):
        self.fs = fs
        self.snore_proc  = SnorePreprocessor(fs=fs)
        self.eeg_proc    = EEGPreprocessor(fs=fs)
        self.ecg_proc    = ECGPreprocessor(fs=fs)
        self.resp_proc   = RespiratoryPreprocessor(fs=fs)
        self.spo2_proc   = SpO2Preprocessor()

        self.modality_map: Dict[str, List[str]] = {
            "snore":       ["Snoring_Sound", "Snoring_Pressure", "Schnarc", "Druck Snore"],
            "eeg":         ["C4:A1", "C3:A2", "F4:A1", "O2:A1"],
            "ecg":         ["ECG", "ECG 2"],
            "ppg":         ["PPG", "Pleth"],
            "airflow":     ["Airflow_Pressure", "Thermal_Flow", "Druck Flow", "Flow Th"],
            "rip":         ["RIP_Abdomen", "RIP_Thorax", "RIP.Abdom", "RIP.Thrx"],
            "spo2":        ["SpO2", "SPO2"],
        }

    def process(self, signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        processed: Dict[str, np.ndarray] = {}

        for modality, channel_candidates in self.modality_map.items():
            proc_fn = {
                "snore":   self.snore_proc,
                "eeg":     self.eeg_proc,
                "ecg":     self.ecg_proc,
                "ppg":     self.ecg_proc,   # same pipeline
                "airflow": self.resp_proc,
                "rip":     self.resp_proc,
                "spo2":    self.spo2_proc,
            }[modality]

            for ch in channel_candidates:
                if ch in signals:
                    try:
                        processed[ch] = proc_fn(signals[ch])
                    except Exception as exc:
                        print(f"[WARNING] Preprocessing failed for {ch}: {exc}")
                    break  # take first available channel name

        # Pass through any remaining channels unchanged
        for ch, arr in signals.items():
            if ch not in processed:
                processed[ch] = arr.copy()

        return processed
