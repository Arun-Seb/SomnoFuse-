"""
src/features/physiological.py

Feature extraction from EEG, ECG/HRV, PPG, and SpO₂ channels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis


# ──────────────────────────────────────────────────────────────────────────────
# EEG spectral features
# ──────────────────────────────────────────────────────────────────────────────

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (11.0, 16.0),   # sleep spindle band
    "beta":  (13.0, 30.0),
}


def bandpower(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    method: str = "welch",
) -> float:
    """Compute absolute band power using Welch's method."""
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    mask = (freqs >= low) & (freqs <= high)
    bp = np.trapezoid(psd[mask], freqs[mask])
    return float(bp)


def eeg_band_features(
    eeg_epoch: np.ndarray,
    fs: float = 256.0,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Compute absolute and relative band power, plus band-power ratios.

    Parameters
    ----------
    eeg_epoch : 1-D ndarray, one EEG channel, one 30-s epoch
    fs : float

    Returns
    -------
    dict of feature_name → float
    """
    if bands is None:
        bands = EEG_BANDS

    total_power = bandpower(eeg_epoch, fs, 0.5, 40.0)
    features: Dict[str, float] = {"total_power": total_power}

    band_powers: Dict[str, float] = {}
    for band_name, (lo, hi) in bands.items():
        bp = bandpower(eeg_epoch, fs, lo, hi)
        band_powers[band_name] = bp
        features[f"abs_{band_name}"] = bp
        features[f"rel_{band_name}"] = bp / (total_power + 1e-12)

    # Clinically useful ratios
    features["theta_alpha_ratio"] = (
        band_powers.get("theta", 0) / (band_powers.get("alpha", 1e-12) + 1e-12)
    )
    features["delta_ratio"] = (
        band_powers.get("delta", 0) / (total_power + 1e-12)
    )
    features["sigma_delta_ratio"] = (
        band_powers.get("sigma", 0) / (band_powers.get("delta", 1e-12) + 1e-12)
    )

    # Spectral edge frequency (95 %)
    freqs, psd = sp_signal.welch(eeg_epoch, fs=fs, nperseg=min(len(eeg_epoch), int(4 * fs)))
    cumulative = np.cumsum(psd)
    cumulative /= cumulative[-1] + 1e-12
    idx = np.searchsorted(cumulative, 0.95)
    features["spectral_edge_95"] = float(freqs[min(idx, len(freqs) - 1)])

    # Hjorth parameters
    diff1 = np.diff(eeg_epoch)
    diff2 = np.diff(diff1)
    activity   = float(np.var(eeg_epoch))
    mobility   = float(np.sqrt(np.var(diff1) / (np.var(eeg_epoch) + 1e-12)))
    complexity = float(
        np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12)
    )
    features.update({"hjorth_activity": activity,
                     "hjorth_mobility": mobility,
                     "hjorth_complexity": complexity})

    return features


def multi_channel_eeg_features(
    channels: Dict[str, np.ndarray],
    fs: float = 256.0,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Aggregate EEG features across multiple channels."""
    if channel_names is None:
        channel_names = ["C4:A1", "C3:A2", "F4:A1", "O2:A1"]

    all_features: Dict[str, float] = {}
    for ch in channel_names:
        if ch in channels:
            ch_feats = eeg_band_features(channels[ch], fs)
            for k, v in ch_feats.items():
                all_features[f"{ch}_{k}"] = v
    return all_features


# ──────────────────────────────────────────────────────────────────────────────
# ECG / HRV features
# ──────────────────────────────────────────────────────────────────────────────

def detect_r_peaks(ecg: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Simple Pan-Tompkins-inspired R-peak detector.
    Returns indices of R peaks in *ecg*.
    """
    # Differentiate → square → integrate
    diff_ecg = np.diff(ecg, prepend=ecg[0])
    squared  = diff_ecg ** 2

    win = int(0.15 * fs)   # 150 ms integration window
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode="same")

    # Find peaks with minimum distance = 0.3 s (200 bpm max)
    min_dist = int(0.3 * fs)
    peaks, _ = sp_signal.find_peaks(integrated, distance=min_dist,
                                     height=np.percentile(integrated, 75))
    return peaks


def hrv_features(
    ecg: np.ndarray,
    fs: float = 256.0,
    min_rr_sec: float = 0.3,
    max_rr_sec: float = 2.0,
) -> Dict[str, float]:
    """
    Time-domain and frequency-domain HRV features from an ECG segment.

    Returned keys: nn_mean, nn_std (SDNN), rmssd, pnn50,
                   lf_power, hf_power, lf_hf_ratio, sdsd
    """
    r_peaks = detect_r_peaks(ecg, fs)
    if len(r_peaks) < 3:
        return {k: np.nan for k in [
            "nn_mean", "nn_std", "rmssd", "pnn50",
            "lf_power", "hf_power", "lf_hf_ratio", "sdsd"]}

    rr = np.diff(r_peaks) / fs          # RR intervals in seconds
    valid = (rr > min_rr_sec) & (rr < max_rr_sec)
    rr = rr[valid]
    if len(rr) < 2:
        return {k: np.nan for k in [
            "nn_mean", "nn_std", "rmssd", "pnn50",
            "lf_power", "hf_power", "lf_hf_ratio", "sdsd"]}

    diff_rr = np.diff(rr)

    # Time domain
    nn_mean = float(rr.mean())
    nn_std  = float(rr.std())
    rmssd   = float(np.sqrt(np.mean(diff_rr ** 2)))
    pnn50   = float(np.mean(np.abs(diff_rr) > 0.05))
    sdsd    = float(diff_rr.std())

    # Frequency domain (interpolate to uniform grid at 4 Hz)
    t_rr = np.cumsum(rr)
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1 / 4.0)
    if len(t_uniform) < 4:
        lf = hf = lf_hf = np.nan
    else:
        rr_uniform = np.interp(t_uniform, t_rr, rr)
        freqs, psd = sp_signal.welch(rr_uniform, fs=4.0,
                                      nperseg=min(len(rr_uniform), 256))
        lf = float(np.trapezoid(psd[(freqs >= 0.04) & (freqs <= 0.15)],
                             freqs[(freqs >= 0.04) & (freqs <= 0.15)]))
        hf = float(np.trapezoid(psd[(freqs >= 0.15) & (freqs <= 0.40)],
                             freqs[(freqs >= 0.15) & (freqs <= 0.40)]))
        lf_hf = float(lf / (hf + 1e-12))

    return {
        "nn_mean": nn_mean, "nn_std": nn_std, "rmssd": rmssd,
        "pnn50": pnn50, "lf_power": lf, "hf_power": hf,
        "lf_hf_ratio": lf_hf, "sdsd": sdsd,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SpO₂ features
# ──────────────────────────────────────────────────────────────────────────────

def spo2_features(spo2: np.ndarray, fs: float = 4.0) -> Dict[str, float]:
    """
    Features from SpO₂ signal within one 30-s epoch (or any window).

    Returns mean, min, std, T90 (% time below 90 %), ODI (desaturation events).
    """
    if len(spo2) == 0:
        return {k: np.nan for k in
                ["spo2_mean", "spo2_min", "spo2_std", "t90", "odi_approx"]}

    mean_spo2 = float(np.nanmean(spo2))
    min_spo2  = float(np.nanmin(spo2))
    std_spo2  = float(np.nanstd(spo2))
    t90       = float(np.mean(spo2 < 90.0))   # fraction

    # Approximate ODI: count 4 % drops
    drops = 0
    baseline = spo2[0]
    in_desat = False
    for val in spo2:
        if not in_desat and val < baseline - 4.0:
            drops += 1
            in_desat = True
        elif in_desat and val >= baseline - 2.0:
            in_desat = False
            baseline = val
        else:
            baseline = max(baseline, val)

    duration_min = len(spo2) / fs / 60.0
    odi = drops / duration_min if duration_min > 0 else 0.0

    return {
        "spo2_mean": mean_spo2,
        "spo2_min":  min_spo2,
        "spo2_std":  std_spo2,
        "t90":       t90,
        "odi_approx": float(odi),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generic waveform statistics (fallback for any channel)
# ──────────────────────────────────────────────────────────────────────────────

def waveform_stats(x: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Return basic statistics over waveform x."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mean":     float(np.nanmean(x)),
        f"{p}std":      float(np.nanstd(x)),
        f"{p}min":      float(np.nanmin(x)),
        f"{p}max":      float(np.nanmax(x)),
        f"{p}p25":      float(np.nanpercentile(x, 25)),
        f"{p}p75":      float(np.nanpercentile(x, 75)),
        f"{p}skewness": float(skew(x[~np.isnan(x)])) if len(x) > 2 else 0.0,
        f"{p}kurtosis": float(kurtosis(x[~np.isnan(x)])) if len(x) > 2 else 0.0,
    }
