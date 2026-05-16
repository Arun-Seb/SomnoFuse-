"""
src/features/respiratory.py

Features derived from airflow, RIP, and annotation-based event counting
(apnea, hypopnea, RERA, flow limitation).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sp_signal


# ──────────────────────────────────────────────────────────────────────────────
# Respiratory rate
# ──────────────────────────────────────────────────────────────────────────────

def estimate_respiratory_rate(
    airflow: np.ndarray,
    fs: float = 256.0,
    freq_min: float = 0.1,   # 6 breaths/min
    freq_max: float = 0.67,  # 40 breaths/min
) -> float:
    """
    Estimate respiratory rate (breaths/min) from an airflow signal
    using the dominant frequency in the respiratory band.
    """
    nperseg = min(len(airflow), int(fs * 30))
    freqs, psd = sp_signal.welch(airflow, fs=fs, nperseg=nperseg)
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not mask.any():
        return np.nan
    dominant_freq = freqs[mask][np.argmax(psd[mask])]
    return float(dominant_freq * 60)


# ──────────────────────────────────────────────────────────────────────────────
# RIP-based thoraco-abdominal synchrony
# ──────────────────────────────────────────────────────────────────────────────

def thoracoabdominal_features(
    rip_thorax: np.ndarray,
    rip_abdomen: np.ndarray,
    fs: float = 32.0,        # RIP channels are 32 Hz native
) -> Dict[str, float]:
    """
    Phase angle and synchrony between thoracic and abdominal RIP signals.

    Returns
    -------
    dict with keys: phase_angle_mean, phase_angle_std, sync_ratio, paradox_fraction
    """
    if len(rip_thorax) != len(rip_abdomen) or len(rip_thorax) == 0:
        return {k: np.nan for k in
                ["phase_angle_mean", "phase_angle_std", "sync_ratio", "paradox_fraction"]}

    # Cross-correlation to find phase lag
    corr = np.correlate(rip_thorax - rip_thorax.mean(),
                        rip_abdomen - rip_abdomen.mean(), mode="full")
    lag = (np.argmax(corr) - (len(rip_thorax) - 1)) / fs
    phase_angle = float(np.degrees(2 * np.pi * lag * estimate_respiratory_rate(
        rip_thorax, fs) / 60))

    # Paradoxical breathing: thorax and abdomen move in opposite directions
    sum_rip = rip_thorax + rip_abdomen
    paradox_fraction = float(np.mean(np.sign(np.diff(rip_thorax)) !=
                                      np.sign(np.diff(rip_abdomen))))
    sync_ratio = float(np.std(sum_rip) / (np.std(rip_thorax) + np.std(rip_abdomen) + 1e-12))

    return {
        "phase_angle_mean":  phase_angle,
        "phase_angle_std":   float(np.std(corr)),
        "sync_ratio":        sync_ratio,
        "paradox_fraction":  paradox_fraction,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Event-based AHI / respiratory indices from annotation DataFrames
# ──────────────────────────────────────────────────────────────────────────────

APNEA_LABELS = {
    "Central_Apnea", "Obstructive_Apnea", "Mixed_Apnea",
    "Central_Hypopnea", "Hypopnea",
}

RESPIRATORY_AROUSAL_LABELS = {
    "Respiratory_Arousal", "Respiratory_Arousal_EEG",
    "Flow_Limitation_Arousal", "Flow_Limitation_Arousal_EEG",
    "Snoring_Arousal", "Snoring_Arousal_EEG",
}


def respiratory_event_features(
    flow_events: Optional[pd.DataFrame],
    arousals: Optional[pd.DataFrame],
    window_start: float,
    window_end: float,
) -> Dict[str, float]:
    """
    Count and characterise respiratory events in a time window.

    Returns AHI-related counts (per-hour), mean durations, etc.
    """
    window_dur_hr = (window_end - window_start) / 3600.0
    window_dur_sec = window_end - window_start

    def _window_mask(df: pd.DataFrame) -> pd.DataFrame:
        return df[
            (df["onset_sec"] >= window_start) & (df["onset_sec"] < window_end)
        ]

    feat: Dict[str, float] = {}

    # ── Flow events ───────────────────────────────────────────────────────────
    if flow_events is not None and not flow_events.empty:
        fe = _window_mask(flow_events)

        for label in ["Central_Apnea", "Obstructive_Apnea", "Mixed_Apnea",
                      "Hypopnea", "Central_Hypopnea", "RERA", "Flow_Limitation"]:
            sub = fe[fe["label"] == label]
            feat[f"{label.lower()}_count"]    = float(len(sub))
            feat[f"{label.lower()}_dur_mean"] = (
                float(sub["duration_sec"].mean()) if not sub.empty else 0.0
            )

        apnea_events = fe[fe["label"].isin(APNEA_LABELS)]
        feat["ahi_approx"] = (
            len(apnea_events) / window_dur_hr if window_dur_hr > 0 else 0.0
        )
        feat["total_apnea_sec"] = float(apnea_events["duration_sec"].sum())
        feat["apnea_burden"] = feat["total_apnea_sec"] / window_dur_sec if window_dur_sec > 0 else 0.0
    else:
        for k in ["ahi_approx", "total_apnea_sec", "apnea_burden",
                  "central_apnea_count", "obstructive_apnea_count",
                  "hypopnea_count", "rera_count"]:
            feat[k] = 0.0

    # ── Respiratory arousals ──────────────────────────────────────────────────
    if arousals is not None and not arousals.empty:
        ar = _window_mask(arousals)
        resp_ar = ar[ar["label"].isin(RESPIRATORY_AROUSAL_LABELS)]
        feat["resp_arousal_count"] = float(len(resp_ar))
        feat["resp_arousal_index"] = (
            len(resp_ar) / window_dur_hr if window_dur_hr > 0 else 0.0
        )
    else:
        feat["resp_arousal_count"] = 0.0
        feat["resp_arousal_index"] = 0.0

    return feat


# ──────────────────────────────────────────────────────────────────────────────
# Waveform-level airflow features
# ──────────────────────────────────────────────────────────────────────────────

def airflow_waveform_features(
    airflow: np.ndarray,
    fs: float = 256.0,
) -> Dict[str, float]:
    """
    Statistical and spectral features from an airflow waveform segment.
    """
    rr = estimate_respiratory_rate(airflow, fs)
    peaks, _ = sp_signal.find_peaks(airflow, distance=int(fs * 1.5))
    troughs, _ = sp_signal.find_peaks(-airflow, distance=int(fs * 1.5))

    amplitude = np.nan
    if len(peaks) > 0 and len(troughs) > 0:
        amplitude = float(np.mean(airflow[peaks]) - np.mean(airflow[troughs]))

    return {
        "resp_rate_bpm":        rr,
        "flow_amplitude_mean":  amplitude,
        "flow_std":             float(np.std(airflow)),
        "flow_skewness":        float(np.mean((airflow - airflow.mean()) ** 3) /
                                      (airflow.std() ** 3 + 1e-12)),
        "flow_peak_count":      float(len(peaks)),
        "flow_trough_count":    float(len(troughs)),
    }
