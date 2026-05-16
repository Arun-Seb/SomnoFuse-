"""
src/features/acoustic.py

Acoustic feature extraction from the Schnarc (snoring vibration, 256 Hz)
and Druck Snore (snoring pressure, 256 Hz) channels.

Feature groups
--------------
1. Frame-level:  MFCCs, delta-MFCCs, spectral centroid/bandwidth/rolloff/flatness,
                 ZCR, RMS energy, chroma, tonnetz
2. Segment-level: statistics over frame features (mean, std, min, max, median)
3. Mel-spectrogram: 2-D tensor for CNN input
4. Event-level:  from Schnarchen Events.txt (dB values, duration stats)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import librosa
    import librosa.feature
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[WARNING] librosa not found – acoustic features will be unavailable. "
          "Install with: pip install librosa")


# ──────────────────────────────────────────────────────────────────────────────
# Frame-level feature extraction
# ──────────────────────────────────────────────────────────────────────────────

class AcousticFrameExtractor:
    """
    Extract per-frame acoustic features from a snoring waveform.

    Parameters
    ----------
    fs : int
        Sampling rate (Hz).
    n_mfcc : int
        Number of MFCC coefficients.
    n_fft : int
        FFT window size (samples).
    hop_length : int
        Hop between frames (samples).
    n_mels : int
        Number of mel filterbanks.
    fmin, fmax : float
        Mel filterbank frequency bounds.
    """

    FEATURE_NAMES = []  # populated in __init__

    def __init__(
        self,
        fs: int = 256,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 64,
        n_mels: int = 64,
        fmin: float = 20.0,
        fmax: float = 125.0,   # Nyquist at 256 Hz = 128 Hz
    ):
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for acoustic features.")

        self.fs = fs
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = min(fmax, fs / 2 - 1)

        # Build feature name list
        names = []
        for i in range(n_mfcc):
            names += [f"mfcc_{i:02d}", f"delta_mfcc_{i:02d}", f"delta2_mfcc_{i:02d}"]
        names += [
            "spectral_centroid", "spectral_bandwidth",
            "spectral_rolloff_85", "spectral_flatness",
            "zcr", "rms_energy",
        ]
        self.FEATURE_NAMES = names

    @property
    def n_features(self) -> int:
        return len(self.FEATURE_NAMES)

    def extract_frames(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract frame-level features.

        Returns
        -------
        features : ndarray, shape (n_frames, n_features)
        """
        y = waveform.astype(np.float32)

        # MFCCs + deltas
        mfccs = librosa.feature.mfcc(
            y=y, sr=self.fs, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
        )
        delta_mfcc  = librosa.feature.delta(mfccs)
        delta2_mfcc = librosa.feature.delta(mfccs, order=2)

        mfcc_stack = np.vstack([mfccs, delta_mfcc, delta2_mfcc])  # (3*n_mfcc, T)

        # Spectral features
        centroid   = librosa.feature.spectral_centroid(
            y=y, sr=self.fs, n_fft=self.n_fft, hop_length=self.hop_length)
        bandwidth  = librosa.feature.spectral_bandwidth(
            y=y, sr=self.fs, n_fft=self.n_fft, hop_length=self.hop_length)
        rolloff    = librosa.feature.spectral_rolloff(
            y=y, sr=self.fs, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=0.85)
        flatness   = librosa.feature.spectral_flatness(
            y=y, n_fft=self.n_fft, hop_length=self.hop_length)
        zcr        = librosa.feature.zero_crossing_rate(
            y=y, hop_length=self.hop_length)
        rms        = librosa.feature.rms(
            y=y, frame_length=self.n_fft, hop_length=self.hop_length)

        spectral_stack = np.vstack([centroid, bandwidth, rolloff, flatness, zcr, rms])
        # shape: (6, T)

        # Combine → (n_features, T) → (T, n_features)
        all_features = np.vstack([mfcc_stack, spectral_stack]).T  # (T, F)
        return all_features.astype(np.float32)

    def extract_stats(self, waveform: np.ndarray) -> np.ndarray:
        """
        Return summary statistics (mean, std, min, max, median) over frames.
        Shape: (n_features * 5,)
        """
        frames = self.extract_frames(waveform)   # (T, F)
        return np.concatenate([
            frames.mean(axis=0),
            frames.std(axis=0),
            frames.min(axis=0),
            frames.max(axis=0),
            np.median(frames, axis=0),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Mel-spectrogram for CNN input
# ──────────────────────────────────────────────────────────────────────────────

class MelSpectrogramExtractor:
    """
    Produce a (n_mels, T) log-Mel spectrogram tensor suitable for 2-D CNNs.
    """

    def __init__(
        self,
        fs: int = 256,
        n_fft: int = 512,
        hop_length: int = 64,
        n_mels: int = 64,
        fmin: float = 20.0,
        fmax: float = 125.0,
        top_db: float = 80.0,
    ):
        if not HAS_LIBROSA:
            raise ImportError("librosa is required.")
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = min(fmax, fs / 2 - 1)
        self.top_db = top_db

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        log_mel : ndarray, shape (n_mels, n_frames)
        """
        y = waveform.astype(np.float32)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.fs,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max, top_db=self.top_db)
        # Normalise to [0, 1]
        log_mel = (log_mel + self.top_db) / self.top_db
        return log_mel.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Event-level statistics from Schnarchen Events.txt
# ──────────────────────────────────────────────────────────────────────────────

def snoring_event_stats(
    snoring_events: pd.DataFrame,
    window_start: float,
    window_end: float,
) -> Dict[str, float]:
    """
    Compute aggregate snoring-event statistics within a time window.

    Parameters
    ----------
    snoring_events : DataFrame with columns onset_sec, duration_sec, label, value
    window_start, window_end : float — seconds

    Returns
    -------
    dict of scalar features
    """
    if snoring_events is None or snoring_events.empty:
        return _zero_event_stats()

    mask = (
        (snoring_events["onset_sec"] >= window_start) &
        (snoring_events["onset_sec"] <  window_end)
    )
    ev = snoring_events[mask]

    snore_ev  = ev[ev["label"] == "Snoring_Sounds"]
    breath_ev = ev[ev["label"] == "Breathing_Sounds"]

    window_dur = window_end - window_start

    def _stats_for(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
        if df.empty:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_total_dur": 0.0,
                f"{prefix}_mean_dur": 0.0,
                f"{prefix}_mean_db": 0.0,
                f"{prefix}_max_db": 0.0,
                f"{prefix}_fraction": 0.0,
            }
        durations = df["duration_sec"].values
        dbs = df["value"].values
        total_dur = float(durations.sum())
        return {
            f"{prefix}_count":     float(len(df)),
            f"{prefix}_total_dur": total_dur,
            f"{prefix}_mean_dur":  float(durations.mean()),
            f"{prefix}_mean_db":   float(np.nanmean(dbs)) if len(dbs) else 0.0,
            f"{prefix}_max_db":    float(np.nanmax(dbs))  if len(dbs) else 0.0,
            f"{prefix}_fraction":  total_dur / window_dur if window_dur > 0 else 0.0,
        }

    stats = {}
    stats.update(_stats_for(snore_ev,  "snore"))
    stats.update(_stats_for(breath_ev, "breath"))
    stats["snore_event_rate"] = len(snore_ev) / window_dur if window_dur > 0 else 0.0
    return stats


def _zero_event_stats() -> Dict[str, float]:
    keys = [
        "snore_count", "snore_total_dur", "snore_mean_dur", "snore_mean_db",
        "snore_max_db", "snore_fraction",
        "breath_count", "breath_total_dur", "breath_mean_dur", "breath_mean_db",
        "breath_max_db", "breath_fraction",
        "snore_event_rate",
    ]
    return {k: 0.0 for k in keys}


# ──────────────────────────────────────────────────────────────────────────────
# Snoring-detection threshold (simple energy-based)
# ──────────────────────────────────────────────────────────────────────────────

def detect_snoring_frames(
    waveform: np.ndarray,
    fs: int = 256,
    frame_len_sec: float = 0.5,
    hop_sec: float = 0.1,
    energy_percentile: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple energy-threshold snoring detector.

    Returns
    -------
    onset_times : ndarray of float (seconds)
    is_snoring  : bool ndarray
    """
    frame_len = int(frame_len_sec * fs)
    hop = int(hop_sec * fs)

    frames, onsets, energies = [], [], []
    pos = 0
    while pos + frame_len <= len(waveform):
        frame = waveform[pos : pos + frame_len]
        energies.append(np.sqrt(np.mean(frame ** 2)))
        onsets.append(pos / fs)
        pos += hop

    energies = np.array(energies)
    threshold = np.percentile(energies, energy_percentile)
    is_snoring = energies > threshold

    return np.array(onsets), is_snoring
