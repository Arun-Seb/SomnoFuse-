"""
src/data/dataset.py

PyTorch Dataset that yields fixed-length multimodal windows
with their arousal / sleep-stage labels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.loader import (
    CPSLoader, PatientRecord, events_in_window, get_sleep_period
)
from src.data.preprocessing import MultimodalPreprocessor, extract_windows
from src.features.acoustic import (
    AcousticFrameExtractor, MelSpectrogramExtractor, snoring_event_stats
)
from src.features.physiological import (
    multi_channel_eeg_features, hrv_features, spo2_features, waveform_stats
)
from src.features.respiratory import respiratory_event_features, airflow_waveform_features
from src.utils.config import ExperimentConfig


# ──────────────────────────────────────────────────────────────────────────────
# Arousal label encoding
# ──────────────────────────────────────────────────────────────────────────────

AROUSAL_LABEL_MAP = {
    "background":                  0,
    "Respiratory_Arousal":         1,
    "Respiratory_Arousal_EEG":     1,
    "Flow_Limitation_Arousal":     2,
    "Flow_Limitation_Arousal_EEG": 2,
    "SpO2_Arousal_EEG":            3,
    "LM_Arousal":                  4,
    "LM_Arousal_EEG":              4,
    "PLM_Arousal":                 5,
    "PLM_Arousal_EEG":             5,
    "Snoring_Arousal":             6,
    "Snoring_Arousal_EEG":         6,
    "Spontaneous_Arousal":         7,
    "Spontaneous_Arousal_EEG":     7,
    "Autonomic_Arousal":           8,
}

AROUSAL_CLASS_NAMES = [
    "Background", "Respiratory", "FlowLimitation", "SpO2",
    "LimbMovement", "PLM", "Snoring", "Spontaneous", "Autonomic"
]

SLEEP_STAGE_MAP = {
    "Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "Artifact": -1
}


def label_window(
    arousals,
    window_start: float,
    window_end: float,
) -> int:
    """Return the dominant arousal label (int) for a window."""
    if arousals is None or arousals.empty:
        return 0
    ev = events_in_window(arousals, window_start, window_end)
    if ev.empty:
        return 0
    # Majority vote; first event wins ties
    counts = ev["label"].map(lambda l: AROUSAL_LABEL_MAP.get(l, 0)).value_counts()
    return int(counts.idxmax())


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class CPSWindowDataset(Dataset):
    """
    Yields one 30-second window per __getitem__ call.

    Each item is a dict:
    {
        "mel_snore"    : Tensor (n_mels, T_frames) – Mel-spectrogram of snoring channel
        "physio_vec"   : Tensor (n_physio_features,) – hand-crafted physiological features
        "snore_vec"    : Tensor (n_snore_features,) – hand-crafted snoring features
        "resp_vec"     : Tensor (n_resp_features,)  – respiratory features
        "label"        : int – arousal class
        "sleep_stage"  : int – sleep stage (0=Wake … 4=REM)
        "patient_id"   : str
        "onset_sec"    : float
    }
    """

    def __init__(
        self,
        records: List[PatientRecord],
        cfg: ExperimentConfig,
        augment: bool = False,
    ):
        self.cfg     = cfg
        self.augment = augment

        self.mel_extractor   = MelSpectrogramExtractor(
            fs=cfg.data.sampling_rate,
            n_fft=cfg.snore.n_fft,
            hop_length=cfg.snore.hop_length,
            n_mels=cfg.snore.n_mels,
            fmin=cfg.snore.fmin,
            fmax=cfg.snore.fmax,
        )
        self.mfcc_extractor  = AcousticFrameExtractor(
            fs=cfg.data.sampling_rate,
            n_mfcc=cfg.snore.n_mfcc,
            n_fft=cfg.snore.n_fft,
            hop_length=cfg.snore.hop_length,
            n_mels=cfg.snore.n_mels,
            fmin=cfg.snore.fmin,
            fmax=cfg.snore.fmax,
        )
        self.preprocessor = MultimodalPreprocessor(fs=cfg.data.sampling_rate)

        # Pre-compute all (record, onset) pairs
        self.samples: List[Tuple[PatientRecord, float]] = []
        for rec in records:
            self._index_record(rec)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_record(self, rec: PatientRecord) -> None:
        """Add all valid windows from a record to self.samples."""
        fs    = self.cfg.data.sampling_rate
        w_sec = self.cfg.data.window_sec
        h_sec = self.cfg.data.hop_sec

        start_sec, end_sec = get_sleep_period(rec)

        # Use snoring channel as reference for duration
        snore_ch = rec.get_signal("Snoring_Sound") or rec.get_signal("Schnarc")
        if snore_ch is None:
            return

        start_s = int(start_sec * fs)
        end_s   = min(int(end_sec * fs), len(snore_ch))
        win_s   = int(w_sec * fs)
        hop_s   = int(h_sec * fs)

        pos = start_s
        while pos + win_s <= end_s:
            onset = pos / fs
            self.samples.append((rec, onset))
            pos += hop_s

    # ── PyTorch interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        rec, onset = self.samples[idx]
        fs    = self.cfg.data.sampling_rate
        w_sec = self.cfg.data.window_sec
        w_end = onset + w_sec

        signals = self.preprocessor.process(rec.signals)

        def _slice(ch: str) -> Optional[np.ndarray]:
            arr = signals.get(ch)
            if arr is None:
                return None
            s = int(onset * fs)
            e = int(w_end * fs)
            return arr[s:e]

        # ── Acoustic features ─────────────────────────────────────────────
        snore_wave = (_slice("Snoring_Sound") or _slice("Schnarc")
                      or np.zeros(int(w_sec * fs), dtype=np.float32))

        try:
            mel = self.mel_extractor(snore_wave)
            snore_stats = self.mfcc_extractor.extract_stats(snore_wave)
        except Exception:
            n_frames = int(w_sec * fs / self.cfg.snore.hop_length) + 1
            mel = np.zeros((self.cfg.snore.n_mels, n_frames), dtype=np.float32)
            snore_stats = np.zeros(self.mfcc_extractor.n_features * 5, dtype=np.float32)

        snore_ev = snoring_event_stats(rec.snoring_events, onset, w_end)
        snore_ev_vec = np.array(list(snore_ev.values()), dtype=np.float32)
        snore_vec = np.concatenate([snore_stats, snore_ev_vec])

        # ── Physiological features ────────────────────────────────────────
        eeg_channels = {ch: signals[ch] for ch in self.cfg.physio.eeg_channels
                        if ch in signals}
        eeg_windows  = {ch: arr[int(onset*fs):int(w_end*fs)]
                        for ch, arr in eeg_channels.items()}
        eeg_feats = multi_channel_eeg_features(eeg_windows, fs)

        ecg_wave  = _slice("ECG") or _slice("ECG 2") or np.zeros(int(w_sec * fs))
        hrv_feats = hrv_features(ecg_wave, fs)

        spo2_wave  = _slice("SpO2") or _slice("SPO2") or np.zeros(int(w_sec * 4))
        spo2_feats = spo2_features(spo2_wave, fs=4.0)

        physio_dict = {**eeg_feats, **hrv_feats, **spo2_feats}
        physio_vec  = np.array([v if v is not None and not np.isnan(float(v)) else 0.0
                                 for v in physio_dict.values()], dtype=np.float32)

        # ── Respiratory features ──────────────────────────────────────────
        airflow_wave = _slice("Airflow_Pressure") or _slice("Druck Flow") or np.zeros(int(w_sec * fs))
        resp_feats   = respiratory_event_features(
            rec.flow_events, rec.arousals, onset, w_end)
        flow_feats   = airflow_waveform_features(airflow_wave, fs)
        resp_vec = np.array(list({**resp_feats, **flow_feats}.values()), dtype=np.float32)

        # ── Labels ────────────────────────────────────────────────────────
        label       = label_window(rec.arousals, onset, w_end)
        sleep_stage = self._get_sleep_stage(rec, onset, w_end)

        # ── Data augmentation ─────────────────────────────────────────────
        if self.augment:
            mel, snore_wave = self._augment(mel, snore_wave)

        return {
            "mel_snore":   torch.from_numpy(mel).unsqueeze(0),  # (1, n_mels, T)
            "physio_vec":  torch.from_numpy(physio_vec),
            "snore_vec":   torch.from_numpy(snore_vec),
            "resp_vec":    torch.from_numpy(resp_vec),
            "label":       label,
            "sleep_stage": sleep_stage,
            "patient_id":  rec.patient_id,
            "onset_sec":   onset,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_sleep_stage(self, rec: PatientRecord, onset: float, end: float) -> int:
        if rec.sleep_stages is None or rec.sleep_stages.empty:
            return -1
        mid = (onset + end) / 2
        mask = (rec.sleep_stages["onset_sec"] <= mid) & \
               (rec.sleep_stages["onset_sec"] + rec.sleep_stages["duration_sec"] >= mid)
        sub = rec.sleep_stages[mask]
        if sub.empty:
            return -1
        stage_str = sub.iloc[0]["label"]
        return SLEEP_STAGE_MAP.get(stage_str, -1)

    def _augment(
        self, mel: np.ndarray, wave: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Light augmentation: additive Gaussian noise + time masking."""
        # Noise injection
        noise_level = np.random.uniform(0.0, 0.02)
        wave = wave + noise_level * np.random.randn(*wave.shape).astype(np.float32)

        # Time masking on mel-spectrogram
        n_frames = mel.shape[1]
        mask_len = np.random.randint(0, max(1, n_frames // 10))
        mask_start = np.random.randint(0, max(1, n_frames - mask_len))
        mel[:, mask_start : mask_start + mask_len] = mel.mean()

        # Frequency masking
        n_mels = mel.shape[0]
        fmask_len = np.random.randint(0, max(1, n_mels // 8))
        fmask_start = np.random.randint(0, max(1, n_mels - fmask_len))
        mel[fmask_start : fmask_start + fmask_len, :] = mel.mean()

        return mel, wave
