"""
src/utils/config.py
Centralised experiment configuration loaded from YAML.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_fold_file: str = "configs/recommended_training_fold.txt"
    test_fold_file: str = "configs/recommended_test_fold.txt"
    sampling_rate: int = 256          # Hz – all channels upsampled to this
    window_sec: float = 30.0          # PSG epoch length in seconds
    hop_sec: float = 15.0             # Window hop (50 % overlap)
    max_patients: Optional[int] = None  # None → use all 113


@dataclass
class SnoreConfig:
    """Parameters for acoustic (snoring) feature extraction."""
    channel: str = "Schnarc"          # raw WFDB channel name
    pressure_channel: str = "Druck Snore"
    n_mfcc: int = 13
    n_fft: int = 512                  # ~2 ms at 256 Hz
    hop_length: int = 64              # ~250 ms hop
    n_mels: int = 128
    fmin: float = 20.0                # Hz
    fmax: float = 2000.0              # Hz  (snoring energy < 2 kHz)
    energy_threshold_db: float = -40.0  # frames below this → silence


@dataclass
class PhysioConfig:
    """EEG / ECG / PPG / SpO₂ feature settings."""
    eeg_channels: List[str] = field(default_factory=lambda: [
        "C4:A1", "C3:A2", "F4:A1", "O2:A1"
    ])
    eeg_bands: dict = field(default_factory=lambda: {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "sigma": (11, 16),
        "beta":  (13, 30),
    })
    ecg_channel: str = "ECG 2"
    ppg_channel: str = "Pleth"
    spo2_channel: str = "SPO2"
    hrv_window_sec: float = 300.0     # 5-min HRV segments


@dataclass
class RespiratoryConfig:
    airflow_channel: str = "Druck Flow"
    thermal_channel: str = "Flow Th"
    rip_abdomen: str = "RIP.Abdom"
    rip_thorax: str = "RIP.Thrx"
    apnea_min_duration_sec: float = 10.0
    hypopnea_threshold: float = 0.3   # 30 % reduction in flow


@dataclass
class ModelConfig:
    model_type: str = "multimodal"    # cnn_audio | transformer | multimodal | baseline_rf
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    num_classes: int = 9              # 8 arousal types + background


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"        # cosine | plateau | none
    early_stopping_patience: int = 10
    seed: int = 42
    device: str = "auto"             # auto | cpu | cuda | mps
    num_workers: int = 4
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class ExperimentConfig:
    name: str = "cps_snore_multimodal"
    data: DataConfig = field(default_factory=DataConfig)
    snore: SnoreConfig = field(default_factory=SnoreConfig)
    physio: PhysioConfig = field(default_factory=PhysioConfig)
    respiratory: RespiratoryConfig = field(default_factory=RespiratoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ── class helpers ─────────────────────────────────────────────────────────
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load config from a YAML file, with defaults for any missing keys."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        for section, sub_cfg in [
            ("data", cfg.data),
            ("snore", cfg.snore),
            ("physio", cfg.physio),
            ("respiratory", cfg.respiratory),
            ("model", cfg.model),
            ("training", cfg.training),
        ]:
            if section in raw:
                for k, v in raw[section].items():
                    if hasattr(sub_cfg, k):
                        setattr(sub_cfg, k, v)

        if "name" in raw:
            cfg.name = raw["name"]
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        """Serialise config back to YAML (for reproducibility)."""
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
