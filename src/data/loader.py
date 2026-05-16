"""
src/data/loader.py

Loads one CPS patient record into a unified structure:
  - Raw WFDB signal channels (numpy arrays, 256 Hz)
  - Parsed TXT annotation files (events with timestamps)
  - YAML questionnaire data

Directory layout expected:
    <data_dir>/<patient_id>/PSG/Analysedaten/<txt-files>
    <data_dir>/<patient_id>/PSG/<patient_id>.wfdb   (actually .hea + .dat)
    <data_dir>/<patient_id>/YAML/<yml-files>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# German → English channel / event name maps
# ──────────────────────────────────────────────────────────────────────────────

CHANNEL_MAP: Dict[str, str] = {
    "Schnarc": "Snoring_Sound",
    "Druck Snore": "Snoring_Pressure",
    "Druck Flow": "Airflow_Pressure",
    "Flow Th": "Thermal_Flow",
    "Beweg.": "Motion",
    "Pos.": "Body_Position",
    "RIP.Abdom": "RIP_Abdomen",
    "RIP.Thrx": "RIP_Thorax",
    "Summe RIPs": "Sum_RIPs",
    "Pleth": "PPG",
    "Pulse": "Pulse_Rate",
    "SPO2": "SpO2",
    "Akku": "Battery",
    "Licht": "Light",
    "ECG 2": "ECG",
    "EMG+": "EMG_pos",
    "EMG-": "EMG_neg",
    "EMG": "EMG",
}

AROUSAL_MAP: Dict[str, str] = {
    "Respiratorische Arousal (EEG)": "Respiratory_Arousal_EEG",
    "Respiratorische Arousal": "Respiratory_Arousal",
    "Flusslimitationen Arousal (EEG)": "Flow_Limitation_Arousal_EEG",
    "Flusslimitationen Arousal": "Flow_Limitation_Arousal",
    "SpO2 Arousal (EEG)": "SpO2_Arousal_EEG",
    "LM Arousal (EEG)": "LM_Arousal_EEG",
    "LM Arousal": "LM_Arousal",
    "PLM Arousal (EEG)": "PLM_Arousal_EEG",
    "PLM Arousal": "PLM_Arousal",
    "Schnarchen Arousal (EEG)": "Snoring_Arousal_EEG",
    "Schnarchen Arousal": "Snoring_Arousal",
    "Arousal (EEG)": "Spontaneous_Arousal_EEG",
    "Arousal": "Spontaneous_Arousal",
}

SLEEP_STAGE_MAP: Dict[str, str] = {
    "N1": "N1", "N2": "N2", "N3": "N3",
    "Rem": "REM", "Wach": "Wake", "Artefakt": "Artifact",
}

POSITION_MAP: Dict[str, str] = {
    "Bauch": "Prone", "Aufrecht": "Upright", "Links": "Left",
    "Rechts": "Right", "Rücken": "Supine", "A": "Unknown",
}

SNORING_EVENT_MAP: Dict[str, str] = {
    "Atemgeräusche": "Breathing_Sounds",
    "Schnarchen": "Snoring_Sounds",
}

FLOW_EVENT_MAP: Dict[str, str] = {
    "Zentrale Apnoe": "Central_Apnea",
    "Hypopnoe": "Hypopnea",
    "Zentrale Hypopnoe": "Central_Hypopnea",
    "Gemischte Apnoe": "Mixed_Apnea",
    "Obstruktive Apnoe": "Obstructive_Apnea",
    "RERA": "RERA",
    "Flusslimitationen": "Flow_Limitation",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnotationDF:
    """A labelled set of time-stamped events for one TXT file."""
    source_file: str
    events: pd.DataFrame  # columns: onset_sec, duration_sec, label, value


@dataclass
class PatientRecord:
    patient_id: str
    # Continuous signals – shape (n_samples,) per channel
    signals: Dict[str, np.ndarray] = field(default_factory=dict)
    fs: int = 256                     # all channels resampled to this
    start_time: Optional[float] = None  # POSIX timestamp of rec start (may be epoch 0)
    # Annotations
    arousals: Optional[pd.DataFrame] = None
    sleep_stages: Optional[pd.DataFrame] = None
    body_positions: Optional[pd.DataFrame] = None
    snoring_events: Optional[pd.DataFrame] = None
    flow_events: Optional[pd.DataFrame] = None
    markers: Optional[pd.DataFrame] = None
    # Questionnaire / demographics
    questionnaire: Dict = field(default_factory=dict)
    # Raw channel metadata from WFDB header
    channel_units: Dict[str, str] = field(default_factory=dict)

    def duration_sec(self) -> float:
        lens = [len(v) for v in self.signals.values()]
        return max(lens) / self.fs if lens else 0.0

    def get_signal(self, name: str) -> Optional[np.ndarray]:
        """Retrieve by English or original German name."""
        if name in self.signals:
            return self.signals[name]
        mapped = CHANNEL_MAP.get(name, name)
        return self.signals.get(mapped)


# ──────────────────────────────────────────────────────────────────────────────
# TXT annotation parser
# ──────────────────────────────────────────────────────────────────────────────

def _parse_txt_file(path: Path, event_map: Dict[str, str]) -> pd.DataFrame:
    """
    Parse a DOMINO-exported TXT annotation file.

    Format (tab-separated):
        <event_name>  <onset_sec>  <duration_sec>  [<value>]
    Lines starting with '#' are skipped.
    """
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["onset_sec", "duration_sec", "label", "value"])

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\t+", line)
            if len(parts) < 2:
                continue
            event_name = parts[0].strip()
            label = event_map.get(event_name, event_name)
            try:
                onset = float(parts[1].replace(",", "."))
                duration = float(parts[2].replace(",", ".")) if len(parts) > 2 else 0.0
                value = float(parts[3].replace(",", ".")) if len(parts) > 3 else np.nan
            except (ValueError, IndexError):
                continue
            rows.append({"onset_sec": onset, "duration_sec": duration,
                         "label": label, "value": value})

    return pd.DataFrame(rows) if rows else \
        pd.DataFrame(columns=["onset_sec", "duration_sec", "label", "value"])


# ──────────────────────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────────────────────

class CPSLoader:
    """
    Load a single CPS patient record from disk.

    Parameters
    ----------
    data_dir : str | Path
        Root directory containing per-patient sub-directories.
    target_fs : int
        Target sampling frequency; WFDB channels are already at 256 Hz.
    """

    def __init__(self, data_dir: str | Path, target_fs: int = 256):
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs

    def list_patients(self) -> List[str]:
        """Return all patient IDs found in data_dir."""
        return sorted(
            p.name for p in self.data_dir.iterdir()
            if p.is_dir() and (p / "PSG").exists()
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, patient_id: str) -> PatientRecord:
        """Load and return a complete PatientRecord for *patient_id*."""
        patient_dir = self.data_dir / patient_id
        record = PatientRecord(patient_id=patient_id)

        self._load_wfdb(patient_dir / "PSG" / patient_id, record)
        self._load_annotations(patient_dir / "PSG" / "Analysedaten", record)
        self._load_questionnaires(patient_dir / "YAML", record)

        return record

    def load_all(
        self,
        patient_ids: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[PatientRecord]:
        """Load multiple patients; defaults to all in data_dir."""
        from tqdm import tqdm

        ids = patient_ids or self.list_patients()
        records = []
        for pid in tqdm(ids, desc="Loading patients", disable=not verbose):
            try:
                records.append(self.load(pid))
            except Exception as exc:
                print(f"[WARNING] Could not load patient {pid}: {exc}")
        return records

    # ── WFDB ─────────────────────────────────────────────────────────────────

    def _load_wfdb(self, record_path: Path, record: PatientRecord) -> None:
        """Read .hea/.dat WFDB record into numpy arrays."""
        try:
            wfdb_rec = wfdb.rdrecord(str(record_path))
        except Exception as exc:
            raise RuntimeError(f"Cannot read WFDB record at {record_path}: {exc}") from exc

        record.fs = wfdb_rec.fs
        for i, sig_name in enumerate(wfdb_rec.sig_name):
            eng_name = CHANNEL_MAP.get(sig_name, sig_name)
            record.signals[eng_name] = wfdb_rec.p_signal[:, i].astype(np.float32)
            if wfdb_rec.units:
                record.channel_units[eng_name] = wfdb_rec.units[i]

        # Keep original names too (for lookup flexibility)
        for i, sig_name in enumerate(wfdb_rec.sig_name):
            if sig_name not in record.signals:
                record.signals[sig_name] = wfdb_rec.p_signal[:, i].astype(np.float32)

    # ── TXT annotations ───────────────────────────────────────────────────────

    def _load_annotations(self, analysedaten_dir: Path, record: PatientRecord) -> None:
        p = analysedaten_dir

        record.arousals = _parse_txt_file(
            p / "Klassifizierte Arousal.txt", AROUSAL_MAP)
        record.sleep_stages = _parse_txt_file(
            p / "Schlafprofil.txt", SLEEP_STAGE_MAP)
        record.body_positions = _parse_txt_file(
            p / "Körperlage.txt", POSITION_MAP)
        record.snoring_events = _parse_txt_file(
            p / "Schnarchen Events.txt", SNORING_EVENT_MAP)
        record.flow_events = _parse_txt_file(
            p / "Flow Events.txt", FLOW_EVENT_MAP)
        record.markers = _parse_txt_file(
            p / "Marker.txt", {
                "Beginn der Messung": "Measurement_Start",
                "Licht aus": "Lights_Off",
                "Licht an": "Lights_On",
                "Ende der Messung": "Measurement_End",
            })

    # ── YAML questionnaires ───────────────────────────────────────────────────

    def _load_questionnaires(self, yaml_dir: Path, record: PatientRecord) -> None:
        if not yaml_dir.exists():
            return
        q: Dict = {}
        for yml_file in yaml_dir.glob("*.yml"):
            try:
                with open(yml_file, encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
                q[yml_file.stem] = content
            except Exception:
                pass
        record.questionnaire = q


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_sleep_period(record: PatientRecord) -> Tuple[float, float]:
    """
    Return (lights_off_sec, lights_on_sec) from Marker.txt.
    Falls back to full recording if markers are missing.
    """
    if record.markers is None or record.markers.empty:
        return 0.0, record.duration_sec()

    off_rows = record.markers[record.markers["label"] == "Lights_Off"]
    on_rows  = record.markers[record.markers["label"] == "Lights_On"]
    start = float(off_rows["onset_sec"].iloc[0]) if not off_rows.empty else 0.0
    end   = float(on_rows["onset_sec"].iloc[0])  if not on_rows.empty  else record.duration_sec()
    return start, end


def events_in_window(
    events: pd.DataFrame,
    window_start: float,
    window_end: float,
    label: Optional[str] = None,
) -> pd.DataFrame:
    """Filter events whose onset falls within [window_start, window_end)."""
    mask = (events["onset_sec"] >= window_start) & (events["onset_sec"] < window_end)
    if label is not None:
        mask &= events["label"] == label
    return events[mask].copy()
