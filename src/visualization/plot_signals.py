"""
src/visualization/plot_signals.py

Interactive and static visualisation tools for CPS sleep recordings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from src.data.loader import PatientRecord, SLEEP_STAGE_MAP


# ──────────────────────────────────────────────────────────────────────────────
# Colour maps
# ──────────────────────────────────────────────────────────────────────────────

STAGE_COLORS = {
    "Wake": "#e74c3c",
    "REM":  "#9b59b6",
    "N1":   "#3498db",
    "N2":   "#2ecc71",
    "N3":   "#1a5276",
    "Artifact": "#bdc3c7",
}

AROUSAL_COLORS = {
    "Snoring_Arousal":       "#e67e22",
    "Snoring_Arousal_EEG":   "#d35400",
    "Respiratory_Arousal":   "#c0392b",
    "Flow_Limitation_Arousal": "#8e44ad",
    "Spontaneous_Arousal":   "#2980b9",
    "PLM_Arousal":           "#27ae60",
}


# ──────────────────────────────────────────────────────────────────────────────
# Multi-channel signal viewer
# ──────────────────────────────────────────────────────────────────────────────

def plot_psg_overview(
    record: PatientRecord,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    channels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 12),
) -> plt.Figure:
    """
    Plot a multi-channel PSG overview with hypnogram and arousal markers.

    Parameters
    ----------
    record : PatientRecord
    start_sec, end_sec : window to display
    channels : list of channel names to plot; defaults to key channels
    """
    if end_sec is None:
        end_sec = min(start_sec + 300, record.duration_sec())  # default: 5 min

    if channels is None:
        channels = ["Snoring_Sound", "Airflow_Pressure", "ECG",
                    "PPG", "SpO2", "C4:A1", "EMG"]

    fs = record.fs
    t = np.arange(int((end_sec - start_sec) * fs)) / fs + start_sec

    # Filter channels to those that actually exist
    available = [ch for ch in channels if ch in record.signals or
                 any(alt in record.signals for alt in [ch, ch.replace(" ", "_")])]

    n_rows = len(available) + 1   # +1 for hypnogram
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(n_rows, 1, figure=fig, hspace=0.05)

    # ── Hypnogram ─────────────────────────────────────────────────────────────
    ax_hyp = fig.add_subplot(gs[0])
    _plot_hypnogram(ax_hyp, record, start_sec, end_sec)

    # ── Signal channels ────────────────────────────────────────────────────────
    axes = []
    for i, ch in enumerate(available):
        ax = fig.add_subplot(gs[i + 1], sharex=ax_hyp if axes else None)
        sig = record.signals.get(ch)
        if sig is None:
            ax.set_ylabel(ch, fontsize=8)
            axes.append(ax)
            continue

        s = int(start_sec * fs)
        e = min(int(end_sec * fs), len(sig))
        sig_slice = sig[s:e]
        t_slice   = np.linspace(start_sec, start_sec + len(sig_slice) / fs, len(sig_slice))

        color = "#e74c3c" if "Snor" in ch else "#2c3e50"
        ax.plot(t_slice, sig_slice, lw=0.5, color=color, rasterized=True)

        # Overlay snoring events
        if "Snor" in ch and record.snoring_events is not None:
            _overlay_events(ax, record.snoring_events, start_sec, end_sec,
                            sig_slice.min(), sig_slice.max(),
                            {"Snoring_Sounds": "#e74c3c", "Breathing_Sounds": "#f39c12"})

        ax.set_ylabel(ch, fontsize=8, rotation=0, ha="right")
        ax.tick_params(labelbottom=False, labelsize=7)
        ax.set_xlim(start_sec, end_sec)
        axes.append(ax)

    axes[-1].tick_params(labelbottom=True, labelsize=8)
    axes[-1].set_xlabel("Time (s)", fontsize=9)

    fig.suptitle(f"Patient {record.patient_id}  |  {start_sec:.0f}–{end_sec:.0f} s",
                 fontsize=11, y=0.98)
    return fig


def _plot_hypnogram(
    ax: plt.Axes,
    record: PatientRecord,
    start_sec: float,
    end_sec: float,
) -> None:
    stage_order = {"Wake": 0, "REM": 1, "N1": 2, "N2": 3, "N3": 4, "Artifact": 5}
    if record.sleep_stages is not None and not record.sleep_stages.empty:
        for _, row in record.sleep_stages.iterrows():
            on  = row["onset_sec"]
            dur = row["duration_sec"]
            label = row["label"]
            if on > end_sec or on + dur < start_sec:
                continue
            y = stage_order.get(label, 5)
            color = STAGE_COLORS.get(label, "#bdc3c7")
            ax.barh(y, min(on + dur, end_sec) - max(on, start_sec),
                    left=max(on, start_sec), height=0.85,
                    color=color, alpha=0.85, linewidth=0)

    ax.set_yticks(list(stage_order.values()))
    ax.set_yticklabels(list(stage_order.keys()), fontsize=7)
    ax.set_ylabel("Stage", fontsize=8, rotation=0, ha="right")
    ax.set_xlim(start_sec, end_sec)
    ax.tick_params(labelbottom=False)
    ax.set_title("Hypnogram", fontsize=8, loc="left")


def _overlay_events(
    ax: plt.Axes,
    events,
    start_sec: float,
    end_sec: float,
    ymin: float,
    ymax: float,
    label_color: Dict[str, str],
) -> None:
    for _, row in events.iterrows():
        on  = row["onset_sec"]
        dur = row["duration_sec"]
        label = row["label"]
        if on > end_sec or on + dur < start_sec:
            continue
        color = label_color.get(label, "#95a5a6")
        ax.axvspan(max(on, start_sec), min(on + dur, end_sec),
                   alpha=0.25, color=color, linewidth=0)


# ──────────────────────────────────────────────────────────────────────────────
# Spectrogram viewer for snoring channel
# ──────────────────────────────────────────────────────────────────────────────

def plot_snoring_spectrogram(
    snore_wave: np.ndarray,
    fs: int = 256,
    snoring_events=None,
    start_sec: float = 0.0,
    title: str = "Snoring Spectrogram",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot spectrogram of the snoring channel with event overlays.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})

    # Spectrogram
    ax = axes[0]
    f, t_spec, Sxx = plt.mlab.specgram(snore_wave, NFFT=256, Fs=fs,
                                         noverlap=192, mode="magnitude")
    t_spec += start_sec
    im = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10),
                        shading="gouraud", cmap="inferno", rasterized=True)
    ax.set_ylim(0, min(128, fs / 2))
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="dB")

    # Waveform
    ax2 = axes[1]
    t_wave = np.linspace(start_sec, start_sec + len(snore_wave) / fs, len(snore_wave))
    ax2.plot(t_wave, snore_wave, lw=0.4, color="#e74c3c", rasterized=True)

    if snoring_events is not None and not snoring_events.empty:
        for _, row in snoring_events.iterrows():
            on, dur = row["onset_sec"], row["duration_sec"]
            col = "#e74c3c" if row["label"] == "Snoring_Sounds" else "#f39c12"
            ax2.axvspan(on, on + dur, alpha=0.3, color=col, linewidth=0)
        patches = [
            mpatches.Patch(color="#e74c3c", alpha=0.4, label="Snoring"),
            mpatches.Patch(color="#f39c12", alpha=0.4, label="Breathing"),
        ]
        ax2.legend(handles=patches, fontsize=7, loc="upper right")

    ax2.set_ylabel("Amplitude", fontsize=9)
    ax2.set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    return fig
