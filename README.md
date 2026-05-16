# 🌙 SomnoFuse

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Data-PhysioNet-lightblue)
![Tests](https://img.shields.io/badge/Tests-20%20passed-brightgreen)

**Multimodal machine learning for sleep arousal detection**, fusing snoring acoustics, EEG, ECG, and respiratory signals from polysomnographic recordings.

Built on the [CPS Dataset (PhysioNet v1.0.0)](https://physionet.org/content/cps-dataset-sleep/1.0.0/) — 113 full-night polysomnography recordings with 36 signal channels and 81 annotated event types.

---

## 🧩 Modality Fusion

| Modality | Channels | Features |
|---|---|---|
| 🔊 **Acoustic** | Schnarc (256 Hz), Druck Snore (256 Hz) | MFCCs, Mel-spectrogram, ZCR, RMS, event dB |
| 🫁 **Respiratory** | Airflow pressure, Thermal flow, RIP abdomen/thorax | Resp. rate, AHI, phase angle, paradox fraction |
| ❤️ **Cardiac** | ECG, PPG (Pleth), SpO₂, PTT | HRV (SDNN, RMSSD, LF/HF), ODI, T90 |
| 🧠 **Neurological** | EEG C3/C4/F4/O2, EMG, EOG | Band powers (δ/θ/α/σ/β), Hjorth, spectral edge |
| 🧍 **Body Position** | Gravitation sensor | Posture labels (supine, prone, left, right) |
| 📋 **Questionnaire** | ESS, PSQI, restless-legs, general sleep | Epworth score, sleep quality indices |

---

## 📁 Repository Structure

```
SomnoFuse/
├── src/
│   ├── data/
│   │   ├── loader.py          # WFDB + TXT + YAML ingestion
│   │   ├── dataset.py         # PyTorch Dataset for multimodal windows
│   │   └── preprocessing.py   # Filtering, normalisation, artefact rejection
│   ├── features/
│   │   ├── acoustic.py        # Snoring MFCCs, spectral, temporal features
│   │   ├── physiological.py   # EEG bands, HRV, SpO₂ features
│   │   ├── respiratory.py     # Apnea/hypopnea indices, flow limitation
│   │   └── fusion.py          # Early / late / attention-weighted fusion
│   ├── models/
│   │   ├── cnn_audio.py       # 2-D Mel-spectrogram CNN + 1-D waveform CNN
│   │   ├── multimodal.py      # Unified multimodal Transformer model
│   │   └── baselines.py       # Random Forest / XGBoost + SHAP
│   ├── visualization/
│   │   └── plot_signals.py    # Multi-channel PSG viewer & snoring spectrogram
│   └── utils/
│       ├── config.py          # Centralised hyper-parameter dataclass
│       └── metrics.py         # Balanced accuracy, F1, kappa, AHI
├── scripts/
│   └── train.py               # Training entry-point (deep + baseline)
├── tests/
│   ├── test_features.py       # 14 unit tests for preprocessing & features
│   └── test_models.py         # 6 model forward-pass smoke tests
├── configs/
│   └── default.yaml           # Full experiment configuration
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SomnoFuse Model                       │
│                                                         │
│  Mel-spectrogram ──► SnoreCNN (SE blocks) ──► snore_emb │
│                              │                          │
│  EEG+HRV+SpO₂  ──► MLP ────►│                          │
│                         Cross-Modal Attention           │
│  Resp features ──► MLP ────►│                          │
│                              ▼                          │
│              4-token Transformer Encoder                │
│                              │                          │
│              Attention-Weighted Fusion                  │
│                              │                          │
│              Classification Head (9 classes)            │
└─────────────────────────────────────────────────────────┘
```

**Arousal classes:** Background · Respiratory · Flow Limitation · SpO₂ · Limb Movement · PLM · Snoring · Spontaneous · Autonomic

---

## ⚙️ Setup

### 1 — Access the dataset
The CPS dataset requires PhysioNet credentials and a signed Data Use Agreement.
Register at https://physionet.org/content/cps-dataset-sleep/1.0.0/

### 2 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3 — Configure

Edit `configs/default.yaml` to set `data_dir` to your local dataset path.

### 4 — Train

```bash
# Deep multimodal model
python scripts/train.py --config configs/default.yaml --model multimodal

# Interpretable baseline
python scripts/train.py --config configs/default.yaml --model baseline_rf
```

### 5 — Test

```bash
pytest tests/ -v
```

---

## 📊 Snoring Feature Pipeline

```
Raw Schnarc (256 Hz)
        │
        ▼
  SnorePreprocessor
  (DC removal → 50 Hz notch → 20–125 Hz bandpass → RMS normalise)
        │
        ├──► MelSpectrogramExtractor  →  (64 mel bins, 64-sample hop)  →  CNN input
        │
        ├──► AcousticFrameExtractor   →  MFCCs x13 + delta + delta2,
        │                                centroid, bandwidth, rolloff, ZCR, RMS
        │
        └──► snoring_event_stats()    →  count, duration, dB (from Schnarchen Events.txt)
```

---

## 📖 Citation

If you use this code, please also cite the original dataset:

```bibtex
@article{PhysioNet-cps-dataset-sleep-1.0.0,
  author  = {Kraft, Stefan and Theissler, Andreas and Wienhausen-Wilke, Vera
             and Walter, Philipp and Kasneci, Gjergji},
  title   = {{Comprehensive Polysomnography (CPS) Dataset}},
  journal = {PhysioNet},
  year    = {2024},
  doi     = {10.13026/sxs0-h317}
}
```

---

## 📜 License

Code: **MIT** — see `LICENSE`.  
Data: PhysioNet Credentialed Health Data License 1.5.0 — see the [dataset page](https://physionet.org/content/cps-dataset-sleep/1.0.0/).
