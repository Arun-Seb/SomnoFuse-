# 🌙 CPS Sleep — Multimodal Snoring Analysis

A research-grade pipeline for multimodal analysis of **snoring and sleep arousals** using the
[CPS Dataset (PhysioNet v1.0.0)](https://physionet.org/content/cps-dataset-sleep/1.0.0/).

The pipeline fuses:
- **Acoustic** — `Schnarc` (snoring vibration, 256 Hz) & `Druck Snore` (snoring pressure, 256 Hz)
- **Respiratory** — airflow pressure, thermal flow, RIP abdomen/thorax
- **Cardiac** — ECG, PPG (Pleth), SpO₂, pulse transit time
- **Neurological** — EEG (C3/C4/F4/O2), EMG, EOG
- **Body position** — gravitation-based posture sensor
- **Questionnaire** — ESS, PSQI, restless-legs, general sleep questionnaire

---

## 📁 Repository Structure

```
cps-sleep-analysis/
├── src/
│   ├── data/
│   │   ├── loader.py          # WFDB + TXT + YAML ingestion
│   │   ├── dataset.py         # PyTorch Dataset for multimodal windows
│   │   └── preprocessing.py   # Filtering, normalisation, artefact rejection
│   ├── features/
│   │   ├── acoustic.py        # Snoring MFCCs, spectral, temporal features
│   │   ├── physiological.py   # EEG bands, HRV, SpO2 features
│   │   ├── respiratory.py     # Apnea/hypopnea indices, flow limitation
│   │   └── fusion.py          # Late-/early-fusion helpers
│   ├── models/
│   │   ├── cnn_audio.py       # 1-D/2-D CNN for snoring spectrograms
│   │   ├── transformer.py     # Temporal Transformer for multimodal streams
│   │   ├── multimodal.py      # Unified multimodal model
│   │   └── baselines.py       # Random forest / XGBoost baselines
│   ├── visualization/
│   │   ├── plot_signals.py    # Multi-channel PSG viewer
│   │   ├── plot_features.py   # Feature importance, SHAP
│   │   └── plot_results.py    # ROC, confusion matrix, sleep hypnogram
│   └── utils/
│       ├── annotations.py     # Parse TXT event files (German → English)
│       ├── metrics.py         # Sensitivity, specificity, AHI, F1
│       └── config.py          # Centralised hyper-parameter dataclass
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_snoring_feature_engineering.ipynb
│   ├── 03_multimodal_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── scripts/
│   ├── download_data.sh       # WFDB download helper (requires credentials)
│   ├── preprocess_all.py      # Batch preprocessing
│   └── train.py               # Training entry-point
├── tests/
│   ├── test_loader.py
│   ├── test_features.py
│   └── test_models.py
├── configs/
│   └── default.yaml           # Full experiment config
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Setup

### 1 — Access the dataset
The CPS dataset requires PhysioNet credentials and a signed DUA.
Follow the instructions at https://physionet.org/content/cps-dataset-sleep/1.0.0/

```bash
# After credentialing, download with wget or the provided helper:
bash scripts/download_data.sh /path/to/data
```

### 2 — Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3 — Configure

Edit `configs/default.yaml` to set `data_dir`, GPUs, and hyper-parameters.

### 4 — Preprocess

```bash
python scripts/preprocess_all.py --config configs/default.yaml
```

### 5 — Train

```bash
python scripts/train.py --config configs/default.yaml --model multimodal
```

---

## 🧠 Modelling Overview

| Model | Inputs | Target | Notes |
|---|---|---|---|
| `SnoreCNN` | Mel-spectrogram (Schnarc) | Snoring event detection | 1-D conv + attention |
| `MultimodalTransformer` | Snore + EEG + ECG + SpO₂ + Resp | Arousal type (9 classes) | Cross-modal attention |
| `BaselineRF` | Hand-crafted features | Arousal / sleep stage | Interpretable baseline |

---

## 📊 Snoring Feature Summary

From the raw 256 Hz `Schnarc` channel the pipeline extracts:

- **Temporal** — RMS energy, ZCR, snore duration/interval
- **Spectral** — MFCCs (13 + delta + delta-delta), spectral centroid, rolloff, bandwidth
- **Time-frequency** — Mel-spectrogram (128 bins, 25 ms hop), CWT scalogram
- **Event-level** — dB values from `Schnarchen Events.txt`, snoring arousal timestamps

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

Code: MIT.  Data: PhysioNet Credentialed Health Data License 1.5.0 — see dataset page.
