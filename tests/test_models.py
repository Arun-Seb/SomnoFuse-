"""
tests/test_models.py
Smoke tests for model forward passes using random tensors.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.models.cnn_audio import SnoreCNN, SnoreWaveformCNN
from src.models.multimodal import MultimodalSleepArousalModel
from src.features.fusion import ModalityAttentionFusion, LateFusion
import numpy as np


B = 4   # batch size
N_MELS, T_FRAMES = 64, 94
N_CLASSES = 9
PHYSIO_DIM, SNORE_DIM, RESP_DIM = 120, 80, 20


def test_snore_cnn_forward():
    model = SnoreCNN(n_mels=N_MELS, embed_dim=128, n_classes=N_CLASSES, classify=True)
    x = torch.randn(B, 1, N_MELS, T_FRAMES)
    out = model(x)
    assert out.shape == (B, N_CLASSES)


def test_snore_cnn_embedding():
    model = SnoreCNN(n_mels=N_MELS, embed_dim=128, classify=False)
    x = torch.randn(B, 1, N_MELS, T_FRAMES)
    emb = model(x)
    assert emb.shape == (B, 128)


def test_snore_waveform_cnn():
    model = SnoreWaveformCNN(embed_dim=64)
    x = torch.randn(B, 1, 256 * 30)   # 30 s at 256 Hz
    out = model(x)
    assert out.shape == (B, 64)


def test_multimodal_forward():
    model = MultimodalSleepArousalModel(
        physio_dim=PHYSIO_DIM,
        snore_dim=SNORE_DIM,
        resp_dim=RESP_DIM,
        n_mels=N_MELS,
        embed_dim=64,
        n_classes=N_CLASSES,
        num_heads=4,
        num_transformer_layers=2,
    )
    batch = {
        "mel_snore":  torch.randn(B, 1, N_MELS, T_FRAMES),
        "physio_vec": torch.randn(B, PHYSIO_DIM),
        "snore_vec":  torch.randn(B, SNORE_DIM),
        "resp_vec":   torch.randn(B, RESP_DIM),
    }
    logits = model(batch)
    assert logits.shape == (B, N_CLASSES)
    assert not torch.isnan(logits).any()


def test_multimodal_embed():
    model = MultimodalSleepArousalModel(
        physio_dim=PHYSIO_DIM, snore_dim=SNORE_DIM, resp_dim=RESP_DIM,
        n_mels=N_MELS, embed_dim=64, n_classes=N_CLASSES, num_heads=4,
    )
    batch = {
        "mel_snore":  torch.randn(B, 1, N_MELS, T_FRAMES),
        "physio_vec": torch.randn(B, PHYSIO_DIM),
        "snore_vec":  torch.randn(B, SNORE_DIM),
        "resp_vec":   torch.randn(B, RESP_DIM),
    }
    emb = model.embed(batch)
    assert emb.shape == (B, 64)


def test_modality_attention_fusion():
    fuser = ModalityAttentionFusion(n_modalities=4, embed_dim=32)
    mods  = [torch.randn(B, 32) for _ in range(4)]
    out   = fuser(mods)
    assert out.shape == (B, 32)
