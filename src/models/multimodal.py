"""
src/models/multimodal.py

Unified multimodal architecture that fuses:
  1. Snoring spectrogram  → SnoreCNN                    → snore_emb
  2. Physiological tabular → MLP projection             → physio_emb
  3. Respiratory tabular   → MLP projection             → resp_emb
  4. Cross-modal attention (snore query, physio+resp kv)
  5. Classification head

Forward input keys mirror CPSWindowDataset output:
  mel_snore, physio_vec, snore_vec, resp_vec
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_audio import SnoreCNN
from src.features.fusion import CrossModalAttention, ModalityAttentionFusion


# ──────────────────────────────────────────────────────────────────────────────
# MLP projector (tabular → embed_dim)
# ──────────────────────────────────────────────────────────────────────────────

class MLPProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Multimodal model
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalSleepArousalModel(nn.Module):
    """
    Full multimodal model for sleep arousal classification.

    Parameters
    ----------
    physio_dim : int   – size of physio_vec (EEG + HRV + SpO₂ features)
    snore_dim  : int   – size of snore_vec  (MFCC stats + event features)
    resp_dim   : int   – size of resp_vec   (respiratory features)
    n_mels     : int   – mel bins in mel_snore
    embed_dim  : int   – shared embedding dimension
    n_classes  : int   – number of arousal classes
    """

    def __init__(
        self,
        physio_dim: int,
        snore_dim: int,
        resp_dim: int,
        n_mels: int = 64,
        embed_dim: int = 256,
        n_classes: int = 9,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Acoustic branch ───────────────────────────────────────────────────
        self.snore_cnn = SnoreCNN(
            n_mels=n_mels, embed_dim=embed_dim, classify=False, dropout=dropout
        )

        # ── Tabular branches ──────────────────────────────────────────────────
        self.physio_proj = MLPProjector(physio_dim, embed_dim, dropout=dropout)
        self.snore_proj  = MLPProjector(snore_dim,  embed_dim, dropout=dropout)
        self.resp_proj   = MLPProjector(resp_dim,   embed_dim, dropout=dropout)

        # ── Cross-modal attention: snore queries physio + resp ────────────────
        self.cross_attn = CrossModalAttention(embed_dim, num_heads, dropout)

        # ── Transformer over the 4 modality tokens ────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        # ── Fusion & classification ───────────────────────────────────────────
        self.fusion = ModalityAttentionFusion(n_modalities=4, embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict with keys mel_snore, physio_vec, snore_vec, resp_vec

        Returns
        -------
        logits : (B, n_classes)
        """
        # Acoustic embedding from Mel-spectrogram
        snore_cnn_emb = self.snore_cnn(batch["mel_snore"])     # (B, D)

        # Tabular embeddings
        physio_emb = self.physio_proj(batch["physio_vec"])     # (B, D)
        snore_emb  = self.snore_proj(batch["snore_vec"])       # (B, D)
        resp_emb   = self.resp_proj(batch["resp_vec"])         # (B, D)

        # Cross-modal: let snore CNN attend to physio + resp
        kv = torch.stack([physio_emb, resp_emb], dim=1)        # (B, 2, D)
        q  = snore_cnn_emb.unsqueeze(1)                        # (B, 1, D)
        refined = self.cross_attn(q, kv).squeeze(1)            # (B, D)

        # Pack all 4 tokens into a sequence for Transformer
        tokens = torch.stack(
            [refined, physio_emb, snore_emb, resp_emb], dim=1  # (B, 4, D)
        )
        tokens = self.transformer(tokens)                       # (B, 4, D)

        # Attention-weighted fusion → single vector
        token_list = [tokens[:, i, :] for i in range(tokens.size(1))]
        fused = self.fusion(token_list)                         # (B, D)

        return self.classifier(fused)                           # (B, n_classes)

    # ── Embedding (for downstream analysis / visualisation) ───────────────────
    @torch.no_grad()
    def embed(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the fused embedding without classification head."""
        snore_cnn_emb = self.snore_cnn(batch["mel_snore"])
        physio_emb    = self.physio_proj(batch["physio_vec"])
        snore_emb     = self.snore_proj(batch["snore_vec"])
        resp_emb      = self.resp_proj(batch["resp_vec"])

        kv      = torch.stack([physio_emb, resp_emb], dim=1)
        q       = snore_cnn_emb.unsqueeze(1)
        refined = self.cross_attn(q, kv).squeeze(1)

        tokens  = torch.stack([refined, physio_emb, snore_emb, resp_emb], dim=1)
        tokens  = self.transformer(tokens)
        token_list = [tokens[:, i, :] for i in range(tokens.size(1))]
        return self.fusion(token_list)
