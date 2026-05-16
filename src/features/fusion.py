"""
src/features/fusion.py

Multimodal feature fusion strategies:
  - Early fusion (feature concatenation)
  - Late fusion (prediction averaging / stacking)
  - Attention-weighted fusion for time-series modalities
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Tabular (hand-crafted feature) early fusion
# ──────────────────────────────────────────────────────────────────────────────

def concatenate_features(
    feature_dicts: List[Dict[str, float]],
    fill_nan: float = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate multiple feature dicts into a single numpy vector.

    Returns
    -------
    vector : ndarray, shape (n_total_features,)
    names  : list of feature names in the same order
    """
    all_keys: List[str] = []
    for d in feature_dicts:
        all_keys.extend(d.keys())

    merged = {}
    for d in feature_dicts:
        merged.update(d)

    values = []
    for k in all_keys:
        v = merged.get(k, fill_nan)
        values.append(fill_nan if (v is None or np.isnan(float(v))) else float(v))

    return np.array(values, dtype=np.float32), all_keys


# ──────────────────────────────────────────────────────────────────────────────
# Learned attention-weighted fusion (for deep models)
# ──────────────────────────────────────────────────────────────────────────────

class ModalityAttentionFusion(nn.Module):
    """
    Weighted sum of modality embeddings using a learnable attention gate.

    Parameters
    ----------
    n_modalities : int
        Number of input modality streams.
    embed_dim : int
        Dimensionality of each modality embedding.
    """

    def __init__(self, n_modalities: int, embed_dim: int):
        super().__init__()
        self.n_modalities = n_modalities
        self.embed_dim = embed_dim
        self.gate = nn.Linear(embed_dim, 1)

    def forward(self, modality_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        modality_embeddings : list of tensors, each (batch, embed_dim)

        Returns
        -------
        fused : (batch, embed_dim)
        """
        stacked = torch.stack(modality_embeddings, dim=1)   # (B, M, D)
        scores  = self.gate(stacked).squeeze(-1)             # (B, M)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)     # (B, M, 1)
        fused   = (stacked * weights).sum(dim=1)             # (B, D)
        return fused


class CrossModalAttention(nn.Module):
    """
    Cross-attention between an acoustic query and physiological key/value pairs.
    Useful when snoring is the primary modality and others provide context.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,   # (B, T_q, D) – acoustic features
        context: torch.Tensor, # (B, T_k, D) – physiological features
    ) -> torch.Tensor:
        attn_out, _ = self.attn(query, context, context)
        return self.norm(query + self.dropout(attn_out))


# ──────────────────────────────────────────────────────────────────────────────
# Late fusion (ensemble)
# ──────────────────────────────────────────────────────────────────────────────

class LateFusion:
    """
    Combine probabilistic predictions from multiple modality-specific models.

    Strategies: 'mean', 'max', 'weighted'
    """

    def __init__(
        self,
        strategy: str = "mean",
        weights: Optional[np.ndarray] = None,
    ):
        assert strategy in ("mean", "max", "weighted")
        self.strategy = strategy
        self.weights  = weights

    def __call__(self, probs: List[np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        probs : list of ndarray, each (n_samples, n_classes) or (n_classes,)

        Returns
        -------
        fused : ndarray, same shape as each element of probs
        """
        stack = np.stack(probs, axis=0)   # (M, ...)
        if self.strategy == "mean":
            return stack.mean(axis=0)
        elif self.strategy == "max":
            return stack.max(axis=0)
        elif self.strategy == "weighted":
            if self.weights is None:
                raise ValueError("weights must be provided for weighted fusion")
            w = np.array(self.weights)
            w = w / w.sum()
            return (stack * w.reshape(-1, *([1] * (stack.ndim - 1)))).sum(axis=0)
        return stack.mean(axis=0)
