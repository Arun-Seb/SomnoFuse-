#!/usr/bin/env python
"""
scripts/train.py

End-to-end training script for the multimodal sleep arousal model.

Usage
-----
python scripts/train.py --config configs/default.yaml --model multimodal
python scripts/train.py --config configs/default.yaml --model baseline_rf
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import ExperimentConfig
from src.data.loader import CPSLoader
from src.data.dataset import CPSWindowDataset, AROUSAL_CLASS_NAMES
from src.models.multimodal import MultimodalSleepArousalModel
from src.models.baselines import (
    build_rf_pipeline, build_xgb_pipeline,
    collate_features, evaluate_classifier, save_model
)
from src.utils.metrics import compute_epoch_metrics, AverageMeter


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────

def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


# ──────────────────────────────────────────────────────────────────────────────
# Class-balanced sampler
# ──────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: CPSWindowDataset) -> WeightedRandomSampler:
    labels = [dataset.samples[i] for i in range(len(dataset))]
    # Quick pass to get labels
    label_list = [dataset[i]["label"] for i in range(len(dataset))]
    counts = np.bincount(label_list, minlength=9).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[l] for l in label_list], dtype=torch.float)
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ──────────────────────────────────────────────────────────────────────────────
# Deep model training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_deep_model(cfg: ExperimentConfig, device: torch.device) -> None:
    from rich.console import Console
    from rich.progress import track
    console = Console()

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = CPSLoader(cfg.data.data_dir, target_fs=cfg.data.sampling_rate)
    all_ids = loader.list_patients()

    train_ids = _read_fold(cfg.data.train_fold_file, all_ids)
    test_ids  = _read_fold(cfg.data.test_fold_file,  all_ids)

    console.print(f"[bold]Train patients:[/bold] {len(train_ids)}  "
                  f"[bold]Test patients:[/bold] {len(test_ids)}")

    train_records = loader.load_all(train_ids)
    test_records  = loader.load_all(test_ids)

    train_ds = CPSWindowDataset(train_records, cfg, augment=True)
    test_ds  = CPSWindowDataset(test_records,  cfg, augment=False)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                               sampler=sampler,
                               num_workers=cfg.training.num_workers,
                               pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.training.batch_size,
                               shuffle=False,
                               num_workers=cfg.training.num_workers,
                               pin_memory=True)

    # Infer feature dimensions from first sample
    sample = train_ds[0]
    physio_dim = sample["physio_vec"].shape[0]
    snore_dim  = sample["snore_vec"].shape[0]
    resp_dim   = sample["resp_vec"].shape[0]

    console.print(f"Feature dims — physio: {physio_dim}, snore: {snore_dim}, resp: {resp_dim}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultimodalSleepArousalModel(
        physio_dim=physio_dim,
        snore_dim=snore_dim,
        resp_dim=resp_dim,
        n_mels=cfg.snore.n_mels,
        embed_dim=cfg.model.hidden_dim,
        n_classes=cfg.model.num_classes,
        num_heads=cfg.model.num_heads,
        num_transformer_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Trainable parameters: [green]{n_params:,}[/green]")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs)
    elif cfg.training.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
    else:
        scheduler = None

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision and device.type == "cuda")

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    patience_count = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        loss_meter = AverageMeter()
        t0 = time.time()

        for batch in train_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            labels = batch_dev["label"].long()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision and device.type == "cuda"):
                logits = model(batch_dev)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), labels.size(0))

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = _evaluate_deep(model, test_loader, device, criterion)
        acc = val_metrics["balanced_accuracy"]

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["loss"])
        elif scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        console.print(
            f"Epoch {epoch:03d}/{cfg.training.epochs} | "
            f"Loss: {loss_meter.avg:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Bal-Acc: {acc:.3f} | "
            f"F1-macro: {val_metrics['f1_macro']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if acc > best_acc:
            best_acc = acc
            patience_count = 0
            torch.save(model.state_dict(), ckpt_dir / f"{cfg.name}_best.pt")
        else:
            patience_count += 1
            if patience_count >= cfg.training.early_stopping_patience:
                console.print("[yellow]Early stopping triggered.[/yellow]")
                break

    console.print(f"\n[bold green]Best balanced accuracy: {best_acc:.4f}[/bold green]")


def _evaluate_deep(model, loader, device, criterion) -> dict:
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            labels = batch_dev["label"].long()
            logits = model(batch_dev)
            loss   = criterion(logits, labels)
            loss_meter.update(loss.item(), labels.size(0))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return compute_epoch_metrics(
        np.array(all_labels), np.array(all_preds),
        extra={"loss": loss_meter.avg}
    )


# ──────────────────────────────────────────────────────────────────────────────
# Baseline training
# ──────────────────────────────────────────────────────────────────────────────

def train_baseline(cfg: ExperimentConfig, model_type: str) -> None:
    loader = CPSLoader(cfg.data.data_dir, target_fs=cfg.data.sampling_rate)
    all_ids = loader.list_patients()

    train_ids = _read_fold(cfg.data.train_fold_file, all_ids)
    test_ids  = _read_fold(cfg.data.test_fold_file,  all_ids)

    train_records = loader.load_all(train_ids)
    test_records  = loader.load_all(test_ids)

    train_ds = CPSWindowDataset(train_records, cfg, augment=False)
    test_ds  = CPSWindowDataset(test_records,  cfg, augment=False)

    print("Building feature matrices …")
    X_train, y_train = collate_features([train_ds[i] for i in range(len(train_ds))])
    X_test,  y_test  = collate_features([test_ds[i]  for i in range(len(test_ds))])

    clf = build_rf_pipeline() if model_type == "baseline_rf" else build_xgb_pipeline()
    print(f"Fitting {model_type} on {len(X_train)} samples …")
    clf.fit(X_train, y_train)

    results = evaluate_classifier(clf, X_test, y_test, AROUSAL_CLASS_NAMES)
    print("\n── Evaluation Results ──")
    print(f"Balanced Accuracy : {results['balanced_accuracy']:.4f}")
    if "roc_auc_ovr" in results:
        print(f"ROC-AUC (OvR)     : {results['roc_auc_ovr']:.4f}")
    print("\nClassification Report:\n", results["report"])

    out_path = Path(cfg.training.checkpoint_dir) / f"{cfg.name}_{model_type}.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(clf, str(out_path))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_fold(fold_file: str, all_ids: list) -> list:
    p = Path(fold_file)
    if not p.exists():
        print(f"[WARNING] Fold file {fold_file} not found; using all patients.")
        return all_ids
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CPS Sleep — Multimodal Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model",  default="multimodal",
                        choices=["multimodal", "baseline_rf", "baseline_xgb"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = ExperimentConfig.from_yaml(args.config)
    seed_everything(cfg.training.seed)

    if args.model == "multimodal":
        device = get_device(cfg.training.device)
        print(f"Using device: {device}")
        train_deep_model(cfg, device)
    else:
        train_baseline(cfg, args.model)


if __name__ == "__main__":
    main()
