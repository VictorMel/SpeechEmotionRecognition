"""Training and evaluation loops for Emotion Detection.

Unified abstractions replacing legacy DeepLearningPyTorch epoch runners.
Design:
- Support AMP, gradient accumulation, early stopping, tensorboard hooks.
- Callable hooks for distillation (teacher forward) & ensemble aggregation.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    amp: bool = True
    accumulate_steps: int = 1
    early_patience: int = 8
    early_metric: str = "val_f1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################
# Metrics
###############################################################################

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=1) == targets).float().mean()

###############################################################################
# Core Loop
###############################################################################

def train_loop(model: nn.Module,
               loader_train: DataLoader,
               loader_val: DataLoader,
               cfg: TrainConfig,
               criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = accuracy,
               optimizer: Optional[torch.optim.Optimizer] = None,
               scheduler: Optional[Any] = None,
               tb_writer: Optional[Any] = None,
               teacher_model: Optional[nn.Module] = None,
               distill_weight: float = 0.0,
               run_name: str = "run",
               checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = cfg.device
    model.to(device)
    if teacher_model is not None:
        teacher_model.to(device)
        teacher_model.eval()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    best_metric = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_metric = 0.0
        num_samples = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader_train):
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(x)
                loss = criterion(logits, y)
                if teacher_model is not None and distill_weight > 0:
                    with torch.no_grad():
                        t_logits = teacher_model(x)
                    # Simple KL distillation
                    kd_loss = torch.nn.functional.kl_div(
                        torch.log_softmax(logits / 1.0, dim=1),
                        torch.softmax(t_logits / 1.0, dim=1),
                        reduction="batchmean"
                    )
                    loss = loss + distill_weight * kd_loss
            scaler.scale(loss).backward()
            if (step + 1) % cfg.accumulate_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                batch_metric = metric_fn(logits, y)
            running_loss += loss.item() * x.size(0)
            running_metric += batch_metric.item() * x.size(0)
            num_samples += x.size(0)
        train_loss = running_loss / num_samples
        train_metric = running_metric / num_samples

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_metric_total = 0.0
        val_samples = 0
        with torch.no_grad():
            for x, y in loader_val:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                metric_val = metric_fn(logits, y)
                val_loss_total += loss.item() * x.size(0)
                val_metric_total += metric_val.item() * x.size(0)
                val_samples += x.size(0)
        val_loss = val_loss_total / val_samples
        val_metric = val_metric_total / val_samples

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metric": train_metric,
            "val_loss": val_loss,
            "val_metric": val_metric,
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
        }
        history.append(rec)

        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("Metric/train", train_metric, epoch)
            tb_writer.add_scalar("Metric/val", val_metric, epoch)
            tb_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        improved = val_metric > best_metric
        if improved:
            best_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, f"{run_name}-epoch{epoch:03}-val{val_metric:.4f}.pt")
            torch.save({"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch}, ckpt_path)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{cfg.epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_metric={val_metric:.4f} time={epoch_time:.1f}s")
        if patience_counter >= cfg.early_patience:
            print("Early stopping triggered.")
            break

    return {"history": history, "best_metric": best_metric, "best_epoch": best_epoch}
