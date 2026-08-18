"""
train.py — Main training loop (Step 4)

Two-phase training:
  Phase 1 (epochs 1-5):  backbone frozen, only fc trains, lr=1e-4
  Phase 2 (epochs 6-20): all layers unfrozen, lr=1e-5

Logs per-epoch metrics to results/training_log.csv.
Saves best model checkpoint to checkpoints/best_model.pth.
"""

import os
import csv
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# project imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from dataset import CatPainDataset
from model import build_model, unfreeze_all

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyperparameters ─────────────────────────────────────────────────────────
SEED = 42
BATCH_SIZE = 32
FROZEN_EPOCHS = 5
FINETUNE_EPOCHS = 15
TOTAL_EPOCHS = FROZEN_EPOCHS + FINETUNE_EPOCHS
LR_FROZEN = 1e-4
LR_FINETUNE = 1e-5

random.seed(SEED)
torch.manual_seed(SEED)


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights to handle imbalance."""
    labels = [dataset.samples[i][1] for i in range(len(dataset))]
    n_total = len(labels)
    n_pain = sum(labels)
    n_no_pain = n_total - n_pain
    # weight[c] = n_total / (n_classes * n_c)
    w_no_pain = n_total / (2 * n_no_pain) if n_no_pain > 0 else 1.0
    w_pain = n_total / (2 * n_pain) if n_pain > 0 else 1.0
    return torch.tensor([w_no_pain, w_pain], dtype=torch.float)


def run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── datasets & loaders ──────────────────────────────────────────────────
    train_set = CatPainDataset(DATA_DIR, split="train", seed=SEED)
    val_set = CatPainDataset(DATA_DIR, split="val", seed=SEED)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── model ───────────────────────────────────────────────────────────────
    # Phase 1: backbone frozen
    model = build_model(freeze_backbone=True).to(device)

    # class-weighted loss to handle imbalance
    # false negative (missed pain) is worse than false positive → use weights
    class_weights = compute_class_weights(train_set).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FROZEN
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ── logging ─────────────────────────────────────────────────────────────
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    best_val_loss = float("inf")
    best_ckpt = os.path.join(CHECKPOINTS_DIR, "best_model.pth")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "phase", "train_loss", "val_loss",
                         "train_acc", "val_acc", "lr"])

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(1, TOTAL_EPOCHS + 1):

        # Phase 2: switch to fine-tuning after frozen epochs
        if epoch == FROZEN_EPOCHS + 1:
            print("\n── Phase 2: Fine-tuning (all layers unfrozen) ──")
            unfreeze_all(model)
            # lower lr to protect pretrained weights
            for g in optimizer.param_groups:
                g["lr"] = LR_FINETUNE

        phase = "frozen" if epoch <= FROZEN_EPOCHS else "finetune"
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, is_train=False)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{TOTAL_EPOCHS} [{phase}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
            f"lr={current_lr:.2e}"
        )

        # save best checkpoint based on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

        # append to log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, phase, f"{train_loss:.6f}", f"{val_loss:.6f}",
                              f"{train_acc:.6f}", f"{val_acc:.6f}", f"{current_lr:.2e}"])

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoint: {best_ckpt}")
    print(f"Log: {log_path}")

    # TODO (Future Work - Grad-CAM):
    # Implement Grad-CAM to visualize which facial regions the model attends to.
    # Hypothesis: model should focus on ear position and orbital region per FGS.
    # Library: pytorch-grad-cam
    # This would make the model interpretable and clinically meaningful.


if __name__ == "__main__":
    main()
