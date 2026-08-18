"""
evaluate.py — Test set evaluation (Step 5)

Loads best checkpoint, runs on test set ONCE, prints and saves metrics.
Run after training is complete.

Expected output:
==========================================
FINAL TEST SET RESULTS
==========================================
Accuracy:  0.83
Precision: 0.81  (weighted)
Recall:    0.84  (weighted)
F1-Score:  0.82  (weighted)   ← THIS goes in resume
------------------------------------------
Confusion Matrix:
             Predicted
             no_pain  pain
Actual no_pain  [ 68    7  ]
       pain    [  9   66  ]
==========================================
"""

# TODO (Senior-level):
# In clinical use, false negatives (missed pain) are worse than false positives.
# Tune classification threshold below 0.5 to maximize Recall for pain class.
# Use precision-recall curve on validation set to select threshold.
# Reference: FGS paper uses 0.39/1.0 as clinical intervention threshold.

import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CatPainDataset
from model import build_model

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINT = os.path.join(ROOT, "checkpoints", "best_model.pth")
SEED = 42
BATCH_SIZE = 32
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_training_curves():
    """Plot loss and accuracy curves from training_log.csv."""
    import pandas as pd
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    if not os.path.exists(log_path):
        print("training_log.csv not found, skipping curve plots.")
        return

    df = pd.read_csv(log_path)
    epochs = df["epoch"].tolist()

    # loss curve
    plt.figure()
    plt.plot(epochs, df["train_loss"], label="Train Loss")
    plt.plot(epochs, df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"))
    plt.close()
    print("Saved: results/loss_curve.png")

    # accuracy curve
    plt.figure()
    plt.plot(epochs, df["train_acc"], label="Train Acc")
    plt.plot(epochs, df["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Val Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "acc_curve.png"))
    plt.close()
    print("Saved: results/acc_curve.png")


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()
    print("Saved: results/confusion_matrix.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load test set ────────────────────────────────────────────────────────
    test_set = CatPainDataset(DATA_DIR, split="test", seed=SEED)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── load best checkpoint ─────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run src/train.py first.")
        return

    model = build_model(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()

    # ── inference ────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── metrics ──────────────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    class_names = ["no_pain", "pain"]
    result_str = (
        "==========================================\n"
        "FINAL TEST SET RESULTS\n"
        "==========================================\n"
        f"Accuracy:  {acc:.4f}\n"
        f"Precision: {prec:.4f}  (weighted)\n"
        f"Recall:    {rec:.4f}  (weighted)\n"
        f"F1-Score:  {f1:.4f}  (weighted)   <- THIS goes in resume\n"
        "------------------------------------------\n"
        "Confusion Matrix:\n"
        "             Predicted\n"
        f"             {class_names[0]}  {class_names[1]}\n"
        f"Actual {class_names[0]}  {cm[0].tolist()}\n"
        f"       {class_names[1]}     {cm[1].tolist()}\n"
        "==========================================\n"
    )

    print(result_str)

    # save to file
    metrics_path = os.path.join(RESULTS_DIR, "final_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(result_str)
    print(f"Saved: results/final_metrics.txt")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, class_names)
    plot_training_curves()


if __name__ == "__main__":
    main()
