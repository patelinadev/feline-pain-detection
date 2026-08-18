"""
model.py — ResNet-50 builder (Step 3)
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_model(freeze_backbone=True):
    """
    Returns ResNet-50 with pretrained ImageNet weights.
    Final fc layer replaced for binary classification (pain vs no_pain).

    Args:
        freeze_backbone: if True, freeze all layers except fc

    Interview note: two-phase training (frozen → unfrozen) prevents
    catastrophic forgetting of ImageNet features in early epochs.
    ResNet-50 chosen over VGG/AlexNet because residual connections
    solve the vanishing gradient problem in deep networks.
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # replace final fc layer: 2048 → 2 (binary classification)
    model.fc = nn.Linear(2048, 2)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return model


def unfreeze_all(model):
    """Unfreeze all model parameters for fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True
