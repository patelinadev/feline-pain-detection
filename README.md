# Feline Pain Detection

**Status:** pipeline complete; 130 of a target 500 images annotated; training not yet run.

Binary classifier for feline pain detection using ResNet-50 fine-tuned on manually annotated cat images.
Reproduction and extension of Feighelstein et al. (2022) *Scientific Reports*.

Status: pipeline complete; 130 of a target 500 images annotated; training not yet run.
<!-- TODO (Future Work - Route C):
Construct original labeled dataset by collaborating with veterinary clinics.
Protocol: photograph cats pre-surgery (no_pain) and post-surgery before
analgesia (pain) — identical to Evangelista et al. (2019) Sci Reports.
This addresses the critical data scarcity gap in the field (all existing
labeled datasets have <500 images and are not publicly available).
-->

## Quickstart

```bash
pip install -r requirements.txt

# Step 1: Annotate images (put Crawford raw images in data/raw/ first)
python scripts/annotate.py

# Step 2: Train
python src/train.py

# Step 3: Evaluate
python src/evaluate.py
```

## Project Structure

```
data/
  pain/           # labeled pain images
  no_pain/        # labeled no-pain images
  raw/            # Crawford dataset (CAT_00 … CAT_06)
  annotations.csv # output of annotate.py
checkpoints/
  best_model.pth
results/
  training_log.csv
  loss_curve.png
  acc_curve.png
  confusion_matrix.png
  final_metrics.txt
scripts/
  annotate.py
src/
  dataset.py
  model.py
  train.py
  evaluate.py
```

## Key Design Decisions

| Decision | Why |
|----------|-----|
| ResNet-50 | Residual connections solve vanishing gradient in deep networks |
| Two-phase training | Prevents catastrophic forgetting of ImageNet features |
| Class weights in CrossEntropyLoss | Handles imbalance; false negatives (missed pain) are worse |
| ReduceLROnPlateau | Adaptive LR — only reduces when val loss actually plateaus |
| F1 as headline metric | Handles class imbalance better than accuracy |
| seed=42 everywhere | Reproducibility |

## References

- Evangelista et al. (2019) *Sci Reports* — FGS original validation
- Feighelstein et al. (2022) *Sci Reports* — First ResNet-50 on cat pain; 72% acc; 464 images
- Steagall et al. (2023) *Sci Reports* — Fully automated; 95.5% accuracy; 3447 images
