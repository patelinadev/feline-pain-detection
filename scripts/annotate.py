"""
annotate.py — Manual FGS labeling tool (Step 1)

Usage:
    python scripts/annotate.py

Loads images from data/raw/ one by one and prompts the user to label each image
using the Feline Grimace Scale (FGS). Labels are saved to data/annotations.csv.

FGS Decision Rule (Evangelista et al. 2019):
    Score each of 5 Action Units from 0-2, sum them, divide by 10.
    If score >= 0.39 → pain, else → no_pain

    Action Units:
        - Ear position:        0=forward, 1=slightly apart, 2=rotated outwards
        - Orbital tightening:  0=open,    1=partially closed, 2=squinted
        - Muzzle tension:      0=relaxed, 1=mild tension, 2=tense/elliptical
        - Whiskers change:     0=loose,   1=slightly straight, 2=straight/forward
        - Head position:       0=above shoulder, 1=at shoulder, 2=below shoulder

Target: 250 pain + 250 no_pain (stop collecting once either class hits 250).
"""

# TODO (Future Work - Route B):
# Instead of manual visual labeling, use CatFLW 48-point facial landmarks
# to compute quantitative geometric descriptors aligned with FGS Action Units:
#   - Ear rotation angle (points 3,4 relative to midline)
#   - Inter-eyelid distance ratio (orbital tightening proxy)
#   - Whisker spread angle
# Dataset: https://www.kaggle.com/datasets/georgemartvel/catflw
# Reference: Steagall et al. (2023) Nature Sci Reports - 95.5% accuracy using
# XGBoost on geometric descriptors from 3447 images

import os
import csv
import glob
from datetime import datetime

import matplotlib
matplotlib.use("MacOSX")  # native macOS backend, no Tk dependency
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ANNOTATIONS_CSV = os.path.join(DATA_DIR, "annotations.csv")
TARGET_PER_CLASS = 250
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_existing_annotations():
    """Return a dict of {filename: label} for already-labeled images."""
    if not os.path.exists(ANNOTATIONS_CSV):
        return {}
    with open(ANNOTATIONS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["label"] for row in reader}


def count_labels(annotations):
    pain_count = sum(1 for v in annotations.values() if v == "pain")
    no_pain_count = sum(1 for v in annotations.values() if v == "no_pain")
    return pain_count, no_pain_count


def get_all_images():
    """Recursively find all images under data/raw/."""
    images = []
    for ext in VALID_EXTENSIONS:
        images.extend(glob.glob(os.path.join(RAW_DIR, "**", f"*{ext}"), recursive=True))
        images.extend(glob.glob(os.path.join(RAW_DIR, "**", f"*{ext.upper()}"), recursive=True))
    return sorted(set(images))


def append_annotation(filename, label):
    file_exists = os.path.exists(ANNOTATIONS_CSV)
    with open(ANNOTATIONS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "timestamp"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "filename": filename,
            "label": label,
            "timestamp": datetime.now().isoformat(),
        })


def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.clf()
    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(image_path), fontsize=10)
    plt.tight_layout()
    plt.pause(0.001)  # render without blocking


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: Raw image directory not found: {RAW_DIR}")
        print("Download the Crawford dataset and place images under data/raw/CAT_00/ ... CAT_06/")
        return

    all_images = get_all_images()
    if not all_images:
        print(f"No images found in {RAW_DIR}")
        return

    existing = load_existing_annotations()
    pain_count, no_pain_count = count_labels(existing)

    print(f"\nFeline Pain Annotation Tool")
    print(f"Total images found: {len(all_images)}")
    print(f"Already labeled: {len(existing)}  (pain={pain_count}, no_pain={no_pain_count})")
    print(f"Target: {TARGET_PER_CLASS} per class\n")
    print("Controls: p=pain  n=no_pain  s=skip  q=quit\n")

    plt.ion()
    fig = plt.figure(figsize=(6, 6))

    for image_path in all_images:
        rel_path = os.path.relpath(image_path, DATA_DIR)

        # Resume: skip already labeled
        if rel_path in existing:
            continue

        # Stop when target is reached for both classes
        if pain_count >= TARGET_PER_CLASS and no_pain_count >= TARGET_PER_CLASS:
            print(f"\nTarget reached: {pain_count} pain, {no_pain_count} no_pain. Done!")
            break

        show_image(image_path)

        while True:
            prompt = (
                f"[pain={pain_count}/{TARGET_PER_CLASS}  no_pain={no_pain_count}/{TARGET_PER_CLASS}] "
                f"{os.path.basename(image_path)} "
                f"[p=pain / n=no_pain / s=skip / q=quit]: "
            )
            choice = input(prompt).strip().lower()

            if choice == "p":
                if pain_count >= TARGET_PER_CLASS:
                    print(f"  Pain class already full ({TARGET_PER_CLASS}). Skipping.")
                    break
                append_annotation(rel_path, "pain")
                existing[rel_path] = "pain"
                pain_count += 1
                break
            elif choice == "n":
                if no_pain_count >= TARGET_PER_CLASS:
                    print(f"  No-pain class already full ({TARGET_PER_CLASS}). Skipping.")
                    break
                append_annotation(rel_path, "no_pain")
                existing[rel_path] = "no_pain"
                no_pain_count += 1
                break
            elif choice == "s":
                break
            elif choice == "q":
                print(f"\nSession ended. Saved: pain={pain_count}, no_pain={no_pain_count}")
                plt.close()
                return
            else:
                print("  Invalid input. Use p / n / s / q")

    plt.close()
    print(f"\nAnnotation complete. pain={pain_count}, no_pain={no_pain_count}")
    print(f"Labels saved to: {ANNOTATIONS_CSV}")


if __name__ == "__main__":
    main()
