"""
prepare_data.py — Build data/pain/ and data/no_pain/ from data/annotations.csv (Step 1.5)

Usage:
    python scripts/prepare_data.py            # copy images into class folders
    python scripts/prepare_data.py --dry-run  # report what would happen, copy nothing
    python scripts/prepare_data.py --symlink  # symlink instead of copy (saves disk)
    python scripts/prepare_data.py --force    # re-copy even files already present

Why this exists:
    scripts/annotate.py writes labels to data/annotations.csv (paths under
    data/raw/), but src/dataset.py loads images from data/pain/ and
    data/no_pain/. This script bridges the two so `python src/train.py` works.

    It is idempotent: re-running it only copies files that are new or changed,
    so it is safe to run again after each annotation session.

Naming: outputs are prefixed with the source subfolder, e.g.
    raw/CAT_00/00000001_000.jpg -> CAT_00_00000001_000.jpg
so identically named frames from different cats (CAT_00 ... CAT_06) never
overwrite each other.
"""

import argparse
import csv
import os
import shutil
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
ANNOTATIONS_CSV = os.path.join(DATA_DIR, "annotations.csv")
VALID_LABELS = ("pain", "no_pain")


def dest_name(rel_path):
    """raw/CAT_00/00000001_000.jpg -> CAT_00_00000001_000.jpg."""
    parts = rel_path.replace(os.sep, "/").split("/")
    if len(parts) >= 3:  # raw/<cat>/<file>
        return f"{parts[1]}_{parts[-1]}"
    return parts[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Copy labeled images from data/raw/ into data/pain/ and "
                    "data/no_pain/ according to data/annotations.csv."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen without copying anything")
    parser.add_argument("--symlink", action="store_true",
                        help="create symlinks instead of copies (saves disk)")
    parser.add_argument("--force", action="store_true",
                        help="re-copy files even if already present")
    args = parser.parse_args()

    if not os.path.exists(ANNOTATIONS_CSV):
        print(f"ERROR: annotations file not found: {ANNOTATIONS_CSV}")
        print("Run scripts/annotate.py first to create it.")
        return 1

    with open(ANNOTATIONS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("annotations.csv is empty — nothing to do.")
        return 0

    copied = {"pain": 0, "no_pain": 0}
    skipped = 0
    missing = []
    bad_labels = []

    for row in rows:
        rel_path = (row.get("filename") or "").strip()
        label = (row.get("label") or "").strip()

        if label not in VALID_LABELS:
            bad_labels.append((rel_path, label))
            continue

        src = os.path.join(DATA_DIR, rel_path)
        if not os.path.isfile(src):
            missing.append(rel_path)
            continue

        dest_dir = os.path.join(DATA_DIR, label)
        dest = os.path.join(dest_dir, dest_name(rel_path))

        # idempotency: skip files already present with the same size
        if os.path.exists(dest) and not args.force:
            try:
                if os.path.getsize(dest) == os.path.getsize(src):
                    skipped += 1
                    continue
            except OSError:
                pass

        if args.dry_run:
            copied[label] += 1
            continue

        os.makedirs(dest_dir, exist_ok=True)
        if os.path.lexists(dest):
            os.remove(dest)
        if args.symlink:
            os.symlink(os.path.abspath(src), dest)
        else:
            shutil.copy2(src, dest)
        copied[label] += 1

    verb = "would copy" if args.dry_run else ("linked" if args.symlink else "copied")
    print(f"\nAnnotations: {len(rows)}  "
          f"(pain={sum(1 for r in rows if r.get('label') == 'pain')}, "
          f"no_pain={sum(1 for r in rows if r.get('label') == 'no_pain')})")
    print(f"{verb.capitalize()}: pain={copied['pain']}, no_pain={copied['no_pain']}, "
          f"total={copied['pain'] + copied['no_pain']}")
    print(f"Already present (skipped): {skipped}")

    if missing:
        print(f"\nWARNING: {len(missing)} labeled file(s) not found under data/:")
        for p in missing[:20]:
            print(f"  - {p}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if bad_labels:
        print(f"\nWARNING: {len(bad_labels)} row(s) with unexpected labels (skipped):")
        for p, lab in bad_labels[:10]:
            print(f"  - {p}: {lab!r}")

    if not args.dry_run and not missing:
        print("\nDone. Next step: python src/train.py")
    elif not args.dry_run:
        print("\nDone with warnings — missing files were skipped, the rest are ready.")
        print("Next step: python src/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
