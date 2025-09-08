#!/usr/bin/env python3
"""
Validate CustomDataset files and generate ImageSets/train.txt and val.txt.

Checks each points .npy and labels .txt for basic validity, filters to the
intersection of valid IDs, shuffles, and splits into train/val by ratio
(default 80/20). Useful to ensure the dataset won't crash during training.

Usage:
  python tools/validate_and_make_imagesets.py \
      --data-root /abs/path/to/dataset/root \
      --train-ratio 0.8 \
      --require-labels \
      --seed 42 \
      --allowed-classes belt_loader high_loader Other_vehicle \
      --require-at-least-one-allowed

Dataset layout:
  <data-root>/
    ├─ points/<id>.npy
    ├─ labels/<id>.txt
    └─ ImageSets/ (created/overwritten)

Exit codes:
  0 = success, 1 = validation found 0 valid samples, 2 = bad args/path
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np


@dataclass
class ValidationStats:
    points_ok: Set[str]
    points_bad: Dict[str, str]
    labels_ok: Set[str]
    labels_bad: Dict[str, str]


def list_ids_from_dir(dir_path: Path, suffix: str) -> Set[str]:
    return {p.stem for p in dir_path.glob(f"*{suffix}") if p.is_file()}


def validate_points(points_dir: Path, ids: Set[str], max_logs: int = 20) -> Tuple[Set[str], Dict[str, str]]:
    ok: Set[str] = set()
    bad: Dict[str, str] = {}
    for i, sid in enumerate(sorted(ids)):
        fp = points_dir / f"{sid}.npy"
        try:
            arr = np.load(fp, allow_pickle=False)
            if arr.ndim != 2 or arr.shape[1] < 3:
                bad[sid] = f"bad shape {arr.shape} (expected (N,>=3))"
                continue
            if arr.size == 0:
                bad[sid] = "empty array"
                continue
            # basic finite check on first 3 columns
            pts = arr[:, :3]
            if not np.isfinite(pts).all():
                bad[sid] = "found non-finite values"
                continue
            ok.add(sid)
        except Exception as e:
            bad[sid] = f"load error: {e}"
        if i < max_logs and sid in bad:
            print(f"[points] invalid {sid}: {bad[sid]}")
    return ok, bad


def validate_label_line(line: str) -> bool:
    parts = line.strip().split()
    if len(parts) < 8:
        return False
    try:
        # first 7 are floats, last is class name
        for t in parts[:7]:
            float(t)
        # parts[7] is name (string), accept anything non-empty
        return len(parts[7]) > 0
    except Exception:
        return False


def parse_label_classes(text: str) -> List[str]:
    classes: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        classes.append(parts[7])
    return classes


def validate_labels(labels_dir: Path, ids: Set[str], max_logs: int = 20,
                    allowed_classes: Optional[Set[str]] = None,
                    require_at_least_one_allowed: bool = False
                    ) -> Tuple[Set[str], Dict[str, str], Dict[str, List[str]]]:
    ok: Set[str] = set()
    bad: Dict[str, str] = {}
    id_to_classes: Dict[str, List[str]] = {}
    for i, sid in enumerate(sorted(ids)):
        fp = labels_dir / f"{sid}.txt"
        try:
            text = fp.read_text().strip()
            if text == "":
                bad[sid] = "empty label file"
                continue
            good = True
            for ln in text.splitlines():
                if ln.strip() == "":
                    continue
                if not validate_label_line(ln):
                    good = False
                    break
            if not good:
                bad[sid] = "invalid line format (expect 7 floats + class name)"
                continue
            classes = parse_label_classes(text)
            id_to_classes[sid] = classes
            if allowed_classes is not None and require_at_least_one_allowed:
                if not any(c in allowed_classes for c in classes):
                    bad[sid] = "no allowed classes in labels"
                    continue
            ok.add(sid)
        except FileNotFoundError:
            bad[sid] = "missing label file"
        except Exception as e:
            bad[sid] = f"read/parse error: {e}"
        if i < max_logs and sid in bad:
            print(f"[labels] invalid {sid}: {bad[sid]}")
    return ok, bad, id_to_classes


def write_imagesets(root: Path, train_ids: List[str], val_ids: List[str]) -> None:
    imgsets = root / "ImageSets"
    imgsets.mkdir(parents=True, exist_ok=True)
    (imgsets / "train.txt").write_text("\n".join(train_ids) + "\n")
    (imgsets / "val.txt").write_text("\n".join(val_ids) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate CustomDataset and generate ImageSets train/val splits")
    ap.add_argument("--data-root", type=str, required=True, help="Dataset root containing points/, labels/")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (0-1), default 0.8")
    ap.add_argument("--require-labels", action="store_true", help="Require valid labels for inclusion")
    ap.add_argument("--allowed-classes", nargs="*", default=None,
                    help="Optional list of allowed class names; if provided with --require-at-least-one-allowed, only frames containing at least one of these will be kept")
    ap.add_argument("--require-at-least-one-allowed", action="store_true",
                    help="When used with --allowed-classes and --require-labels, keep only frames that contain at least one allowed class")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling")
    args = ap.parse_args()

    root = Path(args.data_root)
    points_dir = root / "points"
    labels_dir = root / "labels"
    if not points_dir.is_dir():
        print(f"[error] points dir not found: {points_dir}")
        return 2
    if args.require_labels and not labels_dir.is_dir():
        print(f"[error] labels dir not found (required): {labels_dir}")
        return 2
    if not (0.0 < args.train_ratio < 1.0):
        print("[error] --train-ratio must be in (0,1)")
        return 2

    point_ids = list_ids_from_dir(points_dir, ".npy")
    label_ids = list_ids_from_dir(labels_dir, ".txt") if labels_dir.is_dir() else set()

    # Start from the union and validate
    candidate_ids = point_ids if not args.require_labels else (point_ids & label_ids)
    if not candidate_ids:
        print("[error] No candidate IDs found (check directories and flags)")
        return 1

    print(f"Found candidate IDs: {len(candidate_ids)} (points={len(point_ids)}, labels={len(label_ids)})")

    pts_ok, pts_bad = validate_points(points_dir, candidate_ids)
    if args.require_labels:
        allowed_set = set(args.allowed_classes) if args.allowed_classes else None
        lbl_ok, lbl_bad, id_to_classes = validate_labels(
            labels_dir, candidate_ids,
            allowed_classes=allowed_set,
            require_at_least_one_allowed=args.require_at_least_one_allowed,
        )
    else:
        lbl_ok, lbl_bad = set(), {}

    valid_ids = pts_ok if not args.require_labels else (pts_ok & lbl_ok)

    print("Validation summary:")
    print(f"  points: ok={len(pts_ok)}, bad={len(pts_bad)}")
    if args.require_labels:
        print(f"  labels: ok={len(lbl_ok)}, bad={len(lbl_bad)}")
        if args.allowed_classes:
            print(f"  allowed classes filter: {args.allowed_classes}; require_at_least_one={args.require_at_least_one_allowed}")
    print(f"  valid IDs for splitting: {len(valid_ids)}")

    if len(valid_ids) == 0:
        print("[error] No valid IDs after validation")
        return 1

    # Deterministic shuffle
    rng = np.random.default_rng(args.seed)
    valid_ids_list = sorted(valid_ids)
    rng.shuffle(valid_ids_list)
    n_total = len(valid_ids_list)
    n_train = int(round(args.train_ratio * n_total))
    n_train = max(1, min(n_train, n_total - 1))
    train_ids = valid_ids_list[:n_train]
    val_ids = valid_ids_list[n_train:]

    write_imagesets(root, train_ids, val_ids)
    print("Wrote ImageSets:")
    print(f"  train.txt: {len(train_ids)}")
    print(f"  val.txt:   {len(val_ids)}")

    # Optionally, dump bad ID lists for debugging
    debug_dir = root / "ImageSets"
    if pts_bad:
        (debug_dir / "_bad_points.txt").write_text("\n".join(f"{k}\t{v}" for k, v in sorted(pts_bad.items())) + "\n")
    if lbl_bad:
        (debug_dir / "_bad_labels.txt").write_text("\n".join(f"{k}\t{v}" for k, v in sorted(lbl_bad.items())) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
