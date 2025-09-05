#!/usr/bin/env python3
"""
Merge multiple OpenPCDet custom-format datasets into a single target dataset.

Assumptions:
- Each source directory contains `points/` with .npy files and `labels/` with .txt files.
- For each .npy there should be a corresponding .txt with the same stem.
- Target directory will contain `points/` and `labels/` with zero-padded numeric filenames.

Features:
- Reindexes all samples sequentially to avoid filename collisions.
- Continues indexing from existing target content, if any.
- Validates missing pairs with `--on-missing` policy (error or skip).
- Copy or symlink files, with dry-run support.
- Optionally writes a CSV mapping of original -> merged filenames.
 - Optional ImageSets generation (train/val[/test]) with shuffle/ratios.

Example:
  python tools/merge_custom_datasets.py \
      /data/src1 /data/src2 /data/merged_custom \
      --mode copy --on-missing error --pad 6 \
      --imagesets all --split-ratio 80:20 --seed 42

Resulting target structure:
  /data/merged_custom/
    ├── points/
    │   ├── 000000.npy
    │   └── ...
    └── labels/
        ├── 000000.txt
        └── ...

See docs/CUSTOM_DATASET_TUTORIAL.md for format details.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional


NPY_RE = re.compile(r"^(?P<stem>\d+)\.npy$")
TXT_RE = re.compile(r"^(?P<stem>\d+)\.txt$")


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def find_numeric_stems(dirpath: Path, pattern: re.Pattern) -> Dict[str, Path]:
    """Return mapping from numeric stem -> file path for files matching the regex pattern.

    Only includes files where the stem is all digits.
    """
    out: Dict[str, Path] = {}
    if not dirpath.exists():
        return out
    for p in dirpath.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if m:
            out[m.group("stem")] = p
    return out


def detect_padding_digits(points_dir: Path, default: int = 6) -> int:
    """Detect zero-padding width from existing files in `points_dir`, else return default."""
    stems = []
    for p in points_dir.glob("*.npy"):
        m = NPY_RE.match(p.name)
        if m:
            stems.append(m.group("stem"))
    if not stems:
        return default
    # Use the most common length; fall back to max length
    lengths: Dict[int, int] = {}
    for s in stems:
        lengths[len(s)] = lengths.get(len(s), 0) + 1
    # most common length
    pad = max(lengths.items(), key=lambda kv: kv[1])[0]
    return pad


def detect_next_index(points_dir: Path) -> int:
    """Return the next integer index after the max existing numeric stem in `points_dir`."""
    max_idx = -1
    for p in points_dir.glob("*.npy"):
        m = NPY_RE.match(p.name)
        if not m:
            continue
        try:
            idx = int(m.group("stem"))
            if idx > max_idx:
                max_idx = idx
        except ValueError:
            continue
    return max_idx + 1


def numeric_sort_key(stem: str):
    try:
        return int(stem)
    except ValueError:
        return stem


def collect_pairs(src_dir: Path, on_missing: str, verbose: bool = False) -> List[Tuple[Path, Path, str]]:
    """Collect (points_path, labels_path, stem) tuples from a source dataset directory.

    `on_missing`: 'error' or 'skip' when encountering pairs with one side missing.
    """
    points_dir = src_dir / "points"
    labels_dir = src_dir / "labels"
    if not points_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Source {src_dir} must contain 'points/' and 'labels/'")

    points = find_numeric_stems(points_dir, NPY_RE)
    labels = find_numeric_stems(labels_dir, TXT_RE)

    all_stems = sorted(points.keys(), key=numeric_sort_key)
    pairs: List[Tuple[Path, Path, str]] = []
    missing = 0
    for stem in all_stems:
        p = points.get(stem)
        l = labels.get(stem)
        if p is None or l is None:
            missing += 1
            msg = f"Missing pair in {src_dir}: stem {stem} -> points={p is not None}, labels={l is not None}"
            if on_missing == "error":
                raise FileNotFoundError(msg)
            if verbose:
                eprint("[skip]", msg)
            continue
        pairs.append((p, l, stem))
    if verbose and missing:
        eprint(f"[warn] Skipped {missing} samples in {src_dir} due to missing pairs")
    return pairs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # Create relative symlink if possible
        try:
            rel = os.path.relpath(src, start=dst.parent)
            dst.symlink_to(rel)
        except Exception:
            # Fallback to absolute symlink
            dst.symlink_to(src)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def merge_datasets(
    sources: List[Path],
    target: Path,
    mode: str = "copy",
    on_missing: str = "error",
    pad: Optional[int] = None,
    start_index: Optional[int] = None,
    dry_run: bool = False,
    mapping_out: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """Merge multiple source datasets into target. Returns summary counts."""
    if mode not in {"copy", "symlink"}:
        raise ValueError("mode must be 'copy' or 'symlink'")
    if on_missing not in {"error", "skip"}:
        raise ValueError("on_missing must be 'error' or 'skip'")

    tgt_points = target / "points"
    tgt_labels = target / "labels"

    if pad is None:
        pad = detect_padding_digits(tgt_points, default=6)

    if start_index is None:
        start_index = detect_next_index(tgt_points)

    if verbose:
        eprint(f"Target: {target}")
        eprint(f"Mode: {mode}, On-missing: {on_missing}, Pad: {pad}, Start-index: {start_index}")
        eprint("Preparing target directories ...")

    if not dry_run:
        ensure_dir(tgt_points)
        ensure_dir(tgt_labels)

    mapping_fp = None
    mapping_writer = None
    if mapping_out is not None and not dry_run:
        mapping_fp = mapping_out.open("w", newline="")
        mapping_writer = csv.writer(mapping_fp)
        mapping_writer.writerow([
            "target_index",
            "target_points",
            "target_labels",
            "source_dir",
            "source_stem",
            "source_points",
            "source_labels",
        ])

    total_pairs = 0
    written = 0
    skipped = 0

    next_idx = start_index
    created_names: List[str] = []  # zero-padded stems created in this run

    for src in sources:
        pairs = collect_pairs(src, on_missing=on_missing, verbose=verbose)
        if verbose:
            eprint(f"Collected {len(pairs)} pairs from {src}")
        for p_path, l_path, stem in pairs:
            total_pairs += 1
            new_name = str(next_idx).zfill(pad)
            dst_p = tgt_points / f"{new_name}.npy"
            dst_l = tgt_labels / f"{new_name}.txt"

            if dst_p.exists() or dst_l.exists():
                # This should not happen with sequential indexing, but guard anyway
                skipped += 1
                if verbose:
                    eprint(f"[skip] Target exists: {dst_p.name} or {dst_l.name}")
                next_idx += 1
                continue

            if not dry_run:
                copy_or_link(p_path, dst_p, mode)
                copy_or_link(l_path, dst_l, mode)
                if mapping_writer is not None:
                    mapping_writer.writerow([
                        new_name,
                        str(dst_p),
                        str(dst_l),
                        str(src),
                        stem,
                        str(p_path),
                        str(l_path),
                    ])

            if verbose:
                eprint(f"[{mode}] {p_path.name} -> {dst_p.name}; {l_path.name} -> {dst_l.name}")
            written += 1
            next_idx += 1
            created_names.append(new_name)

    if mapping_fp is not None:
        mapping_fp.close()

    return {
        "total_pairs": total_pairs,
        "written": written,
        "skipped": skipped,
        "start_index": start_index or 0,
        "end_index": next_idx - 1 if written > 0 else (start_index or 0) - 1,
        "pad": pad,
        "created_names": created_names,
    }


def write_imagesets(
    target: Path,
    stems: List[str],
    split_ratio: Tuple[int, int, int],
    seed: Optional[int] = None,
    verbose: bool = False,
) -> None:
    import random

    img_dir = target / "ImageSets"
    ensure_dir(img_dir)

    # Filter numeric stems and sort for stability before shuffle
    stems = [s for s in stems if s.isdigit()]
    stems.sort(key=numeric_sort_key)

    if seed is not None:
        random.seed(seed)
    random.shuffle(stems)

    t, v, te = split_ratio
    total = len(stems)
    if t + v + te <= 0:
        raise ValueError("split_ratio must have a positive sum")

    # Compute counts proportionally
    def portion(n: int, num: int, den: int) -> int:
        return (n * num) // den

    den = t + v + te
    n_train = portion(total, t, den)
    n_val = portion(total, v, den)
    n_test = total - n_train - n_val

    train = stems[:n_train]
    val = stems[n_train:n_train + n_val]
    test = stems[n_train + n_val:]

    # Writers
    (img_dir / "train.txt").write_text("\n".join(train) + ("\n" if train else ""))
    (img_dir / "val.txt").write_text("\n".join(val) + ("\n" if val else ""))
    if te > 0:
        (img_dir / "test.txt").write_text("\n".join(test) + ("\n" if test else ""))

    if verbose:
        eprint(f"ImageSets written to {img_dir} (train={len(train)}, val={len(val)}, test={len(test)})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", type=Path, help="One or more source dataset directories (each with points/ and labels/)")
    p.add_argument("target", type=Path, help="Target dataset directory to write merged points/ and labels/")
    p.add_argument("--mode", choices=["copy", "symlink"], default="copy", help="How to place files into target (default: copy)")
    p.add_argument("--on-missing", choices=["error", "skip"], default="error", help="Behavior when a pair is missing (default: error)")
    p.add_argument("--pad", type=int, default=None, help="Zero-padding width for target filenames (default: auto-detect or 6)")
    p.add_argument("--start-index", type=int, default=None, help="Starting index for new files (default: continue from target)")
    p.add_argument("--dry-run", action="store_true", help="Do not write anything; just print planned actions")
    p.add_argument("--mapping-out", type=Path, default=None, help="Optional path to write CSV mapping of merges")
    p.add_argument("--verbose", action="store_true", help="Verbose logging to stderr")
    # ImageSets options
    p.add_argument("--imagesets", choices=["none", "new", "all"], default="none",
                   help="Generate ImageSets from 'new' samples or 'all' in target (default: none)")
    p.add_argument("--split-ratio", default="80:20",
                   help="Split ratio train:val[:test], integers, default 80:20")
    p.add_argument("--seed", type=int, default=None, help="Random seed for split shuffling")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    sources: List[Path] = []
    for s in args.sources:
        if not s.exists():
            eprint(f"[error] Source does not exist: {s}")
            return 2
        if not (s / "points").is_dir() or not (s / "labels").is_dir():
            eprint(f"[error] Invalid source (missing points/ or labels/): {s}")
            return 2
        sources.append(s.resolve())

    target: Path = args.target.resolve()
    mapping_out: Optional[Path] = args.mapping_out
    if mapping_out is not None and mapping_out.exists() and not args.dry_run:
        eprint(f"[warn] Mapping output already exists and will be overwritten: {mapping_out}")

    try:
        summary = merge_datasets(
            sources=sources,
            target=target,
            mode=args.mode,
            on_missing=args.on_missing,
            pad=args.pad,
            start_index=args.start_index,
            dry_run=args.dry_run,
            mapping_out=mapping_out,
            verbose=args.verbose,
        )
    except Exception as e:
        eprint(f"[error] {e}")
        return 1

    print("Merge summary:")
    print(f"  Sources: {len(sources)}")
    print(f"  Target: {target}")
    print(f"  Total pairs seen: {summary['total_pairs']}")
    print(f"  Written: {summary['written']}")
    print(f"  Skipped: {summary['skipped']}")
    print(f"  Start index: {summary['start_index']}")
    print(f"  End index: {summary['end_index']}")
    print(f"  Padding: {summary['pad']}")

    if args.dry_run:
        print("(dry-run) No files were written.")

    # Handle ImageSets generation if requested (only when not dry-run)
    if not args.dry_run and args.imagesets != "none":
        # Parse split ratio
        parts = str(args.split_ratio).split(":")
        if not (2 <= len(parts) <= 3) or not all(p.isdigit() for p in parts):
            eprint("[error] --split-ratio must be like '80:20' or '70:20:10'")
            return 2
        t = int(parts[0]); v = int(parts[1]); te = int(parts[2]) if len(parts) == 3 else 0

        if args.imagesets == "new":
            stems = [str(s) for s in summary.get("created_names", [])]
        else:
            # all: gather all stems in target/points
            stems = []
            for pth in (target / "points").glob("*.npy"):
                m = NPY_RE.match(pth.name)
                if m:
                    stems.append(m.group("stem"))
        try:
            write_imagesets(target=target, stems=stems, split_ratio=(t, v, te), seed=args.seed, verbose=args.verbose)
        except Exception as e:
            eprint(f"[error] generating ImageSets: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
