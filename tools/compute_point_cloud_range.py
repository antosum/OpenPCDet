#!/usr/bin/env python3
"""
Compute POINT_CLOUD_RANGE for an OpenPCDet custom dataset.

This scans `.npy` point clouds in `points/` and computes the global min/max
for x, y, z. It prints a YAML snippet and can optionally update a dataset
YAML in-place.

Dataset layout (per docs/CUSTOM_DATASET_TUTORIAL.md):
    data/custom/
      ├─ ImageSets/{train.txt,val.txt,...}
      ├─ points/{000000.npy, ...}
      └─ labels/{000000.txt, ...}

Basic usage:
    # Use all frames under points/
    python tools/compute_point_cloud_range.py data/custom

    # Use only frames listed in ImageSets splits
    python tools/compute_point_cloud_range.py data/custom --splits train,val

    # Use explicit ID list(s)
    python tools/compute_point_cloud_range.py data/custom \
        --ids-file data/custom/ImageSets/train.txt \
        --ids-file data/custom/ImageSets/val.txt

Handle outliers:
    # Ignore the most extreme 1% per-axis via quantiles
    python tools/compute_point_cloud_range.py data/custom --quantile 0.01

Pillar/voxel alignment (e.g., Dynamic CenterPoint Pillars):
    # Align XY to multiples of 16 cells with voxel size 0.2m and make Z 1 cell of 8m,
    # centering XY around 0 and expanding symmetrically.
    python tools/compute_point_cloud_range.py data/custom \
        --splits train,val \
        --voxel-size 0.2,0.2,8.0 \
        --xy-multiple 16 \
        --symmetric-xy \
        --z-cells 1

Update a dataset YAML in-place:
    python tools/compute_point_cloud_range.py data/custom \
        --splits train,val \
        --quantile 0.01 \
        --update-yaml tools/cfgs/dataset_configs/geminai_dataset.yaml

Exit codes:
    0 = success; 1 = failure updating YAML; 2 = bad args / missing files.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def find_points_dir(dataset_dir: Path) -> Path:
    """Return the points directory. Accepts either the dataset root or the points/ dir itself."""
    if (dataset_dir / 'points').is_dir():
        return dataset_dir / 'points'
    if dataset_dir.is_dir() and dataset_dir.name == 'points':
        return dataset_dir
    raise FileNotFoundError(f"Couldn't find 'points/' under {dataset_dir}. Provide the dataset root (containing points/) or points/ directly.")


def load_ids_from_imagesets(dataset_dir: Path, splits: Sequence[str]) -> List[str]:
    imagesets = dataset_dir / 'ImageSets'
    if not imagesets.is_dir():
        raise FileNotFoundError(f"ImageSets/ not found under {dataset_dir}")
    ids: List[str] = []
    for split in splits:
        p = imagesets / f"{split}.txt"
        if not p.is_file():
            raise FileNotFoundError(f"Split file not found: {p}")
        with p.open('r') as f:
            for line in f:
                s = line.strip()
                if s:
                    ids.append(s)
    return ids


def iter_point_files(points_dir: Path, ids: Optional[Iterable[str]] = None) -> Iterable[Path]:
    if ids is None:
        yield from sorted(points_dir.glob('*.npy'))
        return
    id_set = set(ids)
    for id_ in sorted(id_set):
        p = points_dir / f"{id_}.npy"
        if p.is_file():
            yield p
        else:
            # Allow IDs that already include extension
            p_alt = points_dir / id_
            if p_alt.suffix == '.npy' and p_alt.is_file():
                yield p_alt
            else:
                print(f"[warn] Missing point file for id '{id_}' (expected {p})", file=sys.stderr)


def update_yaml_point_cloud_range(yaml_path: Path, pcr: Tuple[float, float, float, float, float, float], precision: int = 3) -> None:
    content = yaml_path.read_text()
    prefix_pattern = re.compile(r"^(\s*POINT_CLOUD_RANGE:\s*)\[[^\]]*\]", re.MULTILINE)
    formatted = (
        f"[{pcr[0]:.{precision}f}, {pcr[1]:.{precision}f}, {pcr[2]:.{precision}f}, "
        f"{pcr[3]:.{precision}f}, {pcr[4]:.{precision}f}, {pcr[5]:.{precision}f}]"
    )

    def repl(m: re.Match) -> str:
        return f"{m.group(1)}{formatted}"

    new_content, n = prefix_pattern.subn(repl, content, count=1)
    if n == 0:
        raise ValueError(f"POINT_CLOUD_RANGE key not found in {yaml_path}")
    yaml_path.write_text(new_content)


def compute_range(
    points_dir: Path,
    files: Iterable[Path],
    quantile: float = 0.0,
    log_per_file: bool = False,
) -> Tuple[float, float, float, float, float, float]:
    q = float(quantile)
    if not (0.0 <= q < 0.5):
        raise ValueError("quantile must be in [0, 0.5)")

    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    # For quantiles, collect per-file mins/maxs to reduce memory usage
    per_file_extents: List[np.ndarray] = [] if q > 0 else None

    count = 0
    for fp in files:
        try:
            arr = np.load(fp, allow_pickle=False)
        except Exception as e:
            print(f"[warn] Failed to load {fp}: {e}", file=sys.stderr)
            continue
        if arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] < 3:
            print(f"[warn] Unexpected array shape in {fp}: {arr.shape}, expected (N, >=3)", file=sys.stderr)
            continue
        pts = arr[:, :3].astype(np.float64, copy=False)
        mask = np.isfinite(pts).all(axis=1)
        if not mask.any():
            continue
        pts = pts[mask]

        if q == 0.0:
            pmin = np.min(pts, axis=0)
            pmax = np.max(pts, axis=0)
            if log_per_file:
                print(
                    f"{fp.name}: X[{pmin[0]:.3f},{pmax[0]:.3f}] "
                    f"Y[{pmin[1]:.3f},{pmax[1]:.3f}] "
                    f"Z[{pmin[2]:.3f},{pmax[2]:.3f}]"
                )
            mins = np.minimum(mins, pmin)
            maxs = np.maximum(maxs, pmax)
        else:
            lo = np.quantile(pts, q, axis=0)
            hi = np.quantile(pts, 1 - q, axis=0)
            if log_per_file:
                print(
                    f"{fp.name} (q={q:.3f}): "
                    f"X[{lo[0]:.3f},{hi[0]:.3f}] "
                    f"Y[{lo[1]:.3f},{hi[1]:.3f}] "
                    f"Z[{lo[2]:.3f},{hi[2]:.3f}]"
                )
            per_file_extents.append(np.stack([lo, hi], axis=0))
        count += 1

    if count == 0:
        raise RuntimeError(f"No valid point files found under {points_dir}")

    if q == 0.0:
        return float(mins[0]), float(mins[1]), float(mins[2]), float(maxs[0]), float(maxs[1]), float(maxs[2])
    else:
        # Aggregate quantiles across files by taking min of per-file lows and max of per-file highs
        ext = np.stack(per_file_extents, axis=0)  # (F, 2, 3)
        lo = np.min(ext[:, 0, :], axis=0)
        hi = np.max(ext[:, 1, :], axis=0)
        return float(lo[0]), float(lo[1]), float(lo[2]), float(hi[0]), float(hi[1]), float(hi[2])


def align_range_to_voxels(
    pcr: Tuple[float, float, float, float, float, float],
    voxel_size: Optional[Tuple[float, float, float]] = None,
    xy_multiple: Optional[int] = None,
    symmetric_xy: bool = False,
    z_cells: Optional[int] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Adjust the computed range to align with pillar/voxel constraints.

    - symmetric_xy: expand to be centered around 0 on X and Y.
    - xy_multiple + voxel_size: ensure (extent_x / vx) and (extent_y / vy) are multiples of xy_multiple.
    - z_cells + voxel_size: ensure (extent_z / vz) equals z_cells by symmetric expansion.
    """
    minx, miny, minz, maxx, maxy, maxz = pcr
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    cz = 0.5 * (minz + maxz)

    if symmetric_xy:
        rx = max(abs(minx), abs(maxx))
        ry = max(abs(miny), abs(maxy))
        minx, maxx = -rx, rx
        miny, maxy = -ry, ry
        cx, cy = 0.0, 0.0

    if voxel_size and xy_multiple:
        vx, vy, vz = voxel_size
        # X
        extent_x = maxx - minx
        cells_x = int(np.ceil(extent_x / max(vx, 1e-9)))
        cells_x_aligned = int(np.ceil(cells_x / xy_multiple) * xy_multiple)
        if cells_x_aligned != cells_x:
            need_x = cells_x_aligned * vx
            minx = cx - need_x / 2.0
            maxx = cx + need_x / 2.0
        # Y
        extent_y = maxy - miny
        cells_y = int(np.ceil(extent_y / max(vy, 1e-9)))
        cells_y_aligned = int(np.ceil(cells_y / xy_multiple) * xy_multiple)
        if cells_y_aligned != cells_y:
            need_y = cells_y_aligned * vy
            miny = cy - need_y / 2.0
            maxy = cy + need_y / 2.0

    if voxel_size and z_cells:
        vx, vy, vz = voxel_size
        extent_z = maxz - minz
        need_z = z_cells * vz
        if need_z < extent_z:
            # If current z span exceeds requested cells, expand to the next multiple
            z_cells_needed = int(np.ceil(extent_z / max(vz, 1e-9)))
            if z_cells_needed % max(z_cells, 1) != 0:
                z_cells_needed = int(np.ceil(z_cells_needed / z_cells) * z_cells)
            need_z = z_cells_needed * vz
        minz = cz - need_z / 2.0
        maxz = cz + need_z / 2.0

    return (float(minx), float(miny), float(minz), float(maxx), float(maxy), float(maxz))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute POINT_CLOUD_RANGE for an OpenPCDet custom dataset (points stored as .npy)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # All frames in points/\n"
            "  python tools/compute_point_cloud_range.py data/custom\n\n"
            "  # Only ImageSets splits\n"
            "  python tools/compute_point_cloud_range.py data/custom --splits train,val\n\n"
            "  # Ignore extreme 1% outliers and update YAML\n"
            "  python tools/compute_point_cloud_range.py data/custom --quantile 0.01 \\\n+            \n    --update-yaml tools/cfgs/dataset_configs/geminai_dataset.yaml\n\n"
            "  # Dynamic CenterPoint Pillars alignment (voxel 0.2,0.2,8.0; XY multiple 16; Z=1 cell; symmetric XY)\n"
            "  python tools/compute_point_cloud_range.py data/custom --splits train,val \\\n+            \n    --voxel-size 0.2,0.2,8.0 --xy-multiple 16 --symmetric-xy --z-cells 1\n"
        ),
    )
    parser.add_argument(
        'dataset_dir',
        type=str,
        help="Path to dataset root (containing points/ and ImageSets/) or the points/ directory",
    )
    parser.add_argument(
        '--splits',
        type=str,
        default=None,
        help="Comma-separated list of splits from ImageSets to include (e.g., 'train,val'). Default: use all .npy in points/",
    )
    parser.add_argument(
        '--ids-file',
        type=str,
        action='append',
        default=None,
        help="Optional path(s) to files listing frame IDs to include (one per line). Overrides --splits if provided.",
    )
    parser.add_argument(
        '--quantile',
        type=float,
        default=0.0,
        help="Optional lower/upper quantile to ignore extreme outliers (0 <= q < 0.5). 0.0 uses strict min/max.",
    )
    parser.add_argument(
        '--log-per-file', action='store_true',
        help='Log per-file x/y/z range while scanning (verbose).'
    )
    parser.add_argument(
        '--precision', type=int, default=3, help='Decimal places for printed YAML line and optional updates.'
    )
    # Optional decoder clipping range (POST_CENTER_LIMIT_RANGE) helpers
    parser.add_argument(
        '--post-xy-margin', type=float, default=0.0,
        help='XY padding (meters) applied to POINT_CLOUD_RANGE when printing POST_CENTER_LIMIT_RANGE.'
    )
    parser.add_argument(
        '--post-z-margin', type=float, default=None,
        help='Z symmetric padding (meters). Used for both min and max unless side-specific margins are given.'
    )
    parser.add_argument(
        '--post-z-min-margin', type=float, default=0.0,
        help='Z min-side padding (meters) for POST_CENTER_LIMIT_RANGE.'
    )
    parser.add_argument(
        '--post-z-max-margin', type=float, default=0.0,
        help='Z max-side padding (meters) for POST_CENTER_LIMIT_RANGE.'
    )
    parser.add_argument(
        '--voxel-size', type=str, default=None,
        help="Optional voxel size as 'vx,vy,vz' (e.g., 0.2,0.2,8.0). Enables alignment options."
    )
    parser.add_argument(
        '--xy-multiple', type=int, default=None,
        help="When used with --voxel-size, expand XY extents so (extent/v) is a multiple of this value (e.g., 16 for pillars)."
    )
    parser.add_argument(
        '--symmetric-xy', action='store_true',
        help='Center XY around 0 and expand symmetrically to cover all points.'
    )
    parser.add_argument(
        '--z-cells', type=int, default=None,
        help='When used with --voxel-size, adjust Z extent so (extent_z / vz) equals this many cells (e.g., 1 for pillars).'
    )
    parser.add_argument(
        '--update-yaml',
        type=str,
        default=None,
        help="Optional path to a dataset YAML to update the POINT_CLOUD_RANGE line in-place.",
    )

    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    try:
        points_dir = find_points_dir(dataset_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    ids: Optional[List[str]] = None
    if args.ids_file:
        ids = []
        for fp in args.ids_file:
            with open(fp, 'r') as f:
                ids.extend([ln.strip() for ln in f if ln.strip()])
    elif args.splits:
        try:
            ids = load_ids_from_imagesets(dataset_dir if (dataset_dir / 'ImageSets').is_dir() else points_dir.parent, [s.strip() for s in args.splits.split(',') if s.strip()])
        except FileNotFoundError as e:
            print(f"[warn] {e}. Falling back to all files in points/.", file=sys.stderr)
            ids = None

    files = list(iter_point_files(points_dir, ids))
    if not files:
        print(f"No point files found under {points_dir} (after applying filters).", file=sys.stderr)
        return 2

    minx, miny, minz, maxx, maxy, maxz = compute_range(
        points_dir, files, quantile=args.quantile, log_per_file=args.log_per_file
    )

    voxel_size_tuple: Optional[Tuple[float, float, float]] = None
    if args.voxel_size:
        try:
            parts = [float(x) for x in args.voxel_size.split(',')]
            if len(parts) != 3:
                raise ValueError
            voxel_size_tuple = (parts[0], parts[1], parts[2])
        except Exception:
            print("[error] --voxel-size must be in form 'vx,vy,vz' (e.g., 0.2,0.2,8.0)", file=sys.stderr)
            return 2

    if voxel_size_tuple or args.symmetric_xy:
        minx, miny, minz, maxx, maxy, maxz = align_range_to_voxels(
            (minx, miny, minz, maxx, maxy, maxz),
            voxel_size=voxel_size_tuple,
            xy_multiple=args.xy_multiple,
            symmetric_xy=args.symmetric_xy,
            z_cells=args.z_cells,
        )

    print(f"Scanned {len(files)} frames from: {points_dir}")
    print(f"X: [{minx:.{args.precision}f}, {maxx:.{args.precision}f}]  "
          f"Y: [{miny:.{args.precision}f}, {maxy:.{args.precision}f}]  "
          f"Z: [{minz:.{args.precision}f}, {maxz:.{args.precision}f}]")
    print()
    print("YAML snippet:")
    print(f"POINT_CLOUD_RANGE: [{minx:.{args.precision}f}, {miny:.{args.precision}f}, {minz:.{args.precision}f}, "
          f"{maxx:.{args.precision}f}, {maxy:.{args.precision}f}, {maxz:.{args.precision}f}]")

    # Compute and print a suggested POST_CENTER_LIMIT_RANGE
    zmin_pad = args.post_z_min_margin
    zmax_pad = args.post_z_max_margin
    if args.post_z_margin is not None:
        if '--post-z-min-margin' not in sys.argv:
            zmin_pad = float(args.post_z_margin)
        if '--post-z-max-margin' not in sys.argv:
            zmax_pad = float(args.post_z_margin)

    post_minx = minx - args.post_xy_margin
    post_miny = miny - args.post_xy_margin
    post_minz = minz - zmin_pad
    post_maxx = maxx + args.post_xy_margin
    post_maxy = maxy + args.post_xy_margin
    post_maxz = maxz + zmax_pad

    print()
    print("Decoder clipping snippet:")
    print(
        f"POST_CENTER_LIMIT_RANGE: [{post_minx:.{args.precision}f}, {post_miny:.{args.precision}f}, {post_minz:.{args.precision}f}, "
        f"{post_maxx:.{args.precision}f}, {post_maxy:.{args.precision}f}, {post_maxz:.{args.precision}f}]"
    )

    if args.update_yaml:
        yaml_path = Path(args.update_yaml)
        if not yaml_path.is_file():
            print(f"[error] YAML file not found: {yaml_path}", file=sys.stderr)
            return 2
        try:
            update_yaml_point_cloud_range(
                yaml_path,
                (minx, miny, minz, maxx, maxy, maxz),
                precision=args.precision,
            )
            print(f"\nUpdated POINT_CLOUD_RANGE in: {yaml_path}")
        except Exception as e:
            print(f"[error] Failed to update YAML: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
