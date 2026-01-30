#!/usr/bin/env python3
"""
nii_bbox.py

Utility to compute a bounding box from a NIfTI (.nii/.nii.gz) and return it in
both voxel (index) space and real/world coordinates (using a robust affine).

Key points:
- Uses a threshold on the image data to define the foreground (default > 0).
- Converts voxel bbox to world coordinates using voxel *edges*:
    min corner uses (i_min, j_min, k_min)
    max corner uses (i_max+1, j_max+1, k_max+1)
  This produces a proper physical-space bounding box for the region.
- Tries to avoid header oddities by selecting a valid sform/qform affine if present.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _require_nibabel():
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "This script requires nibabel. Install with: pip install nibabel"
        ) from e
    return nib


@dataclass(frozen=True)
class BBox:
    # voxel indices are inclusive for min/max
    voxel_min: Tuple[int, int, int]
    voxel_max: Tuple[int, int, int]
    # world bounds are axis-aligned (min/max over transformed corners)
    world_min: Tuple[float, float, float]
    world_max: Tuple[float, float, float]
    affine_source: str


def _pick_affine(img) -> Tuple[np.ndarray, str]:
    """
    Choose an affine that best represents voxel->world coordinates while being robust
    to odd header states.

    Preference:
      1) sform if code > 0 and finite
      2) qform if code > 0 and finite
      3) img.affine (nibabel's best guess)
    """
    # nibabel API: get_sform(coded=True) -> (affine, code)
    sform, scode = img.get_sform(coded=True)
    if int(scode) > 0 and np.isfinite(sform).all():
        return np.asarray(sform, dtype=np.float64), f"sform(code={int(scode)})"

    qform, qcode = img.get_qform(coded=True)
    if int(qcode) > 0 and np.isfinite(qform).all():
        return np.asarray(qform, dtype=np.float64), f"qform(code={int(qcode)})"

    aff = np.asarray(img.affine, dtype=np.float64)
    if not np.isfinite(aff).all():
        raise ValueError("No finite affine found in NIfTI header (sform/qform/img.affine).")

    return aff, "img.affine"


def _apply_affine(aff: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 affine to Nx3 points (ijk) in voxel coordinates.
    """
    ijk = np.asarray(ijk, dtype=np.float64)
    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected Nx3 array of points, got shape={ijk.shape}")

    ones = np.ones((ijk.shape[0], 1), dtype=np.float64)
    pts = np.hstack([ijk, ones])  # Nx4
    w = (aff @ pts.T).T[:, :3]    # Nx3
    return w


def bbox_from_nii(
        path: str | Path,
        *,
        threshold: float = 0.0,
        use_absolute: bool = False,
        allow_empty: bool = False,
) -> BBox:
    """
    Compute bounding box of voxels where:
      data > threshold  (or abs(data) > threshold if use_absolute=True)

    Returns voxel-space bbox (inclusive min/max) and world-space bbox (min/max).

    World bbox is computed from voxel *edges* using (min) and (max+1) corners, so it
    corresponds to a physical bounding box in the output coordinate system.
    """
    nib = _require_nibabel()

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))

    img = nib.load(str(p))
    aff, aff_src = _pick_affine(img)

    data = img.get_fdata(dtype=np.float32)  # applies slope/intercept correctly for values
    if use_absolute:
        mask = np.abs(data) > float(threshold)
    else:
        mask = data > float(threshold)

    if not np.any(mask):
        if allow_empty:
            # Return a "degenerate" bbox; voxel min/max set to (-1,-1,-1)
            neg = (-1, -1, -1)
            nan = (float("nan"), float("nan"), float("nan"))
            return BBox(voxel_min=neg, voxel_max=neg, world_min=nan, world_max=nan, affine_source=aff_src)
        raise ValueError(
            f"No voxels found above threshold={threshold} (use_absolute={use_absolute}). "
            "Try lowering threshold or pass allow_empty=True."
        )

    # Find voxel-space bbox (inclusive indices)
    coords = np.argwhere(mask)  # Nx3 (i,j,k)
    vmin = coords.min(axis=0).astype(int)
    vmax = coords.max(axis=0).astype(int)

    # Convert voxel bbox to world bbox using voxel edges:
    # min corner uses vmin, max corner uses vmax+1 so it encloses the full set of voxels.
    lo = vmin.astype(np.float64)
    hi = (vmax + 1).astype(np.float64)

    # 8 corners of the axis-aligned voxel-edge box
    corners_ijk = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )

    corners_world = _apply_affine(aff, corners_ijk)
    wmin = corners_world.min(axis=0)
    wmax = corners_world.max(axis=0)

    return BBox(
        voxel_min=(int(vmin[0]), int(vmin[1]), int(vmin[2])),
        voxel_max=(int(vmax[0]), int(vmax[1]), int(vmax[2])),
        world_min=(float(wmin[0]), float(wmin[1]), float(wmin[2])),
        world_max=(float(wmax[0]), float(wmax[1]), float(wmax[2])),
        affine_source=aff_src,
    )


def bbox_to_dict(b: BBox) -> Dict[str, Any]:
    return {
        "voxel": {"min": list(b.voxel_min), "max": list(b.voxel_max), "inclusive": True},
        "world": {"min": list(b.world_min), "max": list(b.world_max), "units": "unknown (typically mm)"},
        "affine_source": b.affine_source,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compute voxel and world bounding boxes from a NIfTI file.")
    ap.add_argument("nii", help="Path to .nii or .nii.gz")
    ap.add_argument("--threshold", type=float, default=0.0, help="Foreground threshold (default: 0.0)")
    ap.add_argument(
        "--abs",
        dest="use_absolute",
        action="store_true",
        help="Use abs(data) > threshold instead of data > threshold",
    )
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no foreground voxels exist, output NaNs instead of erroring.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text.",
    )
    args = ap.parse_args(argv)

    b = bbox_from_nii(
        args.nii,
        threshold=args.threshold,
        use_absolute=args.use_absolute,
        allow_empty=args.allow_empty,
    )

    if args.json:
        print(json.dumps(bbox_to_dict(b), indent=2))
    else:
        print(f"File: {Path(args.nii).expanduser().resolve()}")
        print(f"Affine used: {b.affine_source}")
        print(f"Voxel bbox (inclusive): min={b.voxel_min} max={b.voxel_max}")
        print(f"World bbox:             min={b.world_min} max={b.world_max}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
