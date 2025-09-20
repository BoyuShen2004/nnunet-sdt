#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import tifffile as tiff
import cc3d

def keep_largest_component(binary_arr: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component in 2D/3D."""
    connectivity = 8 if binary_arr.ndim == 2 else 26
    labels = cc3d.connected_components(binary_arr.astype(np.uint8), connectivity=connectivity)
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return np.zeros_like(binary_arr, dtype=binary_arr.dtype)
    counts[0] = 0  # ignore background
    keep_label = counts.argmax()
    return labels == keep_label

def process_one(path_in: str, path_out: str) -> None:
    arr = tiff.imread(path_in)

    # If last dim is RGB/RGBA, take the first channel
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., 0]

    fg_val = np.asarray(arr).max()
    bin_arr = arr > 0

    cleaned = keep_largest_component(bin_arr)
    out = cleaned.astype(arr.dtype) * (fg_val if fg_val > 1 else 1)

    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    tiff.imwrite(path_out, out)
    print(f"Saved: {path_out}")

def main():
    p = argparse.ArgumentParser(
        description="Remove small unconnected components by keeping only the largest component."
    )
    p.add_argument("--in_dir",  required=True, help="Input directory OR a single .tif path")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--file",    default=None,
                   help="Optional: process only this filename inside --in_dir "
                        "(ignored if --in_dir is a file path).")
    args = p.parse_args()

    in_path  = args.in_dir
    out_dir  = args.out_dir
    only_one = args.file

    # Case A: --in_dir is a single file
    if os.path.isfile(in_path) and in_path.lower().endswith((".tif", ".tiff")):
        out_name = os.path.basename(in_path)
        dst = os.path.join(out_dir, out_name)
        process_one(in_path, dst)
        return

    # Case B: --in_dir is a directory
    if not os.path.isdir(in_path):
        raise NotADirectoryError(f"--in_dir is neither a .tif nor a directory: {in_path}")

    if only_one is not None:
        src = os.path.join(in_path, only_one)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"File not found in --in_dir: {only_one}")
        dst = os.path.join(out_dir, only_one)
        process_one(src, dst)
        return

    # Process ALL .tif/.tiff in directory
    names = [f for f in os.listdir(in_path) if f.lower().endswith((".tif", ".tiff"))]
    names.sort()
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        src = os.path.join(in_path, name)
        dst = os.path.join(out_dir, name)
        process_one(src, dst)

if __name__ == "__main__":
    main()