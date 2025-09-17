# sdt_to_mask.py
# Convert nnUNet-SDT .npz outputs into binary masks (.tif)
# Uses Otsu-on-logit (for prob-like SDTs in [0,1]) or zero-crossing for raw SDTs.
# Saves _mask.tif and _sdt.tif to DIFFERENT directories.

import os, glob, argparse
import numpy as np
from tifffile import imwrite

# Optional deps for blur + connected components; script still works without them
try:
    from scipy.ndimage import gaussian_filter, label
except Exception:
    gaussian_filter = None
    label = None


# ---------- Utils ----------
def squeeze_first_channel(arr: np.ndarray) -> np.ndarray:
    """Expect arr shape [C, ...]; keep first channel and squeeze nothing else."""
    arr = np.asarray(arr)
    if arr.ndim < 3:
        raise ValueError(f"Unexpected array shape {arr.shape}; expected at least 3D [C,...].")
    if arr.shape[0] != 1:
        arr = arr[:1, ...]
    return arr[0].astype(np.float32)

def gblur(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma and sigma > 0 and gaussian_filter is not None:
        return gaussian_filter(a, sigma)
    return a

def otsu_threshold(x: np.ndarray) -> float:
    """Robust Otsu threshold on 1D array."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    # clip tails to reduce seam/outlier influence
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    x = x[(x >= lo) & (x <= hi)]
    hist, edges = np.histogram(x, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m  = np.cumsum(hist * centers)
    mT = m[-1]
    mu0 = np.divide(m, w0, out=np.zeros_like(m), where=w0 > 0)
    mu1 = np.divide(mT - m, w1, out=np.zeros_like(m), where=w1 > 0)
    sigma_b2 = w0[:-1] * w1[:-1] * (mu0[:-1] - mu1[:-1])**2
    idx = int(np.argmax(sigma_b2))
    return float(centers[idx])

def otsu_logit_threshold(p: np.ndarray, blur_sigma: float = 1.0, eps: float = 1e-6) -> float:
    """Compute Otsu threshold in logit space for probability-like SDTs."""
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    p = gblur(p, blur_sigma)
    z = np.log(p / (1.0 - p))            # logit
    tz = otsu_threshold(z)
    t  = 1.0 / (1.0 + np.exp(-tz))       # back to probability
    return float(t)

def remove_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0 or label is None:
        return mask
    lab, n = label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel())
    keep = np.where(counts >= min_size)[0]
    keep = keep[keep != 0]
    return np.isin(lab, keep)

def keep_largest(mask: np.ndarray) -> np.ndarray:
    if label is None:
        return mask
    lab, n = label(mask)
    if n == 0:
        return mask
    counts = np.bincount(lab.ravel()); counts[0] = 0
    return lab == int(np.argmax(counts))


# ---------- Core conversion ----------
def sdt_to_mask_auto(
    sdt: np.ndarray,
    eps: float = 0.0,
    blur_sigma: float = 1.0,
    target_fg: float = 0.06,
    min_size: int = 0,
    largest: bool = False,
):
    """
    If data look probability-like (range within [0,1]), use Otsu-on-logit.
    Otherwise, fall back to zero-crossing for raw SDT (foreground = sdt < -eps).
    Returns (mask_uint8, method_str, threshold/eps_used).
    """
    sdt = sdt.astype(np.float32)
    mn, mx = float(np.nanmin(sdt)), float(np.nanmax(sdt))

    if 0.0 <= mn and mx <= 1.0:
        # Prob-like SDT (e.g., sigmoid(SDT))
        t = otsu_logit_threshold(sdt, blur_sigma=blur_sigma)
        m_gt = sdt > t
        m_lt = sdt < t
        # Pick polarity closer to expected foreground fraction
        r_gt, r_lt = m_gt.mean(), m_lt.mean()
        mask = m_gt if abs(r_gt - target_fg) <= abs(r_lt - target_fg) else m_lt
        method = f"logit-otsu(t={t:.5f})"
        thr_used = t
    else:
        # Raw SDT with negatives
        mask = sdt < -float(eps)
        method = f"zero-crossing(eps={eps})"
        thr_used = -float(eps)

    # Cleanup
    if min_size > 0:
        mask = remove_small(mask, min_size)
    if largest:
        mask = keep_largest(mask)

    return mask.astype(np.uint8), method, thr_used


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Convert nnUNet-SDT .npz predictions to binary masks.")
    ap.add_argument("--pred_dir", required=True, help="Folder with nnUNetv2_predict outputs (.npz)")
    ap.add_argument("--out_mask_dir", required=True, help="Output folder for binary masks (.tif)")
    ap.add_argument("--out_sdt_dir", help="Optional folder to also save SDT as float32 TIFF")
    ap.add_argument("--eps", type=float, default=0.0, help="Negative margin for raw SDT (used if data not in [0,1])")
    ap.add_argument("--blur_sigma", type=float, default=1.0, help="Gaussian blur sigma before Otsu (prob-like case)")
    ap.add_argument("--target_fg", type=float, default=0.06, help="Expected foreground fraction to choose polarity")
    ap.add_argument("--min_size", type=int, default=0, help="Remove CCs smaller than this many pixels/voxels")
    ap.add_argument("--keep_largest", action="store_true", help="Keep only the largest connected component")
    ap.add_argument("--limit", type=int, default=0, help="Convert only the first N cases (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_mask_dir, exist_ok=True)
    if args.out_sdt_dir:
        os.makedirs(args.out_sdt_dir, exist_ok=True)

    npzs = sorted(glob.glob(os.path.join(args.pred_dir, "*.npz")))
    if not npzs:
        raise RuntimeError(f"No .npz files found in {args.pred_dir}")

    done = 0
    for npz_path in npzs:
        case = os.path.splitext(os.path.basename(npz_path))[0]
        data = np.load(npz_path)

        if "probabilities" in data:
            arr = data["probabilities"]    # shape [C, ...]
        elif "pred" in data:
            arr = data["pred"]
        else:
            raise KeyError(f"{npz_path} missing 'probabilities' or 'pred' key")

        sdt = squeeze_first_channel(arr)

        mask, method, thr = sdt_to_mask_auto(
            sdt,
            eps=args.eps,
            blur_sigma=args.blur_sigma,
            target_fg=args.target_fg,
            min_size=args.min_size,
            largest=args.keep_largest,
        )

        mask_path = os.path.join(args.out_mask_dir, case + "_mask.tif")
        imwrite(mask_path, mask, dtype=np.uint8)

        extra = ""
        if args.out_sdt_dir:
            sdt_path = os.path.join(args.out_sdt_dir, case + "_sdt.tif")
            imwrite(sdt_path, sdt.astype(np.float32))
            extra = f" and {sdt_path}"

        print(f"[OK] {case}: saved {mask_path}{extra} | {method} | fg={mask.mean():.4f}")

        done += 1
        if args.limit and done >= args.limit:
            break


if __name__ == "__main__":
    main()