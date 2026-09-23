"""
night_enhance.py
Implements the two preprocessing methods proposed for improving CLRNet's
accuracy on CULane's real Night category, without any synthetic data
generation:

  1. lane_region_selective_enhancement()  -- Approach 2 (recommended first: simpler, faster)
  2. lane_aware_contrast_enhancement()    -- Approach 1 (adaptive, more involved)

Both reuse a classic trapezoidal ROI mask (no extra model needed) and
CLAHE (contrast-limited adaptive histogram equalization) as the actual
enhancement operator, blended back into the original image with a
feathered mask edge so there is no visible seam.

Usage:
    python night_enhance.py --input_dir <folder of night images> --output_dir <out> --method selective
    python night_enhance.py --input_dir <folder of night images> --output_dir <out> --method aware
"""

import cv2
import numpy as np
import os
import argparse


def get_roi_mask(img_shape, top_ratio=0.55, bottom_width_ratio=0.95, top_width_ratio=0.35):
    """
    Classic trapezoidal ROI mask used in Hough-transform-era lane detection.
    Assumes a forward-facing dashcam view: the road occupies the lower
    portion of the image, narrowing toward the vanishing point near the
    horizon. Tune top_ratio/width ratios per dataset if needed.
    """
    h, w = img_shape[:2]
    top_y = int(h * top_ratio)
    bottom_y = h

    top_left = (int(w * (1 - top_width_ratio) / 2), top_y)
    top_right = (int(w * (1 + top_width_ratio) / 2), top_y)
    bottom_left = (int(w * (1 - bottom_width_ratio) / 2), bottom_y)
    bottom_right = (int(w * (1 + bottom_width_ratio) / 2), bottom_y)

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def feather_mask(mask, blur_ksize=51):
    """Soft-blur mask edges so the enhanced ROI blends into the rest of the image without a visible seam."""
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (blur_ksize, blur_ksize), 0)
    return mask_f


def apply_clahe(img_bgr, clip_limit=3.0, tile_grid_size=(8, 8)):
    """CLAHE applied on the L channel in LAB color space (preserves color, boosts local contrast)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def lane_region_selective_enhancement(img_bgr, clip_limit=3.0):
    """
    Approach 2: enhance ONLY the road/lane ROI, leave the rest of the image
    (sky, buildings, oncoming headlights, etc.) untouched, to avoid adding
    glare/noise outside the region that matters for lane detection.
    """
    mask = get_roi_mask(img_bgr.shape)
    mask_soft = feather_mask(mask)[..., None]  # (H, W, 1) for broadcasting

    enhanced_full = apply_clahe(img_bgr, clip_limit=clip_limit)

    out = (mask_soft * enhanced_full.astype(np.float32) +
           (1 - mask_soft) * img_bgr.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def measure_lane_road_contrast(img_bgr, mask):
    """
    Approach 1, step 2: crude proxy for lane-vs-road contrast within the ROI.
    Uses the standard deviation of pixel intensity inside the ROI as a proxy
    for how well lane markings currently stand out from the road surface --
    low std generally means a flat, low-contrast road/lane boundary.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[mask > 0]
    if roi_pixels.size == 0:
        return 0.0
    return float(roi_pixels.std())


def lane_aware_contrast_enhancement(img_bgr, contrast_low=15.0, contrast_high=40.0,
                                     clip_low=5.0, clip_high=1.5):
    """
    Approach 1: instead of a fixed CLAHE strength, scale clipLimit inversely
    with the measured lane-road contrast -- darker/flatter ROIs (low std)
    get a stronger boost, already-decent ROIs get a gentler one.
    """
    mask = get_roi_mask(img_bgr.shape)
    contrast_score = measure_lane_road_contrast(img_bgr, mask)

    # Linear interpolation between clip_low (weak contrast -> strong enhancement)
    # and clip_high (already good contrast -> light enhancement).
    t = float(np.clip((contrast_score - contrast_low) / (contrast_high - contrast_low), 0, 1))
    adaptive_clip = clip_low + t * (clip_high - clip_low)

    mask_soft = feather_mask(mask)[..., None]
    enhanced_full = apply_clahe(img_bgr, clip_limit=adaptive_clip)

    out = (mask_soft * enhanced_full.astype(np.float32) +
           (1 - mask_soft) * img_bgr.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8), contrast_score, adaptive_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder of CULane Night-category images")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["selective", "aware"], default="selective")
    parser.add_argument("--clip_limit", type=float, default=3.0, help="CLAHE clipLimit for 'selective' method")
    parser.add_argument("--clip_low", type=float, default=5.0,
                         help="'aware' method: clipLimit used for low-contrast ROIs (stronger enhancement)")
    parser.add_argument("--clip_high", type=float, default=1.5,
                         help="'aware' method: clipLimit used for already-decent-contrast ROIs (gentler)")
    parser.add_argument("--contrast_low", type=float, default=15.0, help="'aware' method: contrast_low threshold")
    parser.add_argument("--contrast_high", type=float, default=40.0, help="'aware' method: contrast_high threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png")
    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(exts)])
    print(f"Found {len(files)} images in {args.input_dir}")

    log_rows = []
    for fname in files:
        img_path = os.path.join(args.input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] could not read {fname}")
            continue

        if args.method == "selective":
            out = lane_region_selective_enhancement(img, clip_limit=args.clip_limit)
            log_rows.append((fname, None, args.clip_limit))
        else:
            out, contrast_score, adaptive_clip = lane_aware_contrast_enhancement(
                img,
                contrast_low=args.contrast_low, contrast_high=args.contrast_high,
                clip_low=args.clip_low, clip_high=args.clip_high,
            )
            log_rows.append((fname, contrast_score, adaptive_clip))

        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, out)

    print(f"Done. Enhanced images written to {args.output_dir}")
    if args.method == "aware":
        print("\nfilename, contrast_score, adaptive_clip_limit")
        for row in log_rows:
            print(f"{row[0]}, {row[1]:.2f}, {row[2]:.2f}")


if __name__ == "__main__":
    main()
