"""
night_lane_only_enhance.py
Reimplementation of the teammate's "Lane-Aware Selective Enhancement" idea,
rebuilt to follow the EXACT 4-step pipeline the teammate described in chat:

    Step 1: enhance the estimated area where the lane WOULD be
            (a rough, feathered brighten inside a coarse road ROI -- this is
             purely to help the classical detector see something on a
             near-black frame; it is NOT part of the final output image)
    Step 2: run the previous model (Canny + HSV + HoughLinesP classical
            detector) on that Step-1-enhanced image to detect lane
            trajectories
    Step 3: enhance roughly the area where lanes were actually detected
            (a second, tighter/stronger brighten pass, blended only inside a
             narrow band around the detected trajectories, applied to the
             ORIGINAL image -- not the Step-1 image)
    Step 4: mark the lanes (debug visualization: red polylines + green
            sample points, for visual QA)

This matches how the teammate is building their pipeline -- TWO separate
enhancement passes (a coarse one before detection, a precise one after)
rather than a single pass. Earlier revisions of this script only did a
single full-frame pre-brighten before detection; this version restricts
Step 1 to the estimated ROI (as described) and keeps Step 3 as its own,
clearly separated pass so it's easy to see how it maps onto the 4 steps.

No source code was shared for the teammate's actual implementation -- exact
thresholds/formulas are still our own best-effort reconstruction from the
slides/script/chat description. This script exists so we can run the result
through our already-verified night_eval_clrnet.py pipeline and get a real
CLRNet F1 number for this approach vs. baseline.

Usage:
    python night_lane_only_enhance.py \
        --input_dir ~/culane_night_test \
        --output_dir ~/culane_night_lane_only \
        --debug_dir ~/culane_night_lane_only_debug \
        --mask_width 22
"""

import cv2
import numpy as np
import os
import argparse
import glob


# ---------------- shared helpers ----------------

def gamma_brighten(img_bgr, gamma=0.45):
    """LUT-based gamma brighten (whole-image transform; caller decides where it's applied).
    gamma in (0,1) brightens (smaller gamma = stronger brighten): v_out = v_in ** gamma.

    BUG FIX: this previously computed v_out = v_in ** (1/gamma). With gamma=0.45,
    1/gamma = 2.22, and raising a normalized value (0..1) to a power > 1 makes it
    SMALLER, not larger -- so this was actually DARKENING every image it touched
    instead of brightening it (verified: input 185 -> output ~124, input 8 -> 0).
    This silently undermined both Step 1 (pre-brighten for detection) and, via the
    identical bug in brighten_full() below, the Step 3 final enhancement -- CLAHE's
    local contrast boost was fighting against this erroneous darkening pass. Every
    lane_only F1 number produced before this fix (v1 and v2) was run with this bug
    active and should be treated as unreliable."""
    table = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(img_bgr, table)


def feather_mask(mask, blur_ksize=15):
    """0/255 mask -> soft float mask in [0,1], blurred so blended edges have no hard seam."""
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (blur_ksize, blur_ksize), 0)
    return mask_f


def get_road_roi_mask(img_shape, top_ratio=0.65, bottom_width_ratio=0.95, top_width_ratio=0.5):
    """Coarse trapezoid = the 'estimated area where lane would be' for Step 1,
    and also used in Step 2 to suppress obviously-non-road Hough candidates
    (sky, building facades).

    top_ratio raised from 0.55 -> 0.65 (fix #3): the previous value let the
    ROI reach up into the sky/tree line / distant overpass in wide-FOV night
    shots, which is exactly where the guardrail- and tree-edge mistracking
    cases were coming from. Narrowing the vertical reach trades off some
    look-ahead distance for fewer non-road false positives."""
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
    return mask, top_y


def brighten_full(img_bgr, clip_limit=4.0, gamma=0.6):
    """CLAHE + gamma brighten -- the 'strong' enhancement source used in Step 3
    (the targeted, full-strength enhancement applied inside the lane mask).

    BUG FIX: same inverted-exponent bug as gamma_brighten() above -- the gamma
    LUT here was computing v_out = v_in ** (1/gamma), which DARKENS (verified:
    input 150 -> output 105, input 200 -> output 170). This was fighting
    against CLAHE's local contrast boost immediately before it, partially or
    fully undoing it depending on the image. Now uses v_out = v_in ** gamma,
    which actually brightens."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    table = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
    out = cv2.LUT(out, table)
    return out


# ---------------- STEP 1: enhance the estimated area where lane would be ----------------

def enhance_estimated_area(img_bgr, roi_mask, gamma=0.45, feather_ksize=31):
    """Rough, feathered gamma-brighten restricted to the coarse ROI (the
    'estimated area'). Only used to help Step 2's detector see edges/colors
    on a very dark frame -- this image is discarded after detection and is
    NOT part of the final output."""
    bright = gamma_brighten(img_bgr, gamma=gamma)
    mask_soft = feather_mask(roi_mask, blur_ksize=feather_ksize)[..., None]
    out = mask_soft * bright.astype(np.float32) + (1 - mask_soft) * img_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------- STEP 2: run the previous model (Canny+HSV+Hough) to detect lanes ----------------

def lane_color_mask(img_bgr):
    """HSV threshold for lane-paint-like colors (white and yellow), tolerant of
    orange sodium-vapor street lighting."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))
    yellow = cv2.inRange(hsv, (15, 60, 120), (40, 255, 255))
    return cv2.bitwise_or(white, yellow)


def detect_candidate_pixels(stage1_img_bgr):
    """Canny edges + HSV color mask on the Step-1-enhanced image (this is the
    'previous model' / classical detector's input, not the raw dark frame).

    Fix #2: previously a raw Canny edge alone was enough to count as a
    candidate (edges OR (edges AND color)), so any strong edge with no
    lane-paint color at all -- guardrails, tree/sky boundaries, building
    facades -- could still feed into Hough. Now a candidate pixel must have
    BOTH a Canny edge AND nearby lane-paint color (white/yellow); the color
    mask is mildly dilated first so a few pixels of misalignment between the
    edge and the color region don't kill a real lane match."""
    gray = cv2.cvtColor(stage1_img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    color = lane_color_mask(stage1_img_bgr)
    color_dilated = cv2.dilate(color, np.ones((7, 7), np.uint8))
    candidates = cv2.bitwise_and(edges, color_dilated)
    return candidates


def fit_lane_side(segments, img_h, top_y):
    """segments: list of (x1,y1,x2,y2) already filtered to one side.
    Fit x = a*y + b via least squares over segment endpoints, return sampled
    (x, y) points. Returns None if too few points or too little support.

    Fix #1 + #4: previously this sampled from top_y all the way to img_h-1
    REGARDLESS of where the matched segments actually were -- so a handful
    of edge pixels clustered near the top of the ROI (e.g. a tree/sky
    boundary or a road sign, as seen in a real debug image) got a straight
    line fit through them and then that line was extrapolated across the
    ENTIRE ROI height, drawing a long, confident-looking trajectory that
    only had real evidence for a few pixels of it. Now:
      - sampling is clipped to the observed y-range of the matched segments
        (no extrapolation beyond where there is actual evidence), and
      - a fit is discarded if its segments only span a small fraction of the
        ROI height (< 40%), since that's too little support to trust as a
        real lane trajectory rather than a stray non-lane edge cluster."""
    if len(segments) < 2:
        return None
    ys, xs = [], []
    for x1, y1, x2, y2 in segments:
        ys.extend([y1, y2])
        xs.extend([x1, x2])
    ys = np.array(ys, dtype=np.float64)
    xs = np.array(xs, dtype=np.float64)
    if len(set(ys.tolist())) < 2:
        return None
    a, b = np.polyfit(ys, xs, 1)

    y_min = max(top_y, float(np.min(ys)))
    y_max = min(img_h - 1, float(np.max(ys)))
    roi_height = img_h - top_y
    min_span = 0.4 * roi_height
    if (y_max - y_min) < min_span:
        return None

    sample_ys = np.arange(y_min, y_max, 15)
    sample_xs = a * sample_ys + b
    return [(float(x), float(y)) for x, y in zip(sample_xs, sample_ys)]


def detect_lane_trajectories(img_bgr):
    """Full Step 1 + Step 2: enhance the estimated area, then run the
    classical Canny+HSV+Hough detector on it, and fit left/right trajectories."""
    roi_mask, top_y = get_road_roi_mask(img_bgr.shape)

    # STEP 1
    stage1_img = enhance_estimated_area(img_bgr, roi_mask)

    # STEP 2
    candidates = detect_candidate_pixels(stage1_img)
    candidates = cv2.bitwise_and(candidates, roi_mask)

    h, w = img_bgr.shape[:2]
    lines = cv2.HoughLinesP(candidates, 1, np.pi / 180, threshold=25,
                             minLineLength=25, maxLineGap=40)
    if lines is None:
        return []

    left_segs, right_segs = [], []
    cx = w / 2.0
    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = l
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.35:  # too horizontal -> not a lane line
            continue
        seg_cx = (x1 + x2) / 2.0
        if seg_cx < cx:
            left_segs.append((x1, y1, x2, y2))
        else:
            right_segs.append((x1, y1, x2, y2))

    trajectories = []
    for segs in (left_segs, right_segs):
        pts = fit_lane_side(segs, h, top_y)
        if pts is not None:
            trajectories.append(pts)
    return trajectories


# ---------------- STEP 3: enhance roughly the area where lanes were detected ----------------

def trajectories_to_mask(img_shape, trajectories, width=22):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in trajectories:
        pts_int = np.array([[int(round(x)), int(round(y))] for x, y in pts
                             if 0 <= x < w and 0 <= y < h], dtype=np.int32)
        if len(pts_int) >= 2:
            cv2.polylines(mask, [pts_int], isClosed=False, color=255, thickness=width)
    return mask


def lane_only_enhance(img_bgr, mask_width=22):
    # STEP 1 + STEP 2 (detection)
    trajectories = detect_lane_trajectories(img_bgr)

    # STEP 3: build a narrow band around the DETECTED trajectories and
    # brighten only there, on the ORIGINAL (not Step-1) image
    lane_mask = trajectories_to_mask(img_bgr.shape, trajectories, width=mask_width)
    mask_soft = feather_mask(lane_mask)[..., None]

    strong = brighten_full(img_bgr)
    out = mask_soft * strong.astype(np.float32) + (1 - mask_soft) * img_bgr.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out, trajectories, lane_mask


# ---------------- STEP 4: mark the lanes (debug visualization) ----------------

def draw_debug(img_bgr, trajectories):
    vis = img_bgr.copy()
    for pts in trajectories:
        pts_int = [(int(round(x)), int(round(y))) for x, y in pts]
        for i in range(len(pts_int) - 1):
            cv2.line(vis, pts_int[i], pts_int[i + 1], (0, 0, 255), 2)
        for x, y in pts_int:
            cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
    return vis


# ---------------- image-level metrics (best-effort reconstruction) ----------------

def compute_metrics(orig_bgr, enhanced_bgr, lane_mask, roi_mask):
    gray_o = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_e = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

    lane_bin = lane_mask > 0
    road_bin = (roi_mask > 0) & (~cv2.dilate(lane_mask, np.ones((25, 25), np.uint8)) > 0)
    bg_bin = roi_mask == 0

    def safe_mean(arr, m):
        return float(arr[m].mean()) if m.any() else float("nan")

    def safe_std(arr, m):
        return float(arr[m].std()) if m.any() else float("nan")

    lane_lum = safe_mean(gray_e, lane_bin)
    road_lum = safe_mean(gray_e, road_bin)
    contrast = (lane_lum - road_lum) / (lane_lum + road_lum + 1e-6) if not np.isnan(lane_lum) else float("nan")
    bg_noise = safe_std(gray_e, bg_bin)
    lap = cv2.Laplacian(gray_e, cv2.CV_64F)
    sharpness = float(lap[lane_bin].var()) if lane_bin.any() else float("nan")

    return {
        "lane_luminance": lane_lum,
        "road_luminance": road_lum,
        "contrast": contrast,
        "background_noise": bg_noise,
        "lane_edge_sharpness": sharpness,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--debug_dir", default=None)
    ap.add_argument("--mask_width", type=int, default=22)
    ap.add_argument("--metrics_csv", default=None)
    args = ap.parse_args()

    input_dir = os.path.expanduser(args.input_dir)
    output_dir = os.path.expanduser(args.output_dir)
    debug_dir = os.path.expanduser(args.debug_dir) if args.debug_dir else None
    os.makedirs(output_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))
    print(f"Found {len(files)} images in {input_dir}")

    metrics_rows = []
    n_zero_lane = 0
    for path in files:
        fname = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] could not read {fname}")
            continue

        # Step 1 + 2 + 3 combined:
        out, trajectories, lane_mask = lane_only_enhance(img, mask_width=args.mask_width)
        cv2.imwrite(os.path.join(output_dir, fname), out)

        if len(trajectories) == 0:
            n_zero_lane += 1

        if debug_dir:
            # Step 4: mark the lanes
            vis = draw_debug(img, trajectories)
            cv2.imwrite(os.path.join(debug_dir, fname), vis)

        if args.metrics_csv:
            roi_mask, _ = get_road_roi_mask(img.shape)
            m = compute_metrics(img, out, lane_mask, roi_mask)
            m["fname"] = fname
            m["n_trajectories"] = len(trajectories)
            metrics_rows.append(m)

    print(f"Done. {len(files)} images processed, {n_zero_lane} had ZERO lane trajectories detected "
          f"({100.0 * n_zero_lane / max(1, len(files)):.1f}%).")

    if args.metrics_csv and metrics_rows:
        import csv
        keys = ["fname", "n_trajectories", "lane_luminance", "road_luminance",
                "contrast", "background_noise", "lane_edge_sharpness"]
        with open(os.path.expanduser(args.metrics_csv), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in metrics_rows:
                w.writerow({k: row.get(k) for k in keys})
        print(f"Per-image metrics written to {args.metrics_csv}")

        import statistics
        for key in ["lane_luminance", "road_luminance", "contrast", "background_noise", "lane_edge_sharpness"]:
            vals = [r[key] for r in metrics_rows if not (isinstance(r[key], float) and np.isnan(r[key]))]
            if vals:
                print(f"  avg {key}: {statistics.mean(vals):.3f}")


if __name__ == "__main__":
    main()
