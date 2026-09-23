"""
night_visual_compare.py
Quick, no-GPU visual spot-check: stacks original / selective / aware
versions of a handful of sample images (original | selective | aware,
left to right, with labels) so you can eyeball what CLAHE is actually
doing to real night images before deciding whether to retune or pivot.

Run in the SAME env you used for night_enhance.py (base env is fine --
no PyTorch needed):

    python night_visual_compare.py \
        --baseline_dir ~/culane_night_test \
        --selective_dir ~/culane_night_selective \
        --aware_dir ~/culane_night_aware \
        --output_dir ~/culane_night_visual_check \
        --n 10

Open the images in ~/culane_night_visual_check afterwards (e.g. with
your file explorer via \\wsl$, or `explorer.exe .` from WSL, or just
scp/copy them out) and look specifically at the road surface and lane
markings: is CLAHE over-brightening the road so the (now lighter) lane
paint blends into it? Any new glare/noise around headlights bleeding
onto the road?
"""

import argparse
import os
import cv2
import numpy as np

LABEL_H = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label_bar(width, text):
    bar = np.full((LABEL_H, width, 3), 40, dtype=np.uint8)
    cv2.putText(bar, text, (8, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def build_row(fname, baseline_dir, selective_dir, aware_dir, out_w=520):
    imgs = []
    for d, label in [(baseline_dir, "original"), (selective_dir, "selective"), (aware_dir, "aware")]:
        path = os.path.join(d, fname)
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = out_w / w
        img = cv2.resize(img, (out_w, int(h * scale)))
        img = np.vstack([label_bar(out_w, label), img])
        imgs.append(img)
    return np.hstack(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True)
    ap.add_argument("--selective_dir", required=True)
    ap.add_argument("--aware_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    baseline_dir = os.path.expanduser(args.baseline_dir)
    selective_dir = os.path.expanduser(args.selective_dir)
    aware_dir = os.path.expanduser(args.aware_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fnames = sorted(f for f in os.listdir(baseline_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    # evenly spaced sample across the set rather than just the first N
    if len(fnames) > args.n:
        idxs = np.linspace(0, len(fnames) - 1, args.n).astype(int)
        fnames = [fnames[i] for i in idxs]

    written = 0
    for fname in fnames:
        row = build_row(fname, baseline_dir, selective_dir, aware_dir)
        if row is None:
            print(f"  [skip] missing variant for {fname}")
            continue
        out_path = os.path.join(output_dir, f"compare_{fname}")
        cv2.imwrite(out_path, row)
        written += 1

    print(f"Wrote {written} comparison images to {output_dir}")


if __name__ == "__main__":
    main()
