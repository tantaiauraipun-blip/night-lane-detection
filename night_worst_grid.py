"""
night_worst_grid.py
Builds one grid image of the worst-N baseline failures (from
~/culane_night_worst_baseline.txt, produced by night_eval_clrnet.py) so you
can eyeball all of them at once and tally failure categories quickly
(occlusion by parked/passing vehicles, faint/absent lane paint,
intersection/cross-road framing, genuine low-light, etc.) instead of
opening 20 separate files.

Run in env base (no PyTorch needed):

    python night_worst_grid.py \
        --worst_list ~/culane_night_worst_baseline.txt \
        --image_dir ~/culane_night_test \
        --output ~/culane_night_worst_grid.jpg \
        --cols 4
"""

import argparse
import os
import re
import cv2
import numpy as np

LABEL_H = 24
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_worst_list(path):
    fnames = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"(\S+)", line)
            if m:
                fnames.append(m.group(1))
    return fnames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst_list", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell_w", type=int, default=380)
    args = ap.parse_args()

    worst_list = os.path.expanduser(args.worst_list)
    image_dir = os.path.expanduser(args.image_dir)
    output = os.path.expanduser(args.output)

    fnames = parse_worst_list(worst_list)
    print(f"{len(fnames)} filenames read from {worst_list}")

    cells = []
    cell_h = None
    for fname in fnames:
        path = os.path.join(image_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] could not read {fname}")
            continue
        h, w = img.shape[:2]
        scale = args.cell_w / w
        img = cv2.resize(img, (args.cell_w, int(h * scale)))
        if cell_h is None:
            cell_h = img.shape[0]
        bar = np.full((LABEL_H, args.cell_w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, fname, (6, 17), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cell = np.vstack([bar, img])
        cells.append(cell)

    if not cells:
        print("No images found -- check paths.")
        return

    cols = args.cols
    rows = (len(cells) + cols - 1) // cols
    cell_full_h = cells[0].shape[0]
    grid = np.full((rows * cell_full_h, cols * args.cell_w, 3), 20, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        grid[r * cell_full_h: r * cell_full_h + cell.shape[0],
             c * args.cell_w: c * args.cell_w + args.cell_w] = cell

    cv2.imwrite(output, grid)
    print(f"Wrote grid ({rows}x{cols}, {len(cells)} images) to {output}")


if __name__ == "__main__":
    main()
