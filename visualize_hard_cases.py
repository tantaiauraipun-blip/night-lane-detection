"""
visualize_hard_cases.py
Qualitative companion to official_culane_eval.py / night_eval_clrnet.py.

Three modes:

1) Single-frame mode: draw GT (yellow) + predicted lanes (cyan) overlaid on
   BOTH the raw and enhanced version of one image, for one checkpoint, side
   by side.

       python ~/gen_test/visualize_hard_cases.py \\
           --config work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/config.py \\
           --ckpt   work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/ckpt/11.pth \\
           --fname  05081020_0305.MP4_00120.jpg \\
           --label  original \\
           --out ~/hardcase_original.jpg

   Run again with --config/--ckpt pointed at the finetuned / finetuned_ext
   run dirs (and change --label) to compare across checkpoints -- keep
   --fname the same each time so the comparison is apples-to-apples.

2) Scan mode (count-based, --scan): sweep the held-out set for a checkpoint
   and report frames where the ENHANCED image gets 0 predicted lanes but the
   RAW image (same scene) gets >0. Cheap, but only catches total dropout --
   a run with 0 hits does NOT mean enhancement isn't hurting; it usually
   means the harm is smaller-magnitude and spread across many images
   (wrong lane shape/position -> official IoU@0.5 mismatch) rather than
   full dropout. Use mode 3 to find those.

       python ~/gen_test/visualize_hard_cases.py --scan \\
           --config ... --ckpt ... --label original

3) Official-metric scan mode (--scan_official): for each paired image, runs
   the actual official CULane metric (culane_metric.py, IoU@0.5, same as
   official_culane_eval.py) on both raw and enhanced, and ranks images by
   how many fewer true positives (matched GT lanes) the enhanced version
   gets vs. the raw version. This finds the frames actually driving the
   aggregate F1 gap, even when both images still predict >0 lanes.

       python ~/gen_test/visualize_hard_cases.py --scan_official \\
           --config work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/config.py \\
           --ckpt   work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/ckpt/11.pth \\
           --label  original

Must be run from ~/CLRNet/CLRNet in the clrnet_torch113 conda env, same as
night_eval_clrnet.py / official_culane_eval.py:

    conda activate clrnet_torch113
    unset PYTORCH_CUDA_ALLOC_CONF
    cd ~/CLRNet/CLRNet
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch

sys.path.insert(0, os.getcwd())  # run from ~/CLRNet/CLRNet so `clrnet` package is importable
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner
import clrnet.utils.culane_metric as culane_metric

DEFAULT_CONFIG_PATH = os.path.expanduser(
    "~/CLRNet/CLRNet/work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/config.py"
)
DEFAULT_CKPT_PATH = os.path.expanduser(
    "~/CLRNet/CLRNet/work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/ckpt/11.pth"
)

RAW_DIR = os.path.expanduser("~/culane_night_heldout_raw")
ENH_DIR = os.path.expanduser("~/culane_night_heldout_enhanced")

# CLRNet preprocessing constants (unchanged from night_eval_clrnet.py / official_culane_eval.py)
ORI_W, ORI_H = 1640, 590
IMG_W, IMG_H = 800, 320
CUT_HEIGHT = 270
IOU_THRESHOLD = 0.5
LANE_WIDTH = 30  # px -- matches the official CULane metric's rasterization width

GT_COLOR = (0, 255, 255)     # yellow, BGR
PRED_COLOR = (255, 255, 0)   # cyan, BGR


def load_runner(config_path, ckpt_path):
    print(f"config = {config_path}")
    print(f"ckpt   = {ckpt_path}")
    assert os.path.exists(config_path), f"missing config: {config_path}"
    assert os.path.exists(ckpt_path), f"missing checkpoint: {ckpt_path}"
    cfg = Config.fromfile(config_path)
    cfg.gpus = 1
    cfg.load_from = ckpt_path
    cfg.resume_from = None
    cfg.finetune_from = None
    cfg.view = False
    cfg.seed = 0
    cfg.work_dirs = "/tmp/clrnet_eval"
    runner = Runner(cfg)
    runner.net.eval()
    return runner


def preprocess_image(img_bgr):
    full = cv2.resize(img_bgr, (ORI_W, ORI_H))
    cropped = full[CUT_HEIGHT:, :, :]
    resized = cv2.resize(cropped, (IMG_W, IMG_H))
    img = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return tensor


def get_predicted_lanes(runner, img_bgr):
    tensor = preprocess_image(img_bgr)
    data = {"img": tensor}
    with torch.no_grad():
        output = runner.net(data)
    lanes = runner.net.module.heads.get_lanes(output)[0]
    pred_lanes = []
    for lane in lanes:
        arr = lane.to_array(runner.cfg)
        pred_lanes.append([(float(x), float(y)) for x, y in arr])
    return pred_lanes


def load_gt_lanes_simple(lines_txt_path):
    """Loose GT loader, fine for drawing overlays."""
    if not os.path.exists(lines_txt_path):
        return []
    with open(lines_txt_path, "r") as f:
        data = [list(map(float, line.split())) for line in f.readlines()]
    lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in data]
    return lanes


def load_gt_lanes_official(lines_txt_path):
    """Mirrors clrnet/datasets/culane.py's __getitem__ GT loading exactly
    (drop negative-coord points, dedupe points, keep only lanes with >2
    points, sort by y) -- same as official_culane_eval.py. Use this for
    anything that feeds culane_metric.culane_metric()."""
    if not os.path.exists(lines_txt_path):
        return []
    with open(lines_txt_path, "r") as f:
        data = [list(map(float, line.split())) for line in f.readlines()]
    lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
              if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
    lanes = [list(set(lane)) for lane in lanes]
    lanes = [lane for lane in lanes if len(lane) > 2]
    lanes = [sorted(lane, key=lambda p: p[1]) for lane in lanes]
    return lanes


def draw_lanes(img_bgr, lanes, color, thickness=3):
    out = img_bgr.copy()
    for lane in lanes:
        pts = np.array([[int(round(x)), int(round(y))] for x, y in lane], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=thickness)
        for x, y in pts:
            cv2.circle(out, (x, y), 3, color, -1)
    return out


def annotate(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (img_bgr.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def process_one(runner, img_dir, fname):
    """Returns (annotated_bgr_image, n_gt, n_pred) for one image."""
    img_path = os.path.join(img_dir, fname)
    stem = os.path.splitext(fname)[0]
    lines_path = os.path.join(img_dir, stem + ".lines.txt")
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"could not read {img_path}"
    full = cv2.resize(img_bgr, (ORI_W, ORI_H))  # draw at the same res GT/pred coords use

    gt_lanes = load_gt_lanes_simple(lines_path)
    pred_lanes = get_predicted_lanes(runner, img_bgr)

    vis = draw_lanes(full, gt_lanes, GT_COLOR, thickness=4)
    vis = draw_lanes(vis, pred_lanes, PRED_COLOR, thickness=2)
    return vis, len(gt_lanes), len(pred_lanes)


def single_frame_mode(args):
    runner = load_runner(args.config, args.ckpt)

    raw_vis, raw_ngt, raw_npred = process_one(runner, RAW_DIR, args.fname)
    enh_vis, enh_ngt, enh_npred = process_one(runner, ENH_DIR, args.fname)

    raw_vis = annotate(raw_vis, f"RAW      [{args.label}]  GT={raw_ngt}  pred={raw_npred}")
    enh_vis = annotate(enh_vis, f"ENHANCED [{args.label}]  GT={enh_ngt}  pred={enh_npred}")

    combo = np.hstack([raw_vis, enh_vis])
    cv2.putText(combo, "yellow = ground truth   cyan = predicted", (10, combo.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = os.path.expanduser(args.out)
    cv2.imwrite(out_path, combo)
    print(f"\nfname={args.fname}  label={args.label}")
    print(f"  raw:      GT={raw_ngt:2d}  pred={raw_npred:2d}")
    print(f"  enhanced: GT={enh_ngt:2d}  pred={enh_npred:2d}")
    print(f"Saved comparison image to {out_path}")


def scan_mode(args):
    """Count-based: enhanced predicts 0 lanes, raw predicts >0."""
    runner = load_runner(args.config, args.ckpt)
    img_paths = sorted(glob.glob(os.path.join(ENH_DIR, "*.jpg")))
    print(f"[{args.label}] scanning {len(img_paths)} enhanced images for "
          f"'enhanced predicts 0 lanes but raw predicts >0' cases...")

    hits = []
    for i, enh_path in enumerate(img_paths):
        fname = os.path.basename(enh_path)
        raw_path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(raw_path):
            continue

        enh_img = cv2.imread(enh_path)
        raw_img = cv2.imread(raw_path)
        if enh_img is None or raw_img is None:
            continue

        enh_pred = get_predicted_lanes(runner, enh_img)
        raw_pred = get_predicted_lanes(runner, raw_img)

        if len(enh_pred) == 0 and len(raw_pred) > 0:
            hits.append((fname, len(raw_pred), len(enh_pred)))
            print(f"  [hit] {fname}  raw_pred={len(raw_pred)}  enh_pred=0")

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(img_paths)} scanned, {len(hits)} hits so far")

    print(f"\n=== [{args.label}] {len(hits)} total 'enhanced=0, raw>0' cases out of {len(img_paths)} ===")
    for fname, r, e in hits[: args.top_n]:
        print(f"  {fname}   raw_pred={r}  enh_pred={e}")
    if len(hits) > args.top_n:
        print(f"  ... and {len(hits) - args.top_n} more")
    if not hits:
        print("\nNo total-dropout cases found -- that's actually informative: it means the "
              "aggregate harm from enhancement is NOT concentrated in catastrophic "
              "all-or-nothing failures. Run --scan_official instead to find the frames "
              "where enhancement causes IoU@0.5 mismatches (wrong shape/position) even "
              "though it still predicts some lanes.")
    else:
        print("\nPick a --fname from this list and re-run in single-frame mode to visualize it.")


def scan_official_mode(args):
    """Official-metric-based: rank images by (raw TP - enhanced TP), i.e. how many
    fewer GT lanes the enhanced image correctly matches at IoU@0.5, using the same
    metric as official_culane_eval.py. Surfaces quality degradation, not just dropout."""
    runner = load_runner(args.config, args.ckpt)
    img_paths = sorted(glob.glob(os.path.join(ENH_DIR, "*.jpg")))
    print(f"[{args.label}] scanning {len(img_paths)} images with the official CULane "
          f"metric (IoU@{IOU_THRESHOLD}), ranking by TP lost when enhanced...")

    rows = []
    for i, enh_path in enumerate(img_paths):
        fname = os.path.basename(enh_path)
        stem = os.path.splitext(fname)[0]
        raw_path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(raw_path):
            continue

        enh_img = cv2.imread(enh_path)
        raw_img = cv2.imread(raw_path)
        if enh_img is None or raw_img is None:
            continue

        # GT is shared between raw/enhanced pairs -- read from either dir's .lines.txt
        gt_lines_path = os.path.join(ENH_DIR, stem + ".lines.txt")
        if not os.path.exists(gt_lines_path):
            gt_lines_path = os.path.join(RAW_DIR, stem + ".lines.txt")
        gt_lanes = load_gt_lanes_official(gt_lines_path)
        if not gt_lanes:
            continue

        raw_pred = get_predicted_lanes(runner, raw_img)
        enh_pred = get_predicted_lanes(runner, enh_img)

        raw_metric = culane_metric.culane_metric(
            raw_pred, gt_lanes, width=LANE_WIDTH, iou_thresholds=[IOU_THRESHOLD],
            official=True, img_shape=(ORI_H, ORI_W, 3))
        enh_metric = culane_metric.culane_metric(
            enh_pred, gt_lanes, width=LANE_WIDTH, iou_thresholds=[IOU_THRESHOLD],
            official=True, img_shape=(ORI_H, ORI_W, 3))

        raw_tp, raw_fp, raw_fn = raw_metric[IOU_THRESHOLD]
        enh_tp, enh_fp, enh_fn = enh_metric[IOU_THRESHOLD]
        tp_lost = int(raw_tp) - int(enh_tp)

        rows.append({
            "fname": fname, "n_gt": len(gt_lanes),
            "raw_tp": int(raw_tp), "raw_fp": int(raw_fp), "raw_fn": int(raw_fn),
            "enh_tp": int(enh_tp), "enh_fp": int(enh_fp), "enh_fn": int(enh_fn),
            "tp_lost": tp_lost,
        })

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(img_paths)} scanned")

    rows.sort(key=lambda r: r["tp_lost"], reverse=True)

    print(f"\n=== [{args.label}] top {args.top_n} frames by TP lost (raw_tp - enh_tp) at IoU@{IOU_THRESHOLD} ===")
    print(f"{'fname':45s} {'n_gt':>4s} {'raw(tp/fp/fn)':>15s} {'enh(tp/fp/fn)':>15s} {'tp_lost':>8s}")
    for r in rows[: args.top_n]:
        raw_str = f"{r['raw_tp']}/{r['raw_fp']}/{r['raw_fn']}"
        enh_str = f"{r['enh_tp']}/{r['enh_fp']}/{r['enh_fn']}"
        print(f"{r['fname']:45s} {r['n_gt']:4d} {raw_str:>15s} {enh_str:>15s} {r['tp_lost']:8d}")

    n_worse = sum(1 for r in rows if r["tp_lost"] > 0)
    n_better = sum(1 for r in rows if r["tp_lost"] < 0)
    n_same = len(rows) - n_worse - n_better
    print(f"\nAcross {len(rows)} images: enhanced worse on {n_worse}, better on {n_better}, tied on {n_same}.")
    print("Pick a --fname from the top of this list and re-run in single-frame mode to visualize it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_PATH)
    ap.add_argument("--label", default="original", help="short tag, just for print/annotation")
    ap.add_argument("--fname", default=None, help="basename, e.g. 05081020_0305.MP4_00120.jpg (single-frame mode)")
    ap.add_argument("--out", default="~/hardcase_compare.jpg", help="output path (single-frame mode)")
    ap.add_argument("--scan", action="store_true", help="count-based scan mode")
    ap.add_argument("--scan_official", action="store_true", help="official-metric-based scan mode (recommended)")
    ap.add_argument("--top_n", type=int, default=30, help="how many hits/rows to print in scan modes")
    args = ap.parse_args()

    args.config = os.path.expanduser(args.config)
    args.ckpt = os.path.expanduser(args.ckpt)

    if args.scan_official:
        scan_official_mode(args)
    elif args.scan:
        scan_mode(args)
    else:
        if not args.fname:
            print("ERROR: --fname is required in single-frame mode (or pass --scan / --scan_official). "
                  "Don't know the filename of the frame you already found? Run --scan_official first.")
            sys.exit(1)
        single_frame_mode(args)


if __name__ == "__main__":
    main()
