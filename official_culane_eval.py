"""
official_culane_eval.py
Evaluate CLRNet lane detection using the ACTUAL official CULane metric
(clrnet.utils.culane_metric.culane_metric: lanes rasterized to 30px-wide
strips, IoU@0.5, Hungarian matching) -- NOT the proxy_F1@20px
distance-based metric used by night_eval_clrnet.py.

Reuses the same model loading / preprocessing / held-out image sets as
night_eval_clrnet.py, and the SAME per-image JSON dump convention, so
bootstrap_significance.py works on these dumps completely unchanged --
just pass labels prefixed "official_", e.g.:

    python bootstrap_significance.py official_original official_finetuned official_finetuned_ext

GT-lane loading mirrors clrnet/datasets/culane.py's own __getitem__ loader
exactly (drop negative-coord points, dedupe points, keep only lanes with
>2 points, sort by y) for full fidelity to how the official pipeline
itself scores GT.

Must be run from ~/CLRNet/CLRNet in the clrnet_torch113 conda env, e.g.:

    conda activate clrnet_torch113
    unset PYTORCH_CUDA_ALLOC_CONF
    cd ~/CLRNet/CLRNet

    python ~/gen_test/official_culane_eval.py \\
        --config work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/config.py \\
        --ckpt   work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/ckpt/11.pth \\
        --label  official_original

Precision/recall/F1 use the SAME guard as clrnet.utils.culane_metric.eval_predictions
("if tp != 0" rather than "(tp+fp)>0"), for exact parity with the official numbers.
"""

import os
import sys
import glob
import json
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

# same clean, verified-0%-overlap-with-train.txt held-out sets used by night_eval_clrnet.py
IMAGE_SETS = {
    "heldout_raw": os.path.expanduser("~/culane_night_heldout_raw"),
    "heldout_enhanced": os.path.expanduser("~/culane_night_heldout_enhanced"),
}

IOU_THRESHOLD = 0.5
LANE_WIDTH = 30  # px -- matches the official CULane metric's rasterization width

# CLRNet preprocessing constants (unchanged from night_eval_clrnet.py)
ORI_W, ORI_H = 1640, 590
IMG_W, IMG_H = 800, 320
CUT_HEIGHT = 270


def load_gt_lanes_official(lines_txt_path):
    """Mirror clrnet/datasets/culane.py's __getitem__ GT loading exactly:
    drop negative-coord points, dedupe points, keep only lanes with >2
    points, sort each lane by y. (night_eval_clrnet.py's load_gt_lanes()
    does NOT do this filtering -- it's fine for the proxy metric but would
    silently diverge from official semantics here.)"""
    with open(lines_txt_path, "r") as f:
        data = [list(map(float, line.split())) for line in f.readlines()]
    lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)
              if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
    lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
    lanes = [lane for lane in lanes if len(lane) > 2]  # remove lanes with <=2 points
    lanes = [sorted(lane, key=lambda p: p[1]) for lane in lanes]  # sort by y
    return lanes


def preprocess_image(img_bgr):
    """Reproduce the CLRNet manual preprocessing pipeline (unchanged from night_eval_clrnet.py)."""
    full = cv2.resize(img_bgr, (ORI_W, ORI_H))
    cropped = full[CUT_HEIGHT:, :, :]
    resized = cv2.resize(cropped, (IMG_W, IMG_H))
    img = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return tensor


def get_predicted_lanes(runner, img_bgr):
    """Run the CLRNet forward pass and decode predicted lanes into original-image pixel
    coordinates (unchanged from night_eval_clrnet.py -- this is the same pixel-coordinate
    point-list format culane_metric.culane_metric() expects for `pred`)."""
    tensor = preprocess_image(img_bgr)
    data = {"img": tensor}
    with torch.no_grad():
        output = runner.net(data)
    lanes = runner.net.module.heads.get_lanes(output)[0]  # first (only) image in the batch
    pred_lanes = []
    for lane in lanes:
        arr = lane.to_array(runner.cfg)  # Nx2 array of (x, y) in original image coordinates
        pred_lanes.append([(float(x), float(y)) for x, y in arr])
    return pred_lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                     help="path to the CLRNet run's config.py (relative to ~/CLRNet/CLRNet or absolute)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_PATH,
                     help="path to the .pth checkpoint to evaluate (relative to ~/CLRNet/CLRNet or absolute)")
    ap.add_argument("--label", default="official_original",
                     help="short tag for this run (prefix with 'official_' so it doesn't collide "
                          "with the proxy-metric dumps from night_eval_clrnet.py)")
    args = ap.parse_args()

    config_path = os.path.expanduser(args.config)
    ckpt_path = os.path.expanduser(args.ckpt)
    print(f"[{args.label}] config = {config_path}")
    print(f"[{args.label}] ckpt   = {ckpt_path}")
    assert os.path.exists(config_path), f"missing config: {config_path}"
    assert os.path.exists(ckpt_path), f"missing checkpoint: {ckpt_path}"

    print("Loading CLRNet...")
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
    print("CLRNet loaded.\n")

    all_results = {}
    for set_name, img_dir in IMAGE_SETS.items():
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        print(f"[{args.label}] [{set_name}] {len(img_paths)} images in {img_dir}")
        results = []
        for i, img_path in enumerate(img_paths):
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]
            lines_path = os.path.join(img_dir, stem + ".lines.txt")
            if not os.path.exists(lines_path):
                print(f"  [skip] no GT for {fname}")
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  [skip] could not read {fname}")
                continue

            gt_lanes = load_gt_lanes_official(lines_path)
            pred_lanes = get_predicted_lanes(runner, img_bgr)

            metric = culane_metric.culane_metric(
                pred_lanes, gt_lanes,
                width=LANE_WIDTH,
                iou_thresholds=[IOU_THRESHOLD],
                official=True,
                img_shape=(ORI_H, ORI_W, 3),
            )
            tp, fp, fn = metric[IOU_THRESHOLD]
            results.append({"fname": fname, "tp": int(tp), "fp": int(fp), "fn": int(fn),
                             "n_gt": len(gt_lanes), "n_pred": len(pred_lanes)})

            if (i + 1) % 200 == 0:
                print(f"  [{args.label}] [{set_name}] {i + 1}/{len(img_paths)} done")

        all_results[set_name] = results

    print(f"\n=== [{args.label}] OFFICIAL CULane metric (IoU@{IOU_THRESHOLD}, lane width={LANE_WIDTH}px) summary ===")
    print(f"{'set':20s} {'n':>5s} {'precision':>10s} {'recall':>8s} {'F1':>8s}")
    for set_name, results in all_results.items():
        tp = sum(r["tp"] for r in results)
        fp = sum(r["fp"] for r in results)
        fn = sum(r["fn"] for r in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0.0
        recall = float(tp) / (tp + fn) if tp != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if tp != 0 else 0.0
        print(f"{set_name:20s} {len(results):5d} {precision:10.4f} {recall:8.4f} {f1:8.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}")

    dump_path = os.path.expanduser(f"~/culane_night_eval_perimage_{args.label}.json")
    with open(dump_path, "w") as f:
        json.dump(all_results, f)
    print(f"\nPer-image results dumped to {dump_path}")


if __name__ == "__main__":
    main()
