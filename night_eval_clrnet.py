"""
night_eval_clrnet.py
Evaluate CLRNet lane detection on the CLEAN held-out night set (raw vs.
lane_only-enhanced), against REAL ground-truth lane annotations
(.lines.txt from CULane).

*** v2 (21 Sep 2026) changes vs the version used for v1/v2/v3 lane_only
runs earlier in this project: ***
  - IMAGE_SETS now points at ~/culane_night_heldout_raw and
    ~/culane_night_heldout_enhanced (built by prepare_finetune_and_heldout.py
    from list/test.txt's night rows -- confirmed 0% overlap with
    train.txt). The OLD ~/culane_night_test set is NOT used any more: it
    was found to be 100% contained in train.txt (data leakage), so every
    F1 number from before this change was measuring partially-memorized
    performance, not generalization. See project chat history / summary
    doc for the full writeup.
  - GT (.lines.txt) is now read from ALONGSIDE each image in its own set
    directory (copy_lines_txt() in prepare_finetune_and_heldout.py copies
    it there for both the raw and enhanced held-out copies), instead of a
    single fixed GT_DIR -- the raw and enhanced dirs are self-contained.
  - CONFIG_PATH / CKPT_PATH are now CLI flags (--config/--ckpt), so the
    SAME script evaluates either the original checkpoint or a fine-tuned
    one without editing the file. --label tags the printed section names
    and the worst-N output filename so runs don't overwrite each other.

Must be run from ~/CLRNet/CLRNet in the clrnet_torch113 conda env, e.g.:

    conda deactivate      # repeat if you're nested in another env
    conda activate clrnet_torch113
    unset PYTORCH_CUDA_ALLOC_CONF
    cd ~/CLRNet/CLRNet

    # original (pre-finetune) checkpoint, for the "before" numbers:
    python ~/gen_test/night_eval_clrnet.py \\
        --config work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/config.py \\
        --ckpt   work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/ckpt/11.pth \\
        --label  original

    # fine-tuned checkpoint, for the "after" numbers (adjust run-dir name
    # and epoch number to whatever main.py actually produced):
    python ~/gen_test/night_eval_clrnet.py \\
        --config work_dirs/clr/r18_culane_night_finetune/<run>/config.py \\
        --ckpt   work_dirs/clr/r18_culane_night_finetune/<run>/ckpt/2.pth \\
        --label  finetuned

Run both, then compare the 4 numbers (original x raw, original x
enhanced, finetuned x raw, finetuned x enhanced) side by side.

NOTE: proxy_F1@20px below is NOT the official CULane IoU@0.5 evaluation
tool (that tool expands lanes into ~30px-wide strips and computes mask
IoU). It is a simpler distance-based proxy using the same Hungarian
matching + penalized-drift methodology already used earlier in this
project, with a fixed pixel-distance cutoff standing in for "correct".
Treat it as directional, not as a literature-comparable F1@50 number.
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import cv2
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.getcwd())  # run from ~/CLRNet/CLRNet so `clrnet` package is importable
from clrnet.utils.config import Config
from clrnet.engine.runner import Runner

# ---- defaults: overridable via --config/--ckpt (see argparse in main()) ----
DEFAULT_CONFIG_PATH = os.path.expanduser(
    "~/CLRNet/CLRNet/work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/config.py"
)
DEFAULT_CKPT_PATH = os.path.expanduser(
    "~/CLRNet/CLRNet/work_dirs/clr/r18_culane/20260728_134613_lr_6e-04_b_4/ckpt/11.pth"
)

# clean, verified-0%-overlap-with-train.txt held-out sets (see module
# docstring). GT .lines.txt lives alongside the images in EACH dir.
IMAGE_SETS = {
    "heldout_raw": os.path.expanduser("~/culane_night_heldout_raw"),
    "heldout_enhanced": os.path.expanduser("~/culane_night_heldout_enhanced"),
}

PENALTY_PX = 150.0
MATCH_THRESHOLD_PX = 20.0  # proxy "TP" distance cutoff -- see NOTE above

# CLRNet preprocessing constants (ori_img_w=1640, ori_img_h=590, img_w=800, img_h=320, cut_height=270)
ORI_W, ORI_H = 1640, 590
IMG_W, IMG_H = 800, 320
CUT_HEIGHT = 270


def load_gt_lanes(lines_txt_path):
    """Parse a CULane .lines.txt file into a list of lanes, each a list of (x, y) floats."""
    lanes = []
    with open(lines_txt_path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 4:
                continue
            pts = [(float(vals[i]), float(vals[i + 1])) for i in range(0, len(vals) - 1, 2)]
            lanes.append(pts)
    return lanes


def gt_x_at_y(lane_pts, y_query):
    """Linearly interpolate a lane's x at a given y (CULane files list points from high y to low y)."""
    pts = sorted(lane_pts, key=lambda p: p[1])  # ascending y
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    if y_query < ys[0] or y_query > ys[-1]:
        return None
    return float(np.interp(y_query, ys, xs))


def preprocess_image(img_bgr):
    """Reproduce the CLRNet manual preprocessing pipeline used earlier in this project."""
    full = cv2.resize(img_bgr, (ORI_W, ORI_H))
    cropped = full[CUT_HEIGHT:, :, :]
    resized = cv2.resize(cropped, (IMG_W, IMG_H))
    img = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    return tensor


def get_predicted_lanes(runner, img_bgr):
    """Run the CLRNet forward pass and decode predicted lanes into original-image pixel coordinates."""
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


def match_and_score(pred_lanes, gt_lanes):
    """
    Hungarian-match predicted lanes to GT lanes using mean point-wise x
    distance (at shared y-values) as cost. Returns avg penalized drift plus
    TP/FP/FN counts using MATCH_THRESHOLD_PX as the proxy "correct" cutoff.
    """
    n_gt = len(gt_lanes)
    n_pred = len(pred_lanes)

    if n_gt == 0:
        return None  # nothing to score this image against

    if n_pred == 0:
        return {"avg_drift": PENALTY_PX, "n_gt": n_gt, "n_pred": 0, "tp": 0, "fp": 0, "fn": n_gt}

    cost = np.full((n_pred, n_gt), PENALTY_PX, dtype=np.float64)

    for pi, plane in enumerate(pred_lanes):
        for gi, glane in enumerate(gt_lanes):
            diffs = []
            for (x_p, y_p) in plane:
                if y_p < 230 or y_p > 589:
                    continue
                gx = gt_x_at_y(glane, y_p)
                if gx is not None:
                    diffs.append(abs(x_p - gx))
            if diffs:
                cost[pi, gi] = float(np.mean(diffs))

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_gt = set()
    all_diffs = []
    tp, fp = 0, 0
    for pi, gi in zip(row_ind, col_ind):
        d = cost[pi, gi]
        matched_gt.add(gi)
        all_diffs.append(d if d < PENALTY_PX else PENALTY_PX)
        if d <= MATCH_THRESHOLD_PX:
            tp += 1
        else:
            fp += 1

    fp += max(0, n_pred - len(row_ind))  # predicted lanes the assignment never paired at all

    for gi in range(n_gt):
        if gi not in matched_gt:
            all_diffs.append(PENALTY_PX)

    fn = n_gt - tp

    return {
        "avg_drift": float(np.mean(all_diffs)),
        "n_gt": n_gt,
        "n_pred": n_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def summarize(name, results, label):
    results = [r for r in results if r is not None]
    if not results:
        print(f"{label}/{name}: no valid results")
        return None
    avg_drift = np.mean([r["avg_drift"] for r in results])
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"\n=== [{label}] {name} ===")
    print(f"  images scored              : {len(results)}")
    print(f"  avg drift (penalized, px)  : {avg_drift:.2f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  precision={precision:.4f}  recall={recall:.4f}  proxy_F1@{MATCH_THRESHOLD_PX:.0f}px={f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1, "avg_drift": avg_drift,
            "tp": tp, "fp": fp, "fn": fn, "n": len(results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                     help="path to the CLRNet run's config.py (relative to ~/CLRNet/CLRNet or absolute)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_PATH,
                     help="path to the .pth checkpoint to evaluate (relative to ~/CLRNet/CLRNet or absolute)")
    ap.add_argument("--label", default="original",
                     help="short tag for this run, used in printed headers and the worst-N output filename "
                          "(e.g. 'original' or 'finetuned') so repeated runs don't overwrite each other")
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
        for img_path in img_paths:
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]
            # GT now lives alongside the image in this same set dir (see
            # module docstring -- copy_lines_txt() put it there).
            lines_path = os.path.join(img_dir, stem + ".lines.txt")
            if not os.path.exists(lines_path):
                print(f"  [skip] no GT for {fname}")
                continue

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  [skip] could not read {fname}")
                continue

            gt_lanes = load_gt_lanes(lines_path)
            pred_lanes = get_predicted_lanes(runner, img_bgr)
            score = match_and_score(pred_lanes, gt_lanes)
            if score is not None:
                score["fname"] = fname
            results.append(score)

        all_results[set_name] = results

    summary = {}
    for set_name, results in all_results.items():
        summary[set_name] = summarize(set_name, results, args.label)

    print(f"\n=== [{args.label}] summary table ===")
    print(f"{'set':20s} {'n':>5s} {'precision':>10s} {'recall':>8s} {'proxy_F1':>9s} {'avg_drift':>10s}")
    for set_name, s in summary.items():
        if s is None:
            continue
        print(f"{set_name:20s} {s['n']:5d} {s['precision']:10.4f} {s['recall']:8.4f} "
              f"{s['f1']:9.4f} {s['avg_drift']:10.2f}")

    # --- diagnostic: which specific heldout_raw images does CLRNet actually miss on? ---
    raw_results = [r for r in all_results.get("heldout_raw", []) if r is not None]
    worst = sorted(raw_results, key=lambda r: (r["fn"], r["avg_drift"]), reverse=True)[:20]
    print(f"\n=== [{args.label}] worst 20 heldout_raw images (by FN, then avg_drift) ===")
    out_path = os.path.expanduser(f"~/culane_night_worst_heldout_raw_{args.label}.txt")
    with open(out_path, "w") as f:
        for r in worst:
            line = f"{r['fname']}  n_gt={r['n_gt']} n_pred={r['n_pred']} tp={r['tp']} fp={r['fp']} fn={r['fn']} avg_drift={r['avg_drift']:.1f}"
            print("  " + line)
            f.write(line + "\n")
    print(f"\nFull list written to {out_path}")

    # --- per-image dump for later statistical analysis (e.g. paired bootstrap
    # of the raw-vs-enhanced F1 gap -- see bootstrap_significance.py) ---
    dump = {set_name: [r for r in results if r is not None] for set_name, results in all_results.items()}
    dump_path = os.path.expanduser(f"~/culane_night_eval_perimage_{args.label}.json")
    with open(dump_path, "w") as f:
        json.dump(dump, f)
    print(f"Per-image results dumped to {dump_path}")


if __name__ == "__main__":
    main()
