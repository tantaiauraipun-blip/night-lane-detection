"""
prepare_finetune_and_heldout.py

Two jobs, run together since both need the night_lane_only_enhance v3
pipeline and touch the same HGLane directory tree:

  A) Build a NIGHT-ONLY, ENHANCED fine-tuning training list, by:
       - filtering list/train_gt.txt down to /night/ rows (~3500 images --
         these are images the base checkpoint (11.pth) already trained on,
         which is expected/fine for a *fine-tune*, not a leakage concern)
       - enhancing each with the fixed v3 lane_only_enhance() pipeline into
         a new night_enhanced/ folder (copying the matching .lines.txt
         alongside -- CLRNet's CULane loader reads GT from
         <image_path>[:-3]+'lines.txt', same basename, new folder)
       - writing a new list/train_gt.txt with the image column repointed at
         /night_enhanced/... while leaving the label-PNG column and the
         4 lane-exist flags untouched (pixel enhancement doesn't move lane
         geometry, so segmentation labels stay valid as-is)
       - backing up the original list/train_gt.txt first (one-time; will
         NOT overwrite an existing backup, so this script is safe to re-run)

  B) Build a clean, held-out (0% overlap with train.txt, verified earlier
     in chat) night test set from list/test.txt's /night/ rows (~1000
     images):
       - copies of the RAW images + .lines.txt -> ~/culane_night_heldout_raw
       - an ENHANCED copy (v3 pipeline)        -> ~/culane_night_heldout_enhanced
     This replaces ~/culane_night_test, which turned out to be 100%
     contained in list/train.txt (data leakage) -- see chat history.

IMPORTANT: after running part A, delete the stale dataset cache before
fine-tuning, or CLRNet's loader will silently keep using the OLD (full,
unfiltered) list:
    rm -f ~/CLRNet/CLRNet/cache/culane_train.pkl

Run from any env with cv2 + numpy and access to night_lane_only_enhance.py
(e.g. the same env night_lane_only_enhance.py was run in before). Does NOT
need torch/CLRNet itself -- this is pure data prep.

Usage:
    python prepare_finetune_and_heldout.py [--hglane_root ~/datasets/HGLane]
        [--skip_finetune_list] [--skip_heldout]
"""

import os
import sys
import shutil
import argparse

import cv2

# night_lane_only_enhance.py must be importable -- adjust if you keep it
# somewhere other than ~/gen_test
sys.path.insert(0, os.path.expanduser("~/gen_test"))
from night_lane_only_enhance import lane_only_enhance  # noqa: E402


def enhance_one(src_jpg_path, dst_jpg_path):
    img = cv2.imread(src_jpg_path)
    if img is None:
        print(f"  [WARN] could not read {src_jpg_path}, skipping")
        return False
    out, _, _ = lane_only_enhance(img, mask_width=22)
    cv2.imwrite(dst_jpg_path, out)
    return True


def copy_lines_txt(src_jpg_path, dst_jpg_path):
    src_lines = src_jpg_path[:-3] + "lines.txt"
    dst_lines = dst_jpg_path[:-3] + "lines.txt"
    if os.path.exists(src_lines):
        shutil.copyfile(src_lines, dst_lines)
        return True
    print(f"  [WARN] no .lines.txt found for {src_jpg_path}")
    return False


def build_finetune_list(hglane_root):
    print("=== A) Building night-only enhanced fine-tune training list ===")
    list_dir = os.path.join(hglane_root, "list")
    train_gt_path = os.path.join(list_dir, "train_gt.txt")
    train_gt_backup = os.path.join(list_dir, "train_gt.txt.orig_backup")
    night_enhanced_dir = os.path.join(hglane_root, "night_enhanced")

    assert os.path.exists(train_gt_path), f"missing {train_gt_path}"

    if not os.path.exists(train_gt_backup):
        shutil.copyfile(train_gt_path, train_gt_backup)
        print(f"Backed up original train_gt.txt -> {train_gt_backup}")
    else:
        print(f"Backup already exists at {train_gt_backup} (not overwriting, "
              f"reading from it so re-runs are idempotent)")

    with open(train_gt_backup) as f:
        all_lines = [l.strip() for l in f if l.strip()]

    night_lines = [l for l in all_lines if l.split()[0].lstrip("/").startswith("night/")]
    print(f"Found {len(night_lines)} night rows in train_gt.txt (of {len(all_lines)} total)")

    os.makedirs(night_enhanced_dir, exist_ok=True)

    new_lines = []
    n_ok, n_fail = 0, 0
    for i, line in enumerate(night_lines):
        parts = line.split()
        img_col = parts[0].lstrip("/")            # e.g. night/night_123.jpg
        fname = os.path.basename(img_col)          # night_123.jpg
        src_jpg = os.path.join(hglane_root, img_col)
        dst_jpg = os.path.join(night_enhanced_dir, fname)

        ok = enhance_one(src_jpg, dst_jpg)
        ok = ok and copy_lines_txt(src_jpg, dst_jpg)
        if not ok:
            n_fail += 1
            continue
        n_ok += 1

        new_img_col = "/night_enhanced/" + fname
        new_parts = [new_img_col] + parts[1:]       # keep label path + flags unchanged
        new_lines.append(" ".join(new_parts))

        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(night_lines)} done")

    with open(train_gt_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"Done: {n_ok} enhanced+written, {n_fail} skipped.")
    print(f"New train_gt.txt has {len(new_lines)} rows (was {len(all_lines)} before filtering).")
    print("NOTE: run  rm -f ~/CLRNet/CLRNet/cache/culane_train.pkl  before fine-tuning, "
          "or the loader will silently use the stale cached (old, full) list.")
    print(f"To restore the original full training list later: "
          f"cp {train_gt_backup} {train_gt_path} && rm -f ~/CLRNet/CLRNet/cache/culane_train.pkl")


def build_heldout_set(hglane_root):
    print("\n=== B) Building clean held-out night test set (from list/test.txt) ===")
    list_dir = os.path.join(hglane_root, "list")
    test_list_path = os.path.join(list_dir, "test.txt")
    heldout_raw_dir = os.path.expanduser("~/culane_night_heldout_raw")
    heldout_enh_dir = os.path.expanduser("~/culane_night_heldout_enhanced")

    assert os.path.exists(test_list_path), f"missing {test_list_path}"

    with open(test_list_path) as f:
        test_lines = [l.strip() for l in f if l.strip()]

    night_test_lines = [l for l in test_lines if l.split()[0].lstrip("/").startswith("night/")]
    print(f"Found {len(night_test_lines)} night rows in test.txt (of {len(test_lines)} total)")

    os.makedirs(heldout_raw_dir, exist_ok=True)
    os.makedirs(heldout_enh_dir, exist_ok=True)

    n_ok, n_fail = 0, 0
    for i, line in enumerate(night_test_lines):
        img_col = line.split()[0].lstrip("/")
        fname = os.path.basename(img_col)
        src_jpg = os.path.join(hglane_root, img_col)
        raw_dst = os.path.join(heldout_raw_dir, fname)
        enh_dst = os.path.join(heldout_enh_dir, fname)

        if not os.path.exists(src_jpg):
            print(f"  [WARN] missing source image {src_jpg}")
            n_fail += 1
            continue

        shutil.copyfile(src_jpg, raw_dst)
        copy_lines_txt(src_jpg, raw_dst)

        ok = enhance_one(src_jpg, enh_dst)
        ok = ok and copy_lines_txt(src_jpg, enh_dst)
        if not ok:
            n_fail += 1
            continue
        n_ok += 1

        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(night_test_lines)} done")

    print(f"Done: {n_ok} images copied+enhanced, {n_fail} skipped.")
    print(f"Raw held-out set:      {heldout_raw_dir}  ({n_ok} images)")
    print(f"Enhanced held-out set: {heldout_enh_dir}  ({n_ok} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hglane_root", default=os.path.expanduser("~/datasets/HGLane"))
    ap.add_argument("--skip_finetune_list", action="store_true")
    ap.add_argument("--skip_heldout", action="store_true")
    args = ap.parse_args()

    if not args.skip_finetune_list:
        build_finetune_list(args.hglane_root)
    if not args.skip_heldout:
        build_heldout_set(args.hglane_root)


if __name__ == "__main__":
    main()
