# Nighttime Lane Detection — Enhancement Pipeline & Evaluation

Independent evaluation of a lightweight, lane-aware image-enhancement pipeline
for nighttime lane detection with CLRNet (ResNet-18 backbone) on CULane.
Covers the enhancement pipeline itself, fine-tuning, evaluation under both a
fast proxy metric and the official CULane metric, statistical significance
testing, and qualitative case-study visualization.

See the full walkthrough (narration script + written reference manual) for a
complete guided run: `Operating_Manual_NightLaneDetection_23Sep2026_EN.docx`.

## Repository layout

```
.
├── night_lane_only_enhance.py       # 4-step enhancement pipeline (core)
├── night_enhance.py                 # earlier CLAHE-based enhancement variant
├── prepare_finetune_and_heldout.py  # builds fine-tune training list + leak-free held-out set
├── finetune_night_config.py         # CLRNet fine-tuning config (run via CLRNet's main.py)
├── night_eval_clrnet.py             # proxy metric evaluation (Hungarian x-distance F1@20px)
├── official_culane_eval.py          # official CULane metric evaluation (IoU@0.5)
├── bootstrap_significance.py        # paired bootstrap significance testing (10,000 resamples)
├── visualize_hard_cases.py          # qualitative output images (single-frame + scan modes)
├── night_visual_compare.py          # quick no-GPU visual spot-check (original/selective/aware)
├── night_worst_grid.py              # grid of worst-N baseline failures
└── docs/                            # reports, slides, and the operating manual
```

## Prerequisites

- A conda environment named `clrnet_torch113` with PyTorch, OpenCV, NumPy,
  and [CLRNet](https://github.com/Turoad/CLRNet)'s own dependencies
  installed. See `environment.yml` for a starting point — adjust package
  versions to match what CLRNet's own `requirements.txt` needs for your CUDA
  version.
- The CLRNet repository checked out separately (not included here — it's a
  third-party codebase), with a pretrained checkpoint available.
- The CULane dataset (also not included here — see the [official CULane
  page](https://xingangpan.github.io/projects/CULane.html) for download
  instructions). This repo does **not** ship any dataset images, model
  checkpoints, or trained weights.

Two scripts — `night_lane_only_enhance.py` and the visual-compare / worst-grid
utilities — only need OpenCV and NumPy, no PyTorch. Everything from data
preparation onward needs the full `clrnet_torch113` environment.

## Quick start

```bash
conda activate clrnet_torch113
unset PYTORCH_CUDA_ALLOC_CONF
cd ~/CLRNet/CLRNet     # your local CLRNet checkout, not part of this repo

python /path/to/this/repo/night_lane_only_enhance.py \
    --input_dir ~/culane_night_test \
    --output_dir ~/culane_night_lane_only \
    --debug_dir ~/culane_night_lane_only_debug \
    --mask_width 22
```

For the complete step-by-step sequence — data prep, fine-tuning, both
evaluation metrics, significance testing, and qualitative visualization —
follow `docs/Operating_Manual_NightLaneDetection_23Sep2026_EN.docx` (or the
Thai version, `..._TH.docx`).

## Key findings (summary)

- Enhancement applied only at inference time never beats the unenhanced
  baseline, and the most technically correct version scores worst —
  evidence the effect is distributional, not implementation error.
- The original held-out test set was found to be 100% duplicated inside the
  training set; results after that point use a rebuilt, verified leak-free
  set with zero filename overlap.
- Under the official CULane metric with paired bootstrap significance
  testing, enhancement significantly hurts the non-fine-tuned checkpoint
  (p ≈ 0.002); fine-tuning neutralizes that harm but does not reverse it
  into a proven benefit.
- See `docs/Final_Report_NightLaneDetection_23Sep2026_EN.docx` for the full
  write-up, including comparison against CLRNet and HG-Lane.

## License / academic use

Add your course's or department's preferred license here before publishing
(e.g. MIT for permissive reuse, or a note restricting reuse for academic
integrity purposes). No license is currently declared.
