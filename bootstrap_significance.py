"""
bootstrap_significance.py
Paired bootstrap significance test for the raw-vs-enhanced proxy F1 gap, per
checkpoint, using the per-image TP/FP/FN dumps written by night_eval_clrnet.py
(its --label run auto-writes ~/culane_night_eval_perimage_<label>.json).

Answers: for a given checkpoint, is (enhanced_F1 - raw_F1) distinguishable
from zero given image-to-image noise, or could the observed sign just be
what you'd get from resampling the same 1,000 images?

Usage (run in the same env as night_eval_clrnet.py, e.g. clrnet_torch113 --
only numpy is required):

    python ~/gen_test/bootstrap_significance.py original finetuned finetuned_ext

(pass whichever --label values you have JSON dumps for; default is those
three if no args given)
"""
import sys
import json
import os
import numpy as np

N_BOOT = 10000
SEED = 0


def load(label):
    path = os.path.expanduser(f"~/culane_night_eval_perimage_{label}.json")
    with open(path) as f:
        return json.load(f)


def to_dict_by_fname(results):
    return {r["fname"]: r for r in results}


def f1_from_agg(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def paired_bootstrap(raw_results, enh_results, n_boot=N_BOOT, seed=SEED):
    raw_by_fname = to_dict_by_fname(raw_results)
    enh_by_fname = to_dict_by_fname(enh_results)
    common = sorted(set(raw_by_fname) & set(enh_by_fname))
    n = len(common)
    n_raw_only = len(raw_by_fname) - n
    n_enh_only = len(enh_by_fname) - n
    print(f"  paired images: {n} (raw-only: {n_raw_only}, enhanced-only: {n_enh_only})")

    raw_tp = np.array([raw_by_fname[f]["tp"] for f in common], dtype=np.float64)
    raw_fp = np.array([raw_by_fname[f]["fp"] for f in common], dtype=np.float64)
    raw_fn = np.array([raw_by_fname[f]["fn"] for f in common], dtype=np.float64)
    enh_tp = np.array([enh_by_fname[f]["tp"] for f in common], dtype=np.float64)
    enh_fp = np.array([enh_by_fname[f]["fp"] for f in common], dtype=np.float64)
    enh_fn = np.array([enh_by_fname[f]["fn"] for f in common], dtype=np.float64)

    _, _, raw_f1 = f1_from_agg(raw_tp.sum(), raw_fp.sum(), raw_fn.sum())
    _, _, enh_f1 = f1_from_agg(enh_tp.sum(), enh_fp.sum(), enh_fn.sum())
    observed_diff = enh_f1 - raw_f1

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, _, r_f1 = f1_from_agg(raw_tp[idx].sum(), raw_fp[idx].sum(), raw_fn[idx].sum())
        _, _, e_f1 = f1_from_agg(enh_tp[idx].sum(), enh_fp[idx].sum(), enh_fn[idx].sum())
        diffs[b] = e_f1 - r_f1

    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value approx: fraction of resamples that cross to
    # the opposite sign from the observed difference, doubled
    if observed_diff >= 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    p = min(p, 1.0)

    return {
        "raw_f1": raw_f1, "enh_f1": enh_f1, "observed_diff": observed_diff,
        "boot_mean_diff": diffs.mean(), "ci_lo": ci_lo, "ci_hi": ci_hi,
        "p_approx": p, "n": n,
    }


def main():
    labels = sys.argv[1:] or ["original", "finetuned", "finetuned_ext"]
    print(f"{'label':16s} {'raw_F1':>8s} {'enh_F1':>8s} {'diff':>9s} {'95% CI':>20s} {'p(approx)':>10s}  significant?")
    for label in labels:
        try:
            data = load(label)
        except FileNotFoundError:
            print(f"  [skip] no dump for label={label} "
                  f"(run night_eval_clrnet.py --label {label} first, with the updated script)")
            continue
        res = paired_bootstrap(data["heldout_raw"], data["heldout_enhanced"])
        sig = "NO (CI spans 0)" if res["ci_lo"] <= 0 <= res["ci_hi"] else "yes"
        print(f"{label:16s} {res['raw_f1']:8.4f} {res['enh_f1']:8.4f} {res['observed_diff']:+9.4f} "
              f"[{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] {res['p_approx']:10.3f}  {sig}")


if __name__ == "__main__":
    main()
