import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

# ✅ CHANGE THIS TO YOUR TIMESTAMP FOLDER
BASE_DIR = "eval_outputs/SemanticKITTI/<run_timestamp>/test/sequences"

# thresholds you want to analyze
TAUS = [0.6, 0.7, 0.8, 0.9, 0.95]

# If you want to ignore "empty dominance" effects, you can optionally compute coverage only
# on non-empty predicted voxels. Start with False (simple).
ONLY_NONEMPTY = False


def iter_npz_files(base_dir: str):
    for seq in sorted(os.listdir(base_dir)):
        pred_dir = osp.join(base_dir, seq, "predictions")
        if not osp.isdir(pred_dir):
            continue
        for fn in sorted(os.listdir(pred_dir)):
            if fn.endswith(".teacher_old.npz"):
                yield osp.join(pred_dir, fn)


def main():
    npz_files = list(iter_npz_files(BASE_DIR))
    if len(npz_files) == 0:
        raise RuntimeError(f"No *.teacher_old.npz found under: {BASE_DIR}")

    print(f"Found {len(npz_files)} npz files.")

    # Collect coverage values per tau across frames
    cov_by_tau = {tau: [] for tau in TAUS}

    for p in npz_files:
        data = np.load(p)
        conf = data["teacher_conf_old"]   # (L,W,H)
        pred = data["teacher_pred_old"]   # (L,W,H)

        if ONLY_NONEMPTY:
            # Consider only voxels that teacher predicts as non-empty (>0 and <255)
            valid = (pred > 0) & (pred < 255)
        else:
            # All voxels contribute (simple; good enough for first curve)
            valid = np.ones_like(conf, dtype=bool)

        valid_count = np.maximum(valid.sum(), 1)

        for tau in TAUS:
            cov = ((conf > tau) & valid).sum() / valid_count
            cov_by_tau[tau].append(float(cov))

    # Stats
    taus = np.array(TAUS, dtype=float)
    means = np.array([np.mean(cov_by_tau[t]) for t in TAUS], dtype=float)
    stds  = np.array([np.std(cov_by_tau[t]) for t in TAUS], dtype=float)
    mins  = np.array([np.min(cov_by_tau[t]) for t in TAUS], dtype=float)
    maxs  = np.array([np.max(cov_by_tau[t]) for t in TAUS], dtype=float)
    meds  = np.array([np.median(cov_by_tau[t]) for t in TAUS], dtype=float)

    print("\nCoverage stats across frames:")
    for i, tau in enumerate(TAUS):
        print(f"  tau={tau:>4}: mean={means[i]:.4f}  std={stds[i]:.4f}  "
              f"median={meds[i]:.4f}  min={mins[i]:.4f}  max={maxs[i]:.4f}")

    # Plot
    plt.figure()
    plt.plot(taus, means, marker="o")
    plt.fill_between(taus, means - stds, means + stds, alpha=0.2)

    plt.xlabel("confidence threshold τ")
    plt.ylabel("coverage (fraction of kept voxels)")
    title = "τ vs coverage"
    if ONLY_NONEMPTY:
        title += " (non-empty voxels only)"
    plt.title(title)

    out_png = osp.join(osp.dirname(BASE_DIR), "tau_coverage_curve.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"\nSaved plot to: {out_png}")


if __name__ == "__main__":
    main()