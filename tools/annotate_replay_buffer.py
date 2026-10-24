"""
Add per-frame class information to a replay buffer JSON (in place, idempotent).

For every frame it records
    supervised_classes  - the classes this frame supervises: those introduced at its origin step
    present_classes     - every class actually present in the frame's ground-truth voxels, with the
                          voxel count and whether this frame supervises it
    empty_voxels / unknown_voxels
The dataset loader only reads sequence / frame_id, so the extra fields do not change training.

Example:
  python tools/annotate_replay_buffer.py datasets/Fixed/replay_buffers/2D_GT/step1_classaware_49_2DGT.json
"""
import argparse
import json
import os.path as osp
import shutil

import numpy as np
import yaml

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

CLASS_NAMES = (
    'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
    'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
    'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign',
)
UNKNOWN = 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buffers", nargs="+")
    ap.add_argument("--backup-dir", default=None, help="copy the original JSON here before rewriting")
    args = ap.parse_args()

    with open(osp.join(REPO_ROOT, "configs/datasets/semantic_kitti.yaml")) as f:
        ds = yaml.safe_load(f)["data"]["datasets"]
    label_root = ds["label_root"]
    splits = {int(k): list(v) for k, v in ds["continual_splits"].items()}

    for path in args.buffers:
        with open(path) as f:
            buf = json.load(f)
        if args.backup_dir:
            shutil.copy(path, osp.join(args.backup_dir, osp.basename(path)))

        for fr in buf["frames"]:
            sup = splits[fr["origin_step"]]
            lab = np.load(osp.join(label_root, fr["sequence"], f"{fr['frame_id']}_1_1.npy"))
            ids, cnt = np.unique(lab, return_counts=True)
            counts = dict(zip(ids.tolist(), cnt.tolist()))
            fr["supervised_classes"] = [{"id": g, "name": CLASS_NAMES[g]} for g in sup]
            fr["present_classes"] = [
                {"id": g, "name": CLASS_NAMES[g], "voxels": counts[g], "supervised": g in sup}
                for g in sorted(counts, key=lambda g: -counts[g]) if g not in (0, UNKNOWN)
            ]
            fr["empty_voxels"] = counts.get(0, 0)
            fr["unknown_voxels"] = counts.get(UNKNOWN, 0)

        with open(path, "w") as f:
            json.dump(buf, f, indent=1)
        n_unsup = sum(1 for fr in buf["frames"] for c in fr["present_classes"] if not c["supervised"])
        print(f"[annotate] {path}: {len(buf['frames'])} frames annotated "
              f"({n_unsup} present-but-unsupervised class entries)")


if __name__ == "__main__":
    main()
