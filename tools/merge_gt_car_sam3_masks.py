import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm

IGNORE_INDEX = 255


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge 2D masks: car class from GT (LiDAR-projected) masks, "
                     "all other classes from SAM3 masks."
    )
    parser.add_argument("--sam3-root", default="datasets/Fixed/SAM3_fixed/step2",
                         help="Existing SAM3 mask root (per-sequence PNGs, compact ids).")
    parser.add_argument("--gt-root", default="datasets/Fixed/lidar_2d_masks_fixed/step2",
                         help="Existing GT (LiDAR-projected) mask root, same layout/frames as --sam3-root.")
    parser.add_argument("--out-root", default="datasets/Fixed/SAM3_fixed/step2_car_GT_rest_SAM3",
                         help="Output root for merged masks.")
    parser.add_argument("--car-class-id", type=int, default=12,
                         help="Compact class id for 'car' in step-2 seen-compact space.")
    parser.add_argument("--ignore-index", type=int, default=IGNORE_INDEX)
    return parser.parse_args()


def merge_mask(sam3_mask, gt_mask, car_id, ignore_index):
    """
    sam3_mask, gt_mask: uint8 arrays, same shape, compact class ids (or ignore_index).

    Result:
      - any pixel SAM3 labeled as car is discarded (-> ignore_index) since SAM3's
        car guess is no longer trusted,
      - GT's car pixels are then stamped on top (overwriting whatever was there),
      - every other pixel keeps SAM3's original value.
    """
    merged = sam3_mask.copy()
    merged[merged == car_id] = ignore_index
    merged[gt_mask == car_id] = car_id
    return merged


def main():
    args = parse_args()

    sequences = sorted(
        d for d in os.listdir(args.sam3_root) if osp.isdir(osp.join(args.sam3_root, d))
    )

    total_frames = 0
    total_car_px_gt = 0
    total_car_px_sam3_discarded = 0

    for seq in sequences:
        sam3_seq_dir = osp.join(args.sam3_root, seq)
        gt_seq_dir = osp.join(args.gt_root, seq)
        out_seq_dir = osp.join(args.out_root, seq)
        os.makedirs(out_seq_dir, exist_ok=True)

        frame_files = sorted(f for f in os.listdir(sam3_seq_dir) if f.endswith(".png"))

        for fname in tqdm(frame_files, desc=f"seq {seq}"):
            sam3_path = osp.join(sam3_seq_dir, fname)
            gt_path = osp.join(gt_seq_dir, fname)

            if not osp.exists(gt_path):
                raise FileNotFoundError(
                    f"GT mask missing for {seq}/{fname} (expected at {gt_path}). "
                    f"SAM3 and GT mask sets must cover identical frames."
                )

            sam3_mask = np.array(Image.open(sam3_path).convert("L"), dtype=np.uint8)
            gt_mask = np.array(Image.open(gt_path).convert("L"), dtype=np.uint8)

            if sam3_mask.shape != gt_mask.shape:
                raise ValueError(
                    f"Shape mismatch for {seq}/{fname}: sam3={sam3_mask.shape} gt={gt_mask.shape}"
                )

            total_car_px_sam3_discarded += int((sam3_mask == args.car_class_id).sum())
            total_car_px_gt += int((gt_mask == args.car_class_id).sum())

            merged = merge_mask(sam3_mask, gt_mask, args.car_class_id, args.ignore_index)

            Image.fromarray(merged, mode="L").save(osp.join(out_seq_dir, fname))

        total_frames += len(frame_files)

    print(f"\nDone. Merged {total_frames} frames across sequences {sequences}.")
    print(f"Car pixels contributed by GT (final):        {total_car_px_gt}")
    print(f"Car pixels discarded from SAM3 (overwritten): {total_car_px_sam3_discarded}")
    print(f"Output written to: {args.out_root}")


if __name__ == "__main__":
    main()
