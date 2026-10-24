import os
import os.path as osp
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from ssc_ia.utils.helper import vox2pix


def unique_preserve_order(xs):
    return list(dict.fromkeys(list(xs)))


def flatten_splits_preserve_order(splits, start, end):
    out = []
    for i in range(start, end):
        out.extend(list(splits[i]))
    return unique_preserve_order(out)


def make_2d_mask_from_3d_gt(
    target_3d,
    projected_pix,
    fov_mask,
    pix_z,
    img_h,
    img_w,
    new_class_ids,
    ignore_index=255,
    occ_mask=None,
):
    """
    target_3d:      [X, Y, Z] compact GT labels (new-class IDs; 255 elsewhere)
    projected_pix:  [X*Y*Z, 2]
    fov_mask:       [X*Y*Z]
    pix_z:          [X*Y*Z]
    occ_mask:       optional [X*Y*Z] bool — occupied (non-empty, non-ignore)
                    voxels for occlusion-correct z-buffering.  When provided
                    the z-buffer runs over ALL occupied voxels so old-class
                    surfaces occlude new-class voxels behind them.  Only pixels
                    where the nearest occupied voxel IS a new-class voxel are
                    labelled.

    Output:
        mask_2d: [H, W]
    """

    target_flat = target_3d.reshape(-1)
    mask_2d = np.full((img_h, img_w), ignore_index, dtype=np.uint8)

    if occ_mask is not None:
        # Occlusion-correct path: z-buffer over ALL occupied FOV voxels, then
        # label only pixels where the nearest occupied voxel is a new-class one.
        valid_all = fov_mask.astype(bool) & occ_mask.reshape(-1).astype(bool)

        if valid_all.sum() == 0:
            return mask_2d

        pix_all    = projected_pix[valid_all].astype(np.int64)
        labels_all = target_flat[valid_all].astype(np.int64)
        depth_all  = pix_z[valid_all]

        x_all = pix_all[:, 0]
        y_all = pix_all[:, 1]

        in_img = (x_all >= 0) & (x_all < img_w) & (y_all >= 0) & (y_all < img_h)
        if in_img.sum() == 0:
            return mask_2d

        x_all      = x_all[in_img]
        y_all      = y_all[in_img]
        labels_all = labels_all[in_img]
        depth_all  = depth_all[in_img]

        pixel_id   = y_all * img_w + x_all
        order      = np.argsort(depth_all)  # nearest first
        pid_sorted = pixel_id[order]
        lbl_sorted = labels_all[order]

        unique_pid, first_idx = np.unique(pid_sorted, return_index=True)
        nearest_labels = lbl_sorted[first_idx]

        # Only label pixels where the nearest occupied voxel is a new class.
        is_new = np.isin(nearest_labels, new_class_ids)
        if is_new.sum() == 0:
            return mask_2d

        chosen_y = unique_pid[is_new] // img_w
        chosen_x = unique_pid[is_new] % img_w
        mask_2d[chosen_y, chosen_x] = nearest_labels[is_new].astype(np.uint8)

    else:
        # Legacy path: z-buffer only new-class voxels (ignores old-class occluders).
        valid = fov_mask.astype(bool)
        class_valid = np.isin(target_flat, new_class_ids)
        valid = valid & class_valid

        if valid.sum() == 0:
            return mask_2d

        pix    = projected_pix[valid].astype(np.int64)
        labels = target_flat[valid].astype(np.uint8)
        depth  = pix_z[valid]

        x = pix[:, 0]
        y = pix[:, 1]

        in_img = (x >= 0) & (x < img_w) & (y >= 0) & (y < img_h)
        if in_img.sum() == 0:
            return mask_2d

        x      = x[in_img]
        y      = y[in_img]
        labels = labels[in_img]
        depth  = depth[in_img]

        pixel_id        = y * img_w + x
        order           = np.argsort(depth)
        pixel_id_sorted = pixel_id[order]
        labels_sorted   = labels[order]

        unique_pixel_id, first_idx = np.unique(pixel_id_sorted, return_index=True)
        chosen_y = unique_pixel_id // img_w
        chosen_x = unique_pixel_id % img_w
        mask_2d[chosen_y, chosen_x] = labels_sorted[first_idx]

    return mask_2d


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D GT masks from 3D SemanticKITTI labels for a given CIL step."
    )
    parser.add_argument("--step", type=int, required=True, choices=[0, 1, 2, 3, 4],
                        help="Continual learning step whose new classes to project.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Dataset split to generate masks for (default: train). "
                             "Run with --split val to also generate masks for val sequences.")
    parser.add_argument("--out-root", required=True,
                        help="Output root directory. Masks are written to OUT_ROOT/SEQ/FRAME.png. "
                             "To avoid overwriting the existing Step2 GT "
                             "(datasets/open_vocab_2d_masks/Step2/), pass a different path.")
    parser.add_argument("--data-root",
                        default="datasets/kitti_odometry")
    parser.add_argument("--label-root",
                        default="datasets/preprocess/labels")
    parser.add_argument("--depth-root",
                        default="datasets/kitti_odometry/dataset/depth")
    parser.add_argument("--max-samples", type=int, default=-1)
    args = parser.parse_args()

    # 5-step DCIL (2026-09-29): other-ground(12)/other-vehicle(5) dropped entirely.
    continual_splits = {
        0: [9, 11, 15, 17],    # road, sidewalk, vegetation, terrain
        1: [14, 16, 18],       # fence, trunk, pole
        2: [1, 6, 19],         # car, person, traffic-sign
        3: [4, 10, 13],        # truck, parking, building
        4: [2, 3, 7, 8],       # bicycle, motorcycle, bicyclist, motorcyclist
    }

    # Seen-compact IDs preserve DCIL insertion order (no sorting).
    step_globals  = unique_preserve_order(continual_splits[args.step])
    seen_global   = flatten_splits_preserve_order(continual_splits, 0, args.step + 1)
    global_to_seen_compact = {g: i + 1 for i, g in enumerate(seen_global)}
    step_seen_compact_ids  = np.array(
        [global_to_seen_compact[g] for g in step_globals], dtype=np.int64
    )
    compact_to_global = {0: 0, **{i + 1: g for i, g in enumerate(seen_global)}}

    print(f"Step {args.step}: projecting {len(step_globals)} new classes")
    print(f"  seen_global (DCIL order): {seen_global}")
    for g in step_globals:
        print(f"  global {g} -> seen-compact {global_to_seen_compact[g]}")
    print(f"Saving masks to: {args.out_root}")

    data_config = {
        "input_size": [192, 640],
        "resize": [0.0, 0.0],
        "rot": [0.0, 0.0],
        "flip": False,
        "crop_h": [0.0, 0.0],
        "resize_test": 0.0,
    }

    from ssc_ia.data.datasets.semantic_kitti import SemanticKITTI

    ds = SemanticKITTI(
        split=args.split,
        data_root=args.data_root,
        label_root=args.label_root,
        depth_root=args.depth_root,
        data_config=data_config,
        project_scale=2,
        frustum_size=4,
        context_prior=False,
        load_pose=False,
        load_only_with_target=True,
        continual_splits=continual_splits,
        continual_step=args.step,
        cil_scenario="disjoint",
        filter_by_new_class=False,
        min_new_class_voxels=0,
        ignore_index=255,
        empty_label=0,
        pseudo=None,
    )

    print("Number of samples:", len(ds))
    print("Saving masks to:", args.out_root)

    total = len(ds) if args.max_samples < 0 else min(args.max_samples, len(ds))

    for idx in tqdm(range(total)):
        scan = ds.scans[idx]

        sequence = scan["sequence"]
        frame_id = scan["frame_id"]

        target_path = scan["target_path"]
        target_global = np.load(target_path)

        # Build target_seen_compact: only current-step classes receive a label
        # (seen-compact ID); everything else stays at 255 (ignore).
        target_seen_compact = np.full_like(target_global, fill_value=255, dtype=np.uint8)
        for g in step_globals:
            target_seen_compact[target_global == g] = global_to_seen_compact[g]

        img_path = osp.join(
            ds.data_root,
            "dataset",
            "sequences",
            sequence,
            "image_2",
            frame_id + ".png",
        )

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        post_rot = np.eye(3, dtype=np.float32)
        post_tran = np.zeros(3, dtype=np.float32)

        cam_K = scan["P"][:3, :3]

        projected_pix, fov_mask, pix_z = vox2pix(
            scan["T_velo_2_cam"],
            cam_K,
            ds.voxel_origin,
            ds.voxel_size * 1,   # scale_3d = 1, matches 256*256*32
            (img_h, img_w),
            ds.scene_size,
            post_rot,
            post_tran,
        )

        mask_2d = make_2d_mask_from_3d_gt(
            target_3d=target_seen_compact,
            projected_pix=projected_pix,
            fov_mask=fov_mask,
            pix_z=pix_z,
            img_h=img_h,
            img_w=img_w,
            new_class_ids=step_seen_compact_ids,
            ignore_index=255,
            occ_mask=None,
        )

        out_dir = osp.join(args.out_root, sequence)
        os.makedirs(out_dir, exist_ok=True)

        out_path = osp.join(out_dir, frame_id + ".png")
        Image.fromarray(mask_2d).save(out_path)

    print("Done.")


if __name__ == "__main__":
    main()