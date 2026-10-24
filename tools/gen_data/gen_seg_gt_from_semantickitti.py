import os
import os.path as osp
import argparse
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# 5-step DCIL (2026-09-29): other-ground(12)/other-vehicle(5) dropped entirely.
CONTINUAL_SPLITS = {
    0: [9, 11, 15, 17],
    1: [14, 16, 18],
    2: [1, 6, 19],
    3: [4, 10, 13],
    4: [2, 3, 7, 8],
}


def _unique_preserve_order(xs):
    return list(dict.fromkeys(list(xs)))


def _flatten_splits_preserve_order(splits, start, end):
    out = []
    for i in range(start, end):
        out.extend(list(splits[i]))
    return _unique_preserve_order(out)


# SemanticKITTI raw label ID -> learning label ID used by SemanticKITTI SSC
# 0 = empty/unlabeled, 1..19 = semantic classes
LEARNING_MAP = {
    0: 0, 1: 0,
    10: 1,   # car
    11: 2,   # bicycle
    13: 5,   # other-vehicle
    15: 3,   # motorcycle
    16: 5,   # other-vehicle
    18: 4,   # truck
    20: 5,   # other-vehicle
    30: 6,   # person
    31: 7,   # bicyclist
    32: 8,   # motorcyclist
    40: 9,   # road
    44: 10,  # parking
    48: 11,  # sidewalk
    49: 12,  # other-ground
    50: 13,  # building
    51: 14,  # fence
    52: 0,   # other-structure -> ignored
    60: 9,   # lane-marking -> road
    70: 15,  # vegetation
    71: 16,  # trunk
    72: 17,  # terrain
    80: 18,  # pole
    81: 19,  # traffic-sign
    99: 0,
    252: 1,  # moving-car
    253: 7,  # moving-bicyclist
    254: 6,  # moving-person
    255: 8,  # moving-motorcyclist
    256: 5,  # moving-on-rails
    257: 5,  # moving-bus
    258: 4,  # moving-truck
    259: 5,  # moving-other-vehicle
}


def read_calib(calib_path):
    calib_data = {}

    with open(calib_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue

            key, value = line.strip().split(":", 1)
            calib_data[key] = np.array([float(v) for v in value.split()], dtype=np.float32)

    P2 = calib_data["P2"].reshape(3, 4)

    Tr = np.eye(4, dtype=np.float32)
    Tr[:3, :4] = calib_data["Tr"].reshape(3, 4)

    return P2, Tr


def load_semantic_labels(label_path):
    raw = np.fromfile(label_path, dtype=np.uint32)
    sem_raw = raw & 0xFFFF

    sem_learning = np.zeros_like(sem_raw, dtype=np.uint8)

    for raw_id, train_id in LEARNING_MAP.items():
        sem_learning[sem_raw == raw_id] = train_id

    return sem_learning


def project_lidar_to_image(points_xyz, P2, Tr):
    """
    points_xyz: [N, 3] in Velodyne coordinates

    returns:
        u: [N]
        v: [N]
        depth: [N]
    """
    n = points_xyz.shape[0]

    points_h = np.concatenate(
        [points_xyz, np.ones((n, 1), dtype=np.float32)],
        axis=1,
    )  # [N, 4]

    cam_points = (Tr @ points_h.T).T  # [N, 4]
    depth = cam_points[:, 2]

    img_points = (P2 @ cam_points.T).T  # [N, 3]

    eps = 1e-6
    u = img_points[:, 0] / np.maximum(img_points[:, 2], eps)
    v = img_points[:, 1] / np.maximum(img_points[:, 2], eps)

    return u, v, depth


def make_png_mask(
    u,
    v,
    depth,
    labels,
    img_h,
    img_w,
    keep_class_ids,
    ignore_index=255,
    point_radius=2,
    label_remap=None,
):
    """
    Makes an image-sized semantic mask using an occlusion-correct z-buffer.

    keep_class_ids: global IDs (from LEARNING_MAP) to retain.
    label_remap:    optional dict {global_id: seen_compact_id} applied after
                    filtering so the PNG stores seen-compact IDs.

    The z-buffer runs over ALL in-FOV LiDAR points (any class) so that
    old-class surfaces (road, sidewalk, etc.) correctly occlude new-class
    surfaces behind them.  A pixel is only labeled if the nearest point on
    that ray belongs to keep_class_ids; otherwise it stays ignore_index.
    """
    keep_class_ids = np.array(keep_class_ids, dtype=np.int64)

    # All points inside the image frustum, any class.
    in_fov = (
        (depth > 0.0)
        & (u >= 0)
        & (u < img_w)
        & (v >= 0)
        & (v < img_h)
    )

    mask = np.full((img_h, img_w), ignore_index, dtype=np.uint8)

    if in_fov.sum() == 0:
        return mask

    u_f      = np.round(u[in_fov]).astype(np.int64)
    v_f      = np.round(v[in_fov]).astype(np.int64)
    depth_f  = depth[in_fov]
    labels_f = labels[in_fov].copy()

    # Occlusion-correct z-buffer: nearest point of ANY class wins per pixel.
    pixel_id = v_f * img_w + u_f
    order    = np.argsort(depth_f)
    pid_sort = pixel_id[order]
    lbl_sort = labels_f[order]

    unique_pid, first_idx = np.unique(pid_sort, return_index=True)
    nearest_labels = lbl_sort[first_idx]

    # Keep only pixels where the nearest point is a new-class point.
    is_new     = np.isin(nearest_labels, keep_class_ids)
    chosen_lbl = nearest_labels[is_new].copy()
    chosen_y   = unique_pid[is_new] // img_w
    chosen_x   = unique_pid[is_new] % img_w

    # Remap global IDs to seen-compact IDs before writing.
    if label_remap is not None:
        remapped = np.full_like(chosen_lbl, fill_value=ignore_index, dtype=np.uint8)
        for src, dst in label_remap.items():
            remapped[chosen_lbl == src] = dst
        chosen_lbl = remapped
    else:
        chosen_lbl = chosen_lbl.astype(np.uint8)

    if point_radius <= 0:
        mask[chosen_y, chosen_x] = chosen_lbl
        return mask

    # Draw small disks to make point labels easier to supervise.
    pil_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(pil_mask)

    r = int(point_radius)

    for x, y, lab in zip(chosen_x, chosen_y, chosen_lbl):
        draw.ellipse(
            (int(x) - r, int(y) - r, int(x) + r, int(y) + r),
            fill=int(lab),
        )

    return np.array(pil_mask, dtype=np.uint8)


def save_renderocc_style_bin(out_path, u, v, depth, labels, img_h, img_w, keep_class_ids):
    """
    Saves RenderOcc-like projected point semantic GT:
        [x, y, label] flattened as float32.

    This is mainly for comparison with RenderOcc style.
    Your DISC loader currently uses the PNG mask.
    """
    keep_class_ids = np.array(keep_class_ids, dtype=np.uint8)

    valid = (
        (depth > 0.0)
        & (u >= 0)
        & (u < img_w)
        & (v >= 0)
        & (v < img_h)
        & np.isin(labels, keep_class_ids)
    )

    if valid.sum() == 0:
        arr = np.zeros((0, 3), dtype=np.float32)
    else:
        arr = np.stack(
            [
                u[valid],
                v[valid],
                labels[valid].astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    arr.flatten().tofile(out_path)


def process_one(args_tuple):
    (
        data_root,
        out_mask_root,
        out_bin_root,
        sequence,
        frame_id,
        keep_class_ids,
        ignore_index,
        point_radius,
        label_remap,
    ) = args_tuple

    seq_dir = osp.join(data_root, "dataset", "sequences", sequence)

    img_path = osp.join(seq_dir, "image_2", frame_id + ".png")
    lidar_path = osp.join(seq_dir, "velodyne", frame_id + ".bin")
    label_path = osp.join(seq_dir, "labels", frame_id + ".label")
    calib_path = osp.join(seq_dir, "calib.txt")

    if not (osp.exists(img_path) and osp.exists(lidar_path) and osp.exists(label_path) and osp.exists(calib_path)):
        return False, sequence, frame_id, "missing_file"

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    points_xyz = points[:, :3]

    labels = load_semantic_labels(label_path)

    if labels.shape[0] != points_xyz.shape[0]:
        return False, sequence, frame_id, "num_points_label_mismatch"

    P2, Tr = read_calib(calib_path)

    u, v, depth = project_lidar_to_image(points_xyz, P2, Tr)

    mask = make_png_mask(
        u=u,
        v=v,
        depth=depth,
        labels=labels,
        img_h=img_h,
        img_w=img_w,
        keep_class_ids=keep_class_ids,
        ignore_index=ignore_index,
        point_radius=point_radius,
        label_remap=label_remap,
    )

    out_mask_dir = osp.join(out_mask_root, sequence)
    os.makedirs(out_mask_dir, exist_ok=True)

    out_mask_path = osp.join(out_mask_dir, frame_id + ".png")
    Image.fromarray(mask).save(out_mask_path)

    if out_bin_root is not None:
        out_bin_dir = osp.join(out_bin_root, sequence)
        os.makedirs(out_bin_dir, exist_ok=True)

        out_bin_path = osp.join(out_bin_dir, frame_id + ".bin")
        save_renderocc_style_bin(
            out_path=out_bin_path,
            u=u,
            v=v,
            depth=depth,
            labels=labels,
            img_h=img_h,
            img_w=img_w,
            keep_class_ids=keep_class_ids,
        )

    valid_ratio = float((mask != ignore_index).mean())
    return True, sequence, frame_id, valid_ratio


def collect_frames(data_root, sequences):
    items = []

    for sequence in sequences:
        img_dir = osp.join(data_root, "dataset", "sequences", sequence, "image_2")
        if not osp.exists(img_dir):
            continue

        frame_ids = sorted(
            osp.splitext(f)[0]
            for f in os.listdir(img_dir)
            if f.endswith(".png")
        )

        for frame_id in frame_ids:
            items.append((sequence, frame_id))

    return items


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="datasets/kitti_odometry",
    )
    parser.add_argument(
        "--out-mask-root",
        default="datasets/open_vocab_2d_masks",
    )
    parser.add_argument(
        "--out-bin-root",
        default="datasets/seg_gt_lidarseg_semkitti",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=4,
        choices=[0, 1, 2, 3, 4],
        help="Continual learning step. Masks will be saved under out-mask-root/step{step}.",
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Optional manual sequence override. If not given, sequences are selected from --step.",
    )

    parser.add_argument(
        "--keep-class-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional manual compact class IDs to keep. If not given, class IDs are selected from --step.",
    )

    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=-1)

    args = parser.parse_args()

    step_to_sequences = {
        0: ["01", "10"],
        1: ["03", "04"],
        2: ["05", "09"],
        3: ["06", "07"],
        4: ["00", "02"],
    }

    # keep_class_ids are global SemanticKITTI IDs (what LEARNING_MAP produces).
    # 5-step DCIL (2026-09-29): other-ground(12)/other-vehicle(5) dropped entirely.
    step_to_keep_class_ids = {
        0: [9, 11, 15, 17],
        1: [14, 16, 18],
        2: [1, 6, 19],
        3: [4, 10, 13],
        4: [2, 3, 7, 8],
    }

    sequences = args.sequences
    if sequences is None:
        sequences = step_to_sequences[args.step]

    keep_class_ids = args.keep_class_ids
    if keep_class_ids is None:
        keep_class_ids = step_to_keep_class_ids[args.step]

    # Build global-ID → seen-compact-ID remap so the PNG stores seen-compact IDs.
    seen_global = _flatten_splits_preserve_order(CONTINUAL_SPLITS, 0, args.step + 1)
    global_to_seen_compact = {g: i + 1 for i, g in enumerate(seen_global)}
    label_remap = {g: global_to_seen_compact[g] for g in keep_class_ids}

    out_mask_root_step = osp.join(args.out_mask_root, f"step{args.step}")
    out_bin_root_step = None
    if args.out_bin_root is not None:
        out_bin_root_step = osp.join(args.out_bin_root, f"step{args.step}")

    frames = collect_frames(args.data_root, sequences)

    if args.max_samples > 0:
        frames = frames[: args.max_samples]

    print("Data root:", args.data_root)
    print("Step:", args.step)
    print("Output PNG mask root:", out_mask_root_step)
    print("Output RenderOcc-style bin root:", out_bin_root_step)
    print("Sequences:", sequences)
    print("Number of frames:", len(frames))
    print("Keeping global class IDs:", keep_class_ids)
    print("Seen-compact remap:", label_remap)

    worker_args = [
        (
            args.data_root,
            out_mask_root_step,
            out_bin_root_step,
            sequence,
            frame_id,
            keep_class_ids,
            args.ignore_index,
            args.point_radius,
            label_remap,
        )
        for sequence, frame_id in frames
    ]

    if args.num_workers <= 1:
        results = [process_one(x) for x in tqdm(worker_args)]
    else:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(pool.imap(process_one, worker_args), total=len(worker_args)))

    ok = [r for r in results if r[0]]
    bad = [r for r in results if not r[0]]

    print("Done.")
    print("Successful:", len(ok))
    print("Failed:", len(bad))

    if len(ok) > 0:
        valid_ratios = [r[3] for r in ok if isinstance(r[3], float)]
        if len(valid_ratios) > 0:
            print("Mean valid ratio:", float(np.mean(valid_ratios)))
            print("Max valid ratio:", float(np.max(valid_ratios)))
            print("Min valid ratio:", float(np.min(valid_ratios)))

    if len(bad) > 0:
        print("First failed examples:")
        for r in bad[:10]:
            print(r)


if __name__ == "__main__":
    main()