"""Per-sequence, per-class 2D statistics for SemanticKITTI, for CIL split design.

For every sequence (00..10) and every semantic class this script computes:

  1. Image counts  — number of images in which the class occupies >= 1 pixel
                     (a class counts at most once per image).
  2. Pixel counts  — total number of pixels of the class summed over all
                     images of the sequence.

Labels
------
The training pipeline (``ssc_ia/data/datasets/semantic_kitti.py``) reads the
preprocessed voxel GT ``<label_root>/<seq>/<frame>_1_1.npy``, which is already
in the *global training class space*:

    0 = empty, 1..19 = semantic classes (car .. traffic-sign), 255 = ignore

No raw SemanticKITTI IDs are involved anywhere in the pipeline, so the stats
below are reported in this global 0..19 space (the compact/CIL IDs used during
training are derived per-step from these global IDs).

2D projection
-------------
SemanticKITTI has no native 2D semantic GT. Pixels are obtained exactly the
way the pipeline builds its 2D render-supervision GT: the full-resolution 3D
voxel labels are projected into the left camera with ``vox2pix`` and rasterised
with the occlusion-correct z-buffer of
``tools/generate_gt_2d_masks.make_2d_mask_from_3d_gt`` (``occ_mask`` path),
here run with *all* 19 semantic classes as targets. Consequences:

  * ignore voxels (255) and empty voxels (0) never label a pixel — the same
    exclusion the training losses apply;
  * ``class_0`` (empty) columns are therefore always 0 and are kept only so
    that column index == global class ID.

The camera projection depends only on the per-sequence calibration and image
size, so it is computed once per sequence and reused for every frame.

Usage
-----
    export PYTHONPATH=`pwd`:$PYTHONPATH
    python tools/compute_class_stats.py [--out-dir class_stats] [--sequences 00 01 ...]

Outputs ``image_counts_per_sequence.csv`` and ``pixel_counts_per_sequence.csv``
in ``--out-dir`` (rows = sequences, columns = class_0..class_19) plus a printed
summary. Training/testing/dataset behaviour is not modified.
"""

import argparse
import csv
import glob
import os
import os.path as osp
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from ssc_ia.data.datasets.semantic_kitti import SemanticKITTI
from ssc_ia.utils.helper import vox2pix

# tools/ is not a package; import the sibling module by path.
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from generate_gt_2d_masks import make_2d_mask_from_3d_gt  # noqa: E402

# All sequences with ground truth (train 00-07/09/10 + val 08).
DEFAULT_SEQUENCES = ('00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10')

CLASS_NAMES = SemanticKITTI.META_INFO['class_names']  # index == global class ID
NUM_CLASSES = len(CLASS_NAMES)                        # 20 (incl. empty)
IGNORE_INDEX = 255
EMPTY_LABEL = 0

# Semantic (non-empty) classes 1..19 — the classes that can own a pixel.
SEMANTIC_CLASS_IDS = np.arange(1, NUM_CLASSES, dtype=np.int64)


def compute_sequence_projection(sequence_path, img_shape):
    """Project all voxel centroids of the scene grid into the camera.

    The result depends only on the sequence calibration and the image size,
    both constant within a KITTI odometry sequence, so this runs once per
    sequence. Identity post_rot/post_tran = no image augmentation, matching
    how tools/generate_gt_2d_masks.py projects GT at native resolution.

    Returns (projected_pix [N,2], fov_mask [N], pix_z [N]) with N = X*Y*Z
    voxels of the full-scale 256x256x32 grid.
    """
    calib = SemanticKITTI.read_calib(osp.join(sequence_path, 'calib.txt'))
    cam_K = calib['P2'][:3, :3]

    # Full-scale grid parameters, identical to SemanticKITTI.__init__ /
    # generate_gt_2d_masks.py (scale_3d = 1 -> voxel_size 0.2 m).
    voxel_origin = np.array((0, -25.6, -2))
    voxel_size = 0.2
    scene_size = (51.2, 51.2, 6.4)

    return vox2pix(
        calib['Tr'], cam_K, voxel_origin, voxel_size, img_shape, scene_size,
        np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
    )


def get_image_shape(sequence_path):
    """(H, W) of the left camera images, read lazily from the first frame."""
    img_paths = sorted(glob.glob(osp.join(sequence_path, 'image_2', '*.png')))
    if not img_paths:
        raise FileNotFoundError(f'No images under {sequence_path}/image_2')
    with Image.open(img_paths[0]) as img:
        return img.height, img.width


def process_sequence(sequence, data_root, label_root):
    """Accumulate per-class image and pixel counts over one sequence.

    Returns (image_counts [20], pixel_counts [20], n_frames).
    """
    sequence_path = osp.join(data_root, 'dataset', 'sequences', sequence)
    img_h, img_w = get_image_shape(sequence_path)
    projected_pix, fov_mask, pix_z = compute_sequence_projection(sequence_path, (img_h, img_w))

    # Same frames the training pipeline uses: those with a full-scale target.
    label_paths = sorted(glob.glob(osp.join(label_root, sequence, '*_1_1.npy')))

    image_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for label_path in tqdm(label_paths, desc=f'seq {sequence}', unit='img'):
        target_global = np.load(label_path)

        # Occlusion-correct z-buffer over all occupied voxels: a pixel gets the
        # class of the nearest non-empty, non-ignore voxel along its ray.
        occ_mask = (target_global != EMPTY_LABEL) & (target_global != IGNORE_INDEX)
        mask_2d = make_2d_mask_from_3d_gt(
            target_3d=target_global.astype(np.uint8),
            projected_pix=projected_pix,
            fov_mask=fov_mask,
            pix_z=pix_z,
            img_h=img_h,
            img_w=img_w,
            new_class_ids=SEMANTIC_CLASS_IDS,
            ignore_index=IGNORE_INDEX,
            occ_mask=occ_mask.reshape(-1),
        )

        per_frame = np.bincount(
            mask_2d[mask_2d != IGNORE_INDEX].ravel(), minlength=NUM_CLASSES
        )[:NUM_CLASSES]
        pixel_counts += per_frame
        image_counts += (per_frame > 0).astype(np.int64)

    return image_counts, pixel_counts, len(label_paths)


def write_csv(path, rows):
    """rows: list of (sequence_name, counts[20]) in processing order."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sequence'] + [f'class_{c}' for c in range(NUM_CLASSES)])
        for sequence, counts in rows:
            writer.writerow([sequence] + counts.tolist())


def main():
    parser = argparse.ArgumentParser(
        description='Per-sequence, per-class image/pixel statistics for SemanticKITTI.'
    )
    parser.add_argument('--data-root', default='datasets/kitti_odometry',
                        help='KITTI odometry root (contains dataset/sequences/<seq>).')
    parser.add_argument('--label-root', default='datasets/preprocess/labels',
                        help='Preprocessed voxel GT root (contains <seq>/<frame>_1_1.npy).')
    parser.add_argument('--sequences', nargs='+', default=list(DEFAULT_SEQUENCES),
                        help='Sequences to process (default: 00..10).')
    parser.add_argument('--out-dir', default='class_stats',
                        help='Directory for the output CSV files.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image_rows, pixel_rows = [], []
    total_images = 0
    total_image_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for sequence in args.sequences:
        image_counts, pixel_counts, n_frames = process_sequence(
            sequence, args.data_root, args.label_root
        )
        image_rows.append((sequence, image_counts))
        pixel_rows.append((sequence, pixel_counts))
        total_images += n_frames
        total_image_counts += image_counts
        total_pixel_counts += pixel_counts

    image_csv = osp.join(args.out_dir, 'image_counts_per_sequence.csv')
    pixel_csv = osp.join(args.out_dir, 'pixel_counts_per_sequence.csv')
    write_csv(image_csv, image_rows)
    write_csv(pixel_csv, pixel_rows)

    print('\n================ SUMMARY ================')
    print(f'Sequences processed: {len(args.sequences)} ({", ".join(args.sequences)})')
    print(f'Total images processed: {total_images}')
    print(f'\n{"ID":>3}  {"class":<15} {"images":>8} {"pixels":>14}')
    for c in range(NUM_CLASSES):
        print(f'{c:>3}  {CLASS_NAMES[c]:<15} {total_image_counts[c]:>8} {total_pixel_counts[c]:>14}')
    print(f'\nSaved: {image_csv}')
    print(f'Saved: {pixel_csv}')


if __name__ == '__main__':
    main()
