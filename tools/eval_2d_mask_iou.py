"""
Per-class IoU between SAM3-generated 2D prediction masks and 2D GT masks.

--gt_id_space compact  (default)
    GT PNGs already store seen-compact IDs as produced by the fixed
    generate_gt_2d_masks.py / gen_seg_gt_from_semantickitti.py.
    No remapping is applied.

--gt_id_space global
    GT PNGs store raw global SemanticKITTI IDs.  They are remapped to
    seen-compact IDs before comparison using GT_REMAP below.

GT_REMAP (global -> seen-compact), preserving DCIL insertion order.
5-step DCIL (2026-09-29): other-ground(12)/other-vehicle(5) dropped entirely, 17
classes rebalanced 4/3/3/3/4 across the new DISJOINT_SEQ sequence pairs. See
the README.
  Step 0: {9:1(road), 11:2(sidewalk), 15:3(vegetation), 17:4(terrain)}
  Step 1: {14:5(fence), 16:6(trunk), 18:7(pole)}
  Step 2: {1:8(car), 6:9(person), 19:10(traffic-sign)}
  Step 3: {4:11(truck), 10:12(parking), 13:13(building)}
  Step 4: {2:14(bicycle), 3:15(motorcycle), 7:16(bicyclist), 8:17(motorcyclist)}
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Global SemanticKITTI ID -> seen-compact ID, preserving DCIL insertion order.
GT_REMAP = {
    0: {9: 1, 11: 2, 15: 3, 17: 4},
    1: {14: 5, 16: 6, 18: 7},
    2: {1: 8, 6: 9, 19: 10},
    3: {4: 11, 10: 12, 13: 13},
    4: {2: 14, 3: 15, 7: 16, 8: 17},
}

# Human-readable names keyed by global SemanticKITTI ID.
# Used as fallback for classes absent from the JSON (e.g. other-vehicle in step 2).
_GLOBAL_NAMES = {
    1: "car",           2: "bicycle",       3: "motorcycle",    4: "truck",
    5: "other-vehicle", 6: "person",        7: "bicyclist",     8: "motorcyclist",
    9: "road",         10: "parking",      11: "sidewalk",     12: "other-ground",
   13: "building",     14: "fence",        15: "vegetation",   16: "trunk",
   17: "terrain",      18: "pole",         19: "traffic-sign",
}

# Per-class RGB colors (index = seen-compact ID; 0 = background).
_CLASS_COLORS = {
    0:  (  0,   0,   0),   # background
    # Step 0 classes (seen-compact 1-4)
    1:  (128,  64, 128),   # road
    2:  ( 70,  70,  70),   # sidewalk
    3:  (107, 142,  35),   # vegetation
    4:  (152, 251, 152),   # terrain
    # Step 1 classes (seen-compact 5-7)
    5:  (220, 220,   0),   # fence
    6:  ( 70, 130, 180),   # trunk
    7:  (220,  20,  60),   # pole
    # Step 2 classes (seen-compact 8-10)
    8:  (  0,   0, 142),   # car
    9:  (  0,   0, 142),   # person
    10: (119,  11,  32),   # traffic-sign
    # Step 3 classes (seen-compact 11-13)
    11: (  0,   0, 230),   # truck
    12: (244,  35, 232),   # parking
    13: (250, 170,  30),   # building
    # Step 4 classes (seen-compact 14-17)
    14: (  0,   0,  70),   # bicycle
    15: (  0,  60, 100),   # motorcycle
    16: (  0,  60, 100),   # bicyclist
    17: (  0,   0, 230),   # motorcyclist
}
_IGNORE_COLOR   = ( 70,  70,  70)
_CORRECT_COLOR  = (  0, 200,   0)
_WRONG_COLOR    = (210,   0,   0)
_SEP_COLOR      = (200, 200, 200)   # 3-pixel white bar between panels


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-class IoU between SAM3 pred and 2D GT masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred_root",           required=True,
                   help="Root dir of prediction masks (pred_root/SEQ/FRAME.png).")
    p.add_argument("--gt_root",             required=True,
                   help="Root dir of GT masks (gt_root/SEQ/FRAME.png).")
    p.add_argument("--step",                required=True, type=int, choices=[0, 1, 2, 3, 4],
                   help="Continual learning step (0-4).")
    p.add_argument("--classes_json",        required=True,
                   help="Path to sam_classes_step{N}.json.")
    p.add_argument("--sequences",           nargs="+", default=None,
                   help="Sequences to evaluate (zero-padded). "
                        "Auto-detected from gt_root if omitted.")
    p.add_argument("--ignore_index",        type=int, default=255,
                   help="Pixel value treated as ignore in both pred and GT.")
    p.add_argument("--gt_id_space",         choices=["compact", "global"],
                   default="compact",
                   help="ID space used by the GT masks. "
                        "'compact': GT already stores compact step IDs 1..N "
                        "(output of generate_gt_2d_masks.py --step N). "
                        "'global': GT stores raw global SemanticKITTI IDs; "
                        "they are remapped to compact IDs before comparison.")
    p.add_argument("--out_dir",             required=True,
                   help="Directory for CSV and (optionally) visuals.")
    p.add_argument("--save_visuals",        action="store_true",
                   help="Save pred|GT|error side-by-side images.")
    p.add_argument("--max_visuals_per_seq", type=int, default=10,
                   help="Max visual frames saved per sequence (0 = all).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classes(path):
    with open(path) as f:
        raw = json.load(f)
    return {name: int(idx) for name, idx in raw.items()}


def build_id_to_name(step, class_to_compact):
    """Compact ID -> human name for every class that can appear in GT or pred."""
    # JSON gives us (name -> compact_id); invert it
    id_to_name = {v: k for k, v in class_to_compact.items()}
    # Fill in GT-only classes (e.g. other-vehicle in step 2, not in step-2 JSON)
    inv_remap = {cpt: gid for gid, cpt in GT_REMAP[step].items()}
    for cpt in sorted(GT_REMAP[step].values()):
        if cpt not in id_to_name:
            gid = inv_remap.get(cpt)
            id_to_name[cpt] = _GLOBAL_NAMES.get(gid, f"class_{cpt}")
    return id_to_name


def build_remap_lut(remap_dict, ignore_index, lut_size=256):
    """
    Build a uint8 look-up table of length lut_size.
    lut[global_id] = compact_id   for IDs in remap_dict.
    lut[anything_else] = ignore_index.
    """
    lut = np.full(lut_size, ignore_index, dtype=np.uint8)
    for src, dst in remap_dict.items():
        if 0 <= src < lut_size:
            lut[src] = np.uint8(dst)
    return lut


def colorize(mask, ignore_index):
    """Render a uint8 label mask as an H×W×3 RGB image."""
    h, w = mask.shape
    rgb = np.full((h, w, 3), _IGNORE_COLOR, dtype=np.uint8)
    for cid, color in _CLASS_COLORS.items():
        rgb[mask == cid] = color
    return rgb


def make_error_map(pred, gt_remapped, ignore_index):
    """
    Green  : pred == gt_remapped (both valid)
    Red    : pred != gt_remapped (gt is valid, pred is wrong)
    Gray   : gt_remapped == ignore_index
    """
    h, w = pred.shape
    rgb   = np.full((h, w, 3), _IGNORE_COLOR, dtype=np.uint8)
    valid = gt_remapped != ignore_index
    rgb[valid & (pred == gt_remapped)] = _CORRECT_COLOR
    rgb[valid & (pred != gt_remapped)] = _WRONG_COLOR
    return rgb


def hstack_with_sep(*panels, sep_width=3):
    """Horizontally concatenate RGB panels with a light-gray separator."""
    h = panels[0].shape[0]
    sep = np.full((h, sep_width, 3), _SEP_COLOR, dtype=np.uint8)
    parts = []
    for i, panel in enumerate(panels):
        if i > 0:
            parts.append(sep)
        parts.append(panel)
    return np.concatenate(parts, axis=1)


def warn_unexpected_ids(label, path_name, found_ids, expected_ids, ignore_index):
    unexpected = found_ids - expected_ids - {0, ignore_index}
    if unexpected:
        print(
            f"[WARN] {label} {path_name}: "
            f"unexpected pixel IDs {sorted(unexpected)} "
            f"(expected {sorted(expected_ids)} + ignore={ignore_index}). "
            f"Check that the correct step JSON / mask directory is being used."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    pred_root    = Path(args.pred_root)
    gt_root      = Path(args.gt_root)
    out_dir      = Path(args.out_dir)
    step         = args.step
    ignore_index = args.ignore_index
    out_dir.mkdir(parents=True, exist_ok=True)

    remap_dict       = GT_REMAP[step]
    class_to_compact = load_classes(args.classes_json)
    id_to_name       = build_id_to_name(step, class_to_compact)

    # All compact IDs that can appear in GT for this step (superset of JSON).
    all_class_ids     = sorted(set(remap_dict.values()))
    expected_pred_ids = set(class_to_compact.values())
    # In compact mode GT already uses compact IDs; in global mode it uses
    # the raw global SemanticKITTI IDs that the LUT will remap.
    if args.gt_id_space == "compact":
        expected_gt_ids = set(all_class_ids)
    else:
        expected_gt_ids = set(remap_dict.keys())

    remap_lut = build_remap_lut(remap_dict, ignore_index)

    # ------------------------------------------------------------------
    # Startup summary
    # ------------------------------------------------------------------
    print(f"\n[INFO] Step {step} evaluation")
    print(f"[INFO] pred_root   : {pred_root}")
    print(f"[INFO] gt_root     : {gt_root}")
    print(f"[INFO] gt_id_space : {args.gt_id_space}")
    print(f"[INFO] classes     : { {cid: id_to_name[cid] for cid in all_class_ids} }")
    if args.gt_id_space == "global":
        print(f"[INFO] GT remap    : {remap_dict}")
    else:
        print(f"[INFO] GT remap    : none (GT already in compact IDs)")

    if step == 1 and args.gt_id_space == "global":
        print(
            "\n[WARN] Step 1 global mode: GT_REMAP maps global IDs to seen-compact "
            "IDs 7–11.  Make sure the GT PNGs contain raw global SemanticKITTI IDs "
            "(13/14/16/18/19) for step-1 classes; otherwise use --gt_id_space compact.\n"
        )

    # ------------------------------------------------------------------
    # Discover sequences
    # ------------------------------------------------------------------
    if args.sequences:
        sequences = [str(s).zfill(2) for s in args.sequences]
    else:
        sequences = sorted(
            d.name for d in gt_root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
    print(f"[INFO] Sequences : {sequences}\n")

    # ------------------------------------------------------------------
    # Accumulate TP / FP / FN per class
    # ------------------------------------------------------------------
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    total_frames   = 0
    skipped_frames = 0

    # Deduplicate warnings: only print once per unique unexpected-ID frozenset
    warned_pred_keys = set()
    warned_gt_keys   = set()

    vis_root = out_dir / f"step{step}_visuals" if args.save_visuals else None

    for seq in tqdm(sequences, desc="sequences"):
        pred_seq_dir = pred_root / seq
        gt_seq_dir   = gt_root   / seq

        if not gt_seq_dir.exists():
            print(f"[WARN] GT sequence dir missing   : {gt_seq_dir}")
            continue
        if not pred_seq_dir.exists():
            print(f"[WARN] Pred sequence dir missing : {pred_seq_dir}")
            continue

        gt_frames   = {p.stem: p for p in sorted(gt_seq_dir.glob("*.png"))}
        pred_frames = {p.stem: p for p in sorted(pred_seq_dir.glob("*.png"))}

        only_in_gt = set(gt_frames) - set(pred_frames)
        if only_in_gt:
            print(
                f"[WARN] seq={seq}: {len(only_in_gt)} GT frame(s) have no matching "
                f"pred mask (e.g. {sorted(only_in_gt)[0]}). "
                f"Skipped — FN counts will be underestimated."
            )
            skipped_frames += len(only_in_gt)

        matched = sorted(set(gt_frames) & set(pred_frames))
        if not matched:
            print(f"[WARN] seq={seq}: no matching frames found. Skipping.")
            continue

        vis_seq_dir   = (vis_root / seq) if vis_root else None
        visuals_saved = 0
        max_vis       = args.max_visuals_per_seq  # 0 means unlimited

        for frame_id in tqdm(matched, desc=f"  seq {seq}", leave=False):
            pred_path = pred_frames[frame_id]
            gt_path   = gt_frames[frame_id]

            pred_mask = np.array(Image.open(pred_path))
            gt_raw    = np.array(Image.open(gt_path))

            # --- warn once per unique set of unexpected IDs ---
            pred_unique = set(pred_mask.flat)
            gt_unique   = set(gt_raw.flat)

            pred_warn_key = frozenset(pred_unique - expected_pred_ids - {0, ignore_index})
            gt_warn_key   = frozenset(gt_unique   - expected_gt_ids   - {0, ignore_index})

            if pred_warn_key and pred_warn_key not in warned_pred_keys:
                warned_pred_keys.add(pred_warn_key)
                warn_unexpected_ids(
                    "pred", pred_path.name, pred_unique, expected_pred_ids, ignore_index
                )
            if gt_warn_key and gt_warn_key not in warned_gt_keys:
                warned_gt_keys.add(gt_warn_key)
                warn_unexpected_ids(
                    "GT",   gt_path.name,   gt_unique,   expected_gt_ids,   ignore_index
                )

            # --- resolve GT ID space ---
            if args.gt_id_space == "compact":
                gt_remapped = gt_raw          # already compact; use as-is
            else:
                # Remap global SemanticKITTI IDs -> compact IDs.
                # Any global ID absent from remap_dict becomes ignore_index.
                gt_remapped = remap_lut[gt_raw]

            # --- accumulate TP / FP / FN ---
            # valid: GT pixel carries a known class for this step (not ignore).
            # FP is only counted over valid GT pixels so that pred pixels landing
            # in ignored GT regions are not penalised.
            valid = gt_remapped != ignore_index

            for cid in all_class_ids:
                pred_c = pred_mask == cid
                gt_c   = gt_remapped == cid
                tp[cid] += int((pred_c  &  gt_c).sum())
                fp[cid] += int((pred_c  & ~gt_c & valid).sum())
                fn[cid] += int((~pred_c &  gt_c).sum())

            total_frames += 1

            # --- visual comparison ---
            if vis_root and (max_vis == 0 or visuals_saved < max_vis):
                if vis_seq_dir:
                    vis_seq_dir.mkdir(parents=True, exist_ok=True)
                pred_rgb = colorize(pred_mask,   ignore_index)
                gt_rgb   = colorize(gt_remapped, ignore_index)
                err_rgb  = make_error_map(pred_mask, gt_remapped, ignore_index)
                combined = hstack_with_sep(pred_rgb, gt_rgb, err_rgb)
                Image.fromarray(combined).save(vis_seq_dir / f"{frame_id}.png")
                visuals_saved += 1

    # ------------------------------------------------------------------
    # Compute IoU
    # ------------------------------------------------------------------
    ious      = {}
    miou_vals = []

    for cid in all_class_ids:
        denom  = tp[cid] + fp[cid] + fn[cid]
        iou    = tp[cid] / denom if denom > 0 else float("nan")
        ious[cid] = iou
        if denom > 0:
            miou_vals.append(iou)

    miou = float(np.mean(miou_vals)) if miou_vals else float("nan")

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    col_id   = 4
    col_name = 20
    col_num  = 12
    col_iou  =  8
    line_w   = col_id + 2 + col_name + 2 + col_num + 2 + col_num + 2 + col_num + 2 + col_iou

    print(f"\n{'=' * line_w}")
    print(f"Step {step} IoU  |  {total_frames} frames evaluated"
          + (f"  ({skipped_frames} GT frames skipped — no pred)" if skipped_frames else ""))
    print(f"{'=' * line_w}")
    print(
        f"{'ID':>{col_id}}  {'Name':<{col_name}}"
        f"  {'TP':>{col_num}}  {'FP':>{col_num}}  {'FN':>{col_num}}"
        f"  {'IoU':>{col_iou}}"
    )
    print("-" * line_w)

    for cid in all_class_ids:
        name    = id_to_name.get(cid, f"class_{cid}")
        iou_str = f"{ious[cid]:.4f}" if not np.isnan(ious[cid]) else "     nan"
        print(
            f"{cid:>{col_id}}  {name:<{col_name}}"
            f"  {tp[cid]:>{col_num}}  {fp[cid]:>{col_num}}  {fn[cid]:>{col_num}}"
            f"  {iou_str:>{col_iou}}"
        )

    print("-" * line_w)
    miou_str = f"{miou:.4f}" if not np.isnan(miou) else "nan"
    print(f"{'':>{col_id}}  {'mIoU':<{col_name}}  {'':>{col_num}}  {'':>{col_num}}  {'':>{col_num}}  {miou_str:>{col_iou}}")
    print("=" * line_w)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = out_dir / f"step{step}_iou.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "intersection", "union", "iou"])
        for cid in all_class_ids:
            union   = tp[cid] + fp[cid] + fn[cid]
            iou_val = f"{ious[cid]:.6f}" if not np.isnan(ious[cid]) else "nan"
            writer.writerow([cid, id_to_name.get(cid, f"class_{cid}"),
                             tp[cid], union, iou_val])
        miou_val = f"{miou:.6f}" if not np.isnan(miou) else "nan"
        writer.writerow(["miou", "", "", "", miou_val])

    print(f"\n[INFO] CSV saved : {csv_path}")
    if vis_root:
        print(f"[INFO] Visuals   : {vis_root}")


if __name__ == "__main__":
    main()
