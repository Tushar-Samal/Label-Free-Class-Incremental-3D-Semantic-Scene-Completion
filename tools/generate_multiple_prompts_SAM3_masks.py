import argparse
import inspect
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


IGNORE_INDEX = 255

# Global SemanticKITTI class name -> global label ID (0=empty, 1..19=classes).
# Kept as a local, self-contained constant (mirrors
# ssc_ia.data.datasets.semantic_kitti.SemanticKITTI.META_INFO['class_names'] and
# tools/eval_2d_mask_iou.py's _GLOBAL_NAMES) so this SAM3 tool has no import
# dependency on the training package.
GLOBAL_ID_BY_NAME = {
    "car": 1, "bicycle": 2, "motorcycle": 3, "truck": 4, "other-vehicle": 5,
    "person": 6, "bicyclist": 7, "motorcyclist": 8, "road": 9, "parking": 10,
    "sidewalk": 11, "other-ground": 12, "building": 13, "fence": 14,
    "vegetation": 15, "trunk": 16, "terrain": 17, "pole": 18, "traffic-sign": 19,
}


def _normalize_class_name(name):
    return name.strip().lower().replace(" ", "-")


def resolve_global_id(prompt_name):
    """Map a canonical SemanticKITTI class name (as used as a key in
    classes_json) to its global SemanticKITTI label ID."""
    key = _normalize_class_name(prompt_name)
    if key not in GLOBAL_ID_BY_NAME:
        raise KeyError(
            f"Class name '{prompt_name}' (normalized '{key}') has no known "
            f"global SemanticKITTI ID. Add it to GLOBAL_ID_BY_NAME."
        )
    return GLOBAL_ID_BY_NAME[key]


def build_global_to_compact_lut(class_to_global, class_to_compact, ignore_index, lut_size=256):
    """LUT mapping global label ID -> compact label ID for this run's classes.
    Any global ID not produced by this run's prompts (including ignore_index
    itself) maps to ignore_index."""
    lut = np.full(lut_size, ignore_index, dtype=np.uint8)
    for name, compact_id in class_to_compact.items():
        lut[class_to_global[name]] = np.uint8(compact_id)
    return lut


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--classes_json", type=str, required=True)
    parser.add_argument("--sequences", nargs="+", required=True)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--max_images", type=int, default=-1)

    parser.add_argument("--num_shards", type=int, default=1,
                         help="Split all frames across N parallel processes (one per GPU).")
    parser.add_argument("--shard_id", type=int, default=0,
                         help="This process's shard index in [0, num_shards).")

    args = parser.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"shard_id={args.shard_id} must be in [0, num_shards={args.num_shards})")

    return args


def load_classes(classes_json):
    """Parses the multi-prompt classes_json schema:
        {class_name: {"compact_id": int, "prompts": [str, ...]}}
    Returns (class_to_compact, class_to_prompts), both keyed by canonical
    class name (same semantics as the single-prompt tool's class_to_compact).
    """
    with open(classes_json, "r") as f:
        raw = json.load(f)

    class_to_compact = {name: int(entry["compact_id"]) for name, entry in raw.items()}
    class_to_prompts = {name: list(entry["prompts"]) for name, entry in raw.items()}

    return class_to_compact, class_to_prompts


def build_model(args):
    if args.device == "cpu":
        print("[INFO] CPU mode requested.")
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"[INFO] Loading SAM3 on device={args.device}")

    if args.device.startswith("cuda"):
        cuda_index = torch.device(args.device).index
        if cuda_index is not None:
            # build_sam3_image_model()'s internal _setup_device_and_mode() only
            # special-cases the literal string "cuda" (not "cuda:N") and calls
            # model.cuda() with no index, which always targets the process's
            # *current* CUDA device. Without this, every shard process defaults
            # to GPU 0 regardless of --device, stacking all shards on one GPU.
            torch.cuda.set_device(cuda_index)

    sig = inspect.signature(build_sam3_image_model)
    kwargs = {}

    if args.checkpoint is not None:
        if "checkpoint_path" in sig.parameters:
            kwargs["checkpoint_path"] = args.checkpoint
        elif "ckpt_path" in sig.parameters:
            kwargs["ckpt_path"] = args.checkpoint
        elif "checkpoint" in sig.parameters:
            kwargs["checkpoint"] = args.checkpoint

    # Important:
    # Do NOT pass device into build_sam3_image_model here.
    # We move the whole model manually after loading.
    print("[INFO] build_sam3_image_model kwargs:", kwargs)

    model = build_sam3_image_model(**kwargs)

    # Move model to GPU, but do NOT force dtype=torch.float32.
    model = model.to(args.device)
    model.eval()

    proc_sig = inspect.signature(Sam3Processor)
    if "device" in proc_sig.parameters:
        processor = Sam3Processor(model, device=args.device)
    else:
        processor = Sam3Processor(model)

    return processor


@torch.no_grad()
def predict_prompt(processor, image_pil, prompt, score_threshold, device, debug=False):
    """
    Returns list of (mask_bool, score) pairs for one text prompt.
    Only pairs whose score >= score_threshold are returned.
    """

    with torch.inference_mode():
        state = processor.set_image(image_pil)

        output = processor.set_text_prompt(
            state=state,
            prompt=prompt,
        )

    if "masks" not in output:
        if debug:
            print(f"[DEBUG] prompt='{prompt}' output has no masks. keys={list(output.keys())}")
        return []

    masks = output["masks"]
    scores = output.get("scores", None)

    if debug:
        print(f"\n[DEBUG] prompt='{prompt}'")
        print("[DEBUG] output keys:", list(output.keys()))

        if torch.is_tensor(masks):
            print("[DEBUG] masks tensor shape:", tuple(masks.shape))
            print("[DEBUG] masks numel:", masks.numel())

            if masks.numel() > 0:
                print("[DEBUG] masks min/max:", masks.min().item(), masks.max().item())
            else:
                print("[DEBUG] masks is empty")
        else:
            print("[DEBUG] masks type:", type(masks), "len:", len(masks))

        if scores is not None:
            if torch.is_tensor(scores):
                print("[DEBUG] scores:", scores.detach().cpu().numpy())
            else:
                print("[DEBUG] scores:", scores)
        else:
            print("[DEBUG] scores: None")

    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()

    if scores is not None and torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    if len(masks) == 0:
        if debug:
            print(f"[DEBUG] no masks returned for prompt='{prompt}'")
        return []

    final_masks = []

    for i in range(len(masks)):
        score = 1.0 if scores is None else float(scores[i])

        if score < score_threshold:
            continue

        mask = masks[i]

        if mask.ndim == 3:
            mask = mask[0]

        mask_bool = mask > 0.5

        if mask_bool.sum() == 0:
            continue

        final_masks.append((mask_bool, score))

    if debug:
        print(f"[DEBUG] kept masks for prompt='{prompt}': {len(final_masks)}")

    return final_masks


def aggregate_class_score(class_mask, class_score, candidate_mask, candidate_score):
    """Folds one synonym-prompt detection into the running per-class,
    per-pixel best-score aggregate (see SAM3_multiprompt_vocabulary_design.md
    §3): max score across a class's synonym prompts wins each pixel, so the
    class contributes exactly one (mask, score) pair to the unchanged
    cross-class winner-take-all step below, regardless of how many synonyms
    it has.
    """
    better = candidate_mask & (candidate_score > class_score)
    class_mask[better] = True
    class_score[better] = candidate_score
    return class_mask, class_score


def save_debug_overlay(image_rgb, semantic_mask, out_path):
    overlay = image_rgb.copy()

    unique_ids = np.unique(semantic_mask)
    unique_ids = [int(x) for x in unique_ids if int(x) not in [0, IGNORE_INDEX]]

    for cls_id in unique_ids:
        mask = semantic_mask == cls_id

        color = np.array(
            [
                (37 * cls_id) % 255,
                (17 * cls_id) % 255,
                (97 * cls_id) % 255,
            ],
            dtype=np.uint8,
        )

        colored = np.zeros_like(overlay)
        colored[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    class_to_compact, class_to_prompts = load_classes(args.classes_json)
    class_to_global = {name: resolve_global_id(name) for name in class_to_compact}

    print("Classes (name: global_id -> compact_id, prompts):")
    for name, compact_id in class_to_compact.items():
        print(f"  {name}: {class_to_global[name]} -> {compact_id}  prompts={class_to_prompts[name]}")

    global_to_compact_lut = build_global_to_compact_lut(
        class_to_global, class_to_compact, IGNORE_INDEX
    )

    processor = build_model(args)

    # Global frame counter spanning all requested sequences, so --num_shards
    # processes (one per GPU) split the total frame count evenly instead of
    # being assigned whole sequences (which are very uneven in length).
    global_frame_idx = 0

    for seq in args.sequences:
        seq = str(seq).zfill(2)

        image_dir = data_root / "dataset" / "sequences" / seq / "image_2"

        if not image_dir.exists():
            print(f"[WARN] image dir does not exist: {image_dir}")
            continue

        out_seq_dir = out_root / seq
        out_seq_dir.mkdir(parents=True, exist_ok=True)

        debug_seq_dir = out_root / "_debug" / seq

        image_paths = sorted(image_dir.glob("*.png"))

        if args.max_images > 0:
            image_paths = image_paths[: args.max_images]

        print(f"[INFO] sequence {seq}: processing {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc=f"seq {seq} shard {args.shard_id}/{args.num_shards}"):
            shard_of_this_frame = global_frame_idx % args.num_shards
            global_frame_idx += 1

            if shard_of_this_frame != args.shard_id:
                continue

            frame_id = image_path.stem
            out_path = out_seq_dir / f"{frame_id}.png"

            if out_path.exists():
                continue

            image_pil = Image.open(image_path).convert("RGB")
            image_rgb = np.array(image_pil)

            h, w = image_rgb.shape[:2]

            # SAM3 inference and winner-take-all run entirely in GLOBAL label-ID
            # space, so class identities are preserved exactly as SAM3 saw them.
            # 255 = ignore, 1..19 = global SemanticKITTI class IDs.
            # score_map tracks the best SAM3 score that wrote each pixel;
            # a new class only wins a pixel if its score strictly exceeds the
            # score that previously claimed that pixel.
            semantic_mask_global = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
            score_map            = np.full((h, w), -1.0,         dtype=np.float32)

            debug_this_frame = args.save_debug and frame_id == image_paths[0].stem

            for class_name, compact_id in class_to_compact.items():
                global_id = class_to_global[class_name]

                # Per-class scratch: aggregate every synonym prompt's
                # detections into one (mask, score) pair per pixel before
                # this class competes with any other class below. This is
                # the only structural change vs. the single-prompt tool
                # (see SAM3_multiprompt_vocabulary_design.md §3/§4) -- it
                # closes the prompt-count bias a naive per-prompt loop would
                # introduce (a class with 4 synonyms would otherwise get 4
                # independent shots at a pixel vs. a class with 1).
                class_mask  = np.zeros((h, w), dtype=bool)
                class_score = np.full((h, w), -1.0, dtype=np.float32)

                for prompt in class_to_prompts[class_name]:
                    masks = predict_prompt(
                        processor=processor,
                        image_pil=image_pil,
                        prompt=prompt,
                        score_threshold=args.score_threshold,
                        device=args.device,
                        debug=debug_this_frame,
                    )

                    for mask_bool, score in masks:
                        class_mask, class_score = aggregate_class_score(
                            class_mask, class_score, mask_bool, score
                        )

                    if debug_this_frame:
                        print(
                            f"[DEBUG] class='{class_name}' prompt='{prompt}'"
                            f"  class_pixels_so_far={int(class_mask.sum())}"
                        )

                # Existing, unmodified cross-class winner-take-all -- now
                # operating on one (mask, score) pair per class instead of
                # one per prompt.
                candidate = class_mask
                better    = candidate & (class_score > score_map)
                rejected  = candidate & ~better

                semantic_mask_global[better] = global_id
                score_map[better]            = class_score[better]

                if debug_this_frame:
                    print(
                        f"[DEBUG] class='{class_name}' aggregated_score"
                        f"  candidate={int(candidate.sum())}"
                        f"  written={int(better.sum())}"
                        f"  rejected_by_higher_score={int(rejected.sum())}"
                    )

            # Remap global IDs -> compact IDs as the final step, once mask
            # generation is complete.
            semantic_mask_compact = global_to_compact_lut[semantic_mask_global]

            vals, counts = np.unique(semantic_mask_compact, return_counts=True)
            stats = dict(zip(vals.tolist(), counts.tolist()))

            if debug_this_frame:
                print(f"\n[DEBUG] frame {frame_id} mask stats (compact IDs):", stats)

            Image.fromarray(semantic_mask_compact, mode="L").save(out_path)

            if args.save_debug:
                save_debug_overlay(
                    image_rgb=image_rgb,
                    semantic_mask=semantic_mask_compact,
                    out_path=debug_seq_dir / f"{frame_id}.png",
                )


if __name__ == "__main__":
    main()
