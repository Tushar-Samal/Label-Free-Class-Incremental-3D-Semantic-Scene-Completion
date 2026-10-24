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
    """Map a SAM3 text-prompt class name (as used in classes_json) to its
    global SemanticKITTI label ID."""
    key = _normalize_class_name(prompt_name)
    if key not in GLOBAL_ID_BY_NAME:
        raise KeyError(
            f"Prompt name '{prompt_name}' (normalized '{key}') has no known "
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
    parser.add_argument("--frames_json", type=str, default=None,
                        help="Replay-buffer JSON (tools/select_replay_buffer.py). If set, only "
                             "the listed (sequence, frame_id) images are processed.")

    return parser.parse_args()


def load_classes(classes_json):
    with open(classes_json, "r") as f:
        class_to_compact = json.load(f)

    return {name: int(idx) for name, idx in class_to_compact.items()}

def build_model(args):
    if args.device == "cpu":
        print("[INFO] CPU mode requested.")
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    print(f"[INFO] Loading SAM3 on device={args.device}")

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
    class_to_compact = load_classes(args.classes_json)
    class_to_global = {name: resolve_global_id(name) for name in class_to_compact}

    print("Classes (name: global_id -> compact_id):")
    for name, compact_id in class_to_compact.items():
        print(f"  {name}: {class_to_global[name]} -> {compact_id}")

    global_to_compact_lut = build_global_to_compact_lut(
        class_to_global, class_to_compact, IGNORE_INDEX
    )

    wanted_frames = None
    if args.frames_json:
        with open(args.frames_json, "r") as f:
            buf = json.load(f)
        wanted_frames = {}
        for fr in buf["frames"]:
            wanted_frames.setdefault(str(fr["sequence"]).zfill(2), set()).add(fr["frame_id"])
        print(f"[INFO] --frames_json: {sum(len(v) for v in wanted_frames.values())} frames "
              f"over sequences {sorted(wanted_frames)}")

    processor = build_model(args)

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

        if wanted_frames is not None:
            image_paths = [p for p in image_paths if p.stem in wanted_frames.get(seq, set())]

        if args.max_images > 0:
            image_paths = image_paths[: args.max_images]

        print(f"[INFO] sequence {seq}: processing {len(image_paths)} images")

        for image_path in tqdm(image_paths, desc=f"seq {seq}"):
            frame_id = image_path.stem

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

            for prompt, compact_id in class_to_compact.items():
                global_id = class_to_global[prompt]

                masks = predict_prompt(
                    processor=processor,
                    image_pil=image_pil,
                    prompt=prompt,
                    score_threshold=args.score_threshold,
                    device=args.device,
                    debug=debug_this_frame,
                )

                for mask_bool, score in masks:
                    candidate   = mask_bool                        # True where SAM3 fires
                    better      = candidate & (score > score_map)  # True where this score wins
                    rejected    = candidate & ~better              # True where another class scored higher

                    semantic_mask_global[better] = global_id
                    score_map[better]            = score

                    if debug_this_frame:
                        print(
                            f"[DEBUG] prompt='{prompt}' score={score:.3f}"
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

            Image.fromarray(semantic_mask_compact, mode="L").save(out_seq_dir / f"{frame_id}.png")

            if args.save_debug:
                save_debug_overlay(
                    image_rgb=image_rgb,
                    semantic_mask=semantic_mask_compact,
                    out_path=debug_seq_dir / f"{frame_id}.png",
                )


if __name__ == "__main__":
    main()