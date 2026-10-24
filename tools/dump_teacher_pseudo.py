import os
import os.path as osp
import copy
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm

from ssc_ia import build_from_configs, models
from ssc_ia.data.build import build_data_loaders


def last_if_list(v):
    if isinstance(v, (list, tuple)):
        return v[-1]
    return v


def move_to_cuda(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_cuda(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_cuda(v, device) for v in obj)
    return obj


def build_teacher(cfg, meta, device):
    teacher_ckpt_path = cfg.teacher.ckpt_path
    assert teacher_ckpt_path is not None, "cfg.teacher.ckpt_path must be set"

    old_global = meta["old_global_classes"]
    assert old_global is not None and len(old_global) > 0, "old_global_classes is empty"

    old_global = list(dict.fromkeys(old_global))   # deduplicate, preserve DCIL order
    teacher_kwargs = dict(meta)
    teacher_kwargs["num_classes"]         = 1 + len(old_global)
    teacher_kwargs["cur_global_classes"]  = old_global
    teacher_kwargs["seen_global_classes"] = old_global

    teacher_model = build_from_configs(
        models,
        copy.deepcopy(cfg.model),
        **teacher_kwargs,
    )

    ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model_state = teacher_model.state_dict()
    cleaned_state = {}

    for orig_k, v in state_dict.items():
        k = orig_k[len("model."):] if orig_k.startswith("model.") else orig_k

        if k not in model_state:
            continue
        if not torch.is_tensor(v):
            continue
        if not torch.is_tensor(model_state[k]):
            continue

        if tuple(v.shape) != tuple(model_state[k].shape):
            if v.ndim == 4:
                matched = False
                candidate_perms = [
                    (0, 1, 2, 3),
                    (1, 2, 3, 0),
                    (0, 2, 3, 1),
                    (2, 3, 1, 0),
                    (1, 0, 2, 3),
                    (3, 0, 1, 2),
                    (3, 2, 0, 1),
                    (2, 3, 0, 1),
                ]
                for perm in candidate_perms:
                    v_perm = v.permute(*perm).contiguous()
                    if tuple(v_perm.shape) == tuple(model_state[k].shape):
                        v = v_perm
                        matched = True
                        break
                if not matched:
                    continue
            else:
                continue

        cleaned_state[k] = v

    with torch.no_grad():
        for k, v in cleaned_state.items():
            if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
                model_state[k].copy_(v)

    teacher_model.load_state_dict(model_state, strict=False)
    teacher_model.to(device)
    teacher_model.eval()

    print("[Pseudo Dump] teacher_ckpt:", teacher_ckpt_path)
    print("[Pseudo Dump] old_global:", old_global)
    print("[Pseudo Dump] teacher_num_classes:", 1 + len(old_global))

    return teacher_model


@torch.no_grad()
def make_old_pseudo_labels(teacher, x, y, old_global, global_to_student, tau, ignore_index):
    """
    Teacher output space:
      0 = background
      1..N_old = old classes in sorted(old_global) order

    Returns:
      pseudo_student : labels in student seen-compact space
      valid_mask     : voxels to supervise
      conf_old       : confidence map
    """
    teacher_out = teacher(x)
    teacher_logits = last_if_list(teacher_out["ssc_logits"])  # [B, 1+N_old, X, Y, Z]

    probs = torch.softmax(teacher_logits, dim=1)
    probs_old = probs[:, 1:, ...]  # drop teacher background

    conf_old, old_idx = probs_old.max(dim=1)  # [B, X, Y, Z]

    # old_global arrives in DCIL insertion order (no sorting);
    # teacher channel k+1 corresponds to old_global[k].
    old_global_ordered = torch.tensor(
        list(old_global),
        device=old_idx.device,
        dtype=torch.long,
    )

    pseudo_global = old_global_ordered[old_idx]  # global old class ids
    valid_mask = conf_old >= tau

    if isinstance(y, dict):
        if "target_seen_compact" in y:
            valid_mask = valid_mask & (y["target_seen_compact"] != ignore_index)
        elif "target_compact" in y:
            valid_mask = valid_mask & (y["target_compact"] != ignore_index)

    pseudo_student = torch.zeros_like(pseudo_global)

    for g in old_global:
        pseudo_student[pseudo_global == g] = global_to_student[g]

    return pseudo_student.long(), valid_mask.bool(), conf_old


@hydra.main(version_base=None, config_path="../configs", config_name="config_sema")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert "teacher" in cfg and cfg.teacher.ckpt_path is not None, "teacher.ckpt_path is required"
    assert "pseudo" in cfg, "pseudo config is required"
    assert getattr(cfg.pseudo, "root", None) is not None, "pseudo.root is required"
    assert getattr(cfg.pseudo, "tau", None) is not None, "pseudo.tau is required"

    # Build dataloaders + meta
    dls, meta = build_data_loaders(cfg.data)
    train_loader = dls[0]

    old_global = meta["old_global_classes"]
    seen_global = meta["seen_global_classes"]
    ignore_index = int(meta.get("ignore_index", 255))
    tau = float(cfg.pseudo.tau)
    pseudo_root = cfg.pseudo.root

    assert old_global is not None and len(old_global) > 0, "No old classes found"
    assert seen_global is not None and len(seen_global) > 0, "No seen classes found"

    global_to_student = {g: i + 1 for i, g in enumerate(seen_global)}

    print("[Pseudo Dump] pseudo_root:", pseudo_root)
    print("[Pseudo Dump] tau:", tau)
    print("[Pseudo Dump] seen_global:", seen_global)
    print("[Pseudo Dump] global_to_student:", global_to_student)

    teacher = build_teacher(cfg, meta, device)

    os.makedirs(pseudo_root, exist_ok=True)

    total_frames = 0
    total_kept = 0
    total_voxels = 0

    for batch in tqdm(train_loader, desc="Dumping pseudo labels"):
        x, y = batch
        x = move_to_cuda(x, device)
        y = move_to_cuda(y, device)

        pseudo_student, valid_mask, conf_old = make_old_pseudo_labels(
            teacher=teacher,
            x=x,
            y=y,
            old_global=old_global,
            global_to_student=global_to_student,
            tau=tau,
            ignore_index=ignore_index,
        )

        batch_size = pseudo_student.shape[0]

        for b in range(batch_size):
            sequence = y["sequence"][b] if "sequence" in y else x["sequence"][b]
            frame_id = y["frame_id"][b] if "frame_id" in y else x["frame_id"][b]

            if torch.is_tensor(sequence):
                sequence = sequence.item()
            if torch.is_tensor(frame_id):
                frame_id = frame_id.item()

            sequence = str(sequence).zfill(2)
            frame_id = str(frame_id).zfill(6)

            save_dir = osp.join(pseudo_root, sequence)
            os.makedirs(save_dir, exist_ok=True)
            save_path = osp.join(save_dir, f"{frame_id}.npz")

            pseudo_np = pseudo_student[b].detach().cpu().numpy().astype(np.int64)
            mask_np = valid_mask[b].detach().cpu().numpy().astype(np.bool_)
            conf_np = conf_old[b].detach().cpu().numpy().astype(np.float32)

            np.savez_compressed(
                save_path,
                pseudo_old_compact=pseudo_np,
                pseudo_old_mask=mask_np,
                conf_old=conf_np,
            )

            total_frames += 1
            total_kept += int(mask_np.sum())
            total_voxels += int(mask_np.size)

    coverage = (total_kept / total_voxels) if total_voxels > 0 else 0.0
    print("[Pseudo Dump] total_frames:", total_frames)
    print("[Pseudo Dump] total_kept_voxels:", total_kept)
    print("[Pseudo Dump] total_voxels:", total_voxels)
    print("[Pseudo Dump] mean_coverage:", coverage)


if __name__ == "__main__":
    main()