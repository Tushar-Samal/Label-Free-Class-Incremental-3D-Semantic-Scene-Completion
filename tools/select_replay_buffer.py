"""
Build a fixed-budget replay buffer for continual step N (N >= 1).

A replay frame supervises only the classes introduced at the step it came from, so each
candidate frame is scored on THOSE classes only:
    coverage gain  - voxels of its origin step's classes it adds, weighted 1/sqrt(class
                     frequency), with diminishing returns (sqrt of accumulated voxels)
    hardness       - mean cross-entropy on those same classes under the model that
                     finished step N-1 (rank-normalised to [0, 1])
and frames are picked greedily with score = gain * (1 + lambda * hardness_rank), subject to
    - a per-origin-step quota (budget split by class count per step), and
    - a spacing rule: no two frames of one sequence closer than --spacing metres
      (ego positions from poses.txt).

Candidates for the newest old step (N-1) are all of its frames. Candidates for earlier steps
come ONLY from the previous buffer (--prev-buffer): frames not kept before are gone, so step N
shrinks the stored frames instead of re-sampling them.

Example:
  python tools/select_replay_buffer.py --for-step 1 \
      --ckpt outputs/DISC/DISC-v1-sema/e13Step0.ckpt \
      --out datasets/Fixed/replay_buffers/step1_classaware_49.json
  python tools/select_replay_buffer.py --for-step 2 \
      --ckpt <step-1 checkpoint> \
      --prev-buffer datasets/Fixed/replay_buffers/step1_classaware_49.json \
      --out datasets/Fixed/replay_buffers/step2_classaware_49.json
"""
import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def quotas_by_class_count(splits, for_step, budget):
    """Largest-remainder split of the budget over steps 0..for_step-1, proportional to class count."""
    counts = np.array([len(splits[k]) for k in range(for_step)], dtype=float)
    raw = budget * counts / counts.sum()
    q = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - q))[: budget - q.sum()]:
        q[k] += 1
    return {k: int(q[k]) for k in range(for_step)}


def load_poses(data_root, seq):
    p = np.loadtxt(osp.join(data_root, "dataset", "sequences", seq, "poses.txt")).reshape(-1, 3, 4)
    return p[:, :3, 3]


def per_class_ce(logits, target, n_old, ignore=255):
    """Sum and count of the cross-entropy per old class (compact ids 1..n_old) in one frame.

    logits [1, C, X, Y, Z], target [1, X, Y, Z] holding 1..n_old or `ignore`."""
    valid = target != ignore
    if not bool(valid.any()):
        return [0.0] * n_old, [0] * n_old
    ce = F.cross_entropy(logits.float(), target, ignore_index=ignore, reduction="none")
    t = target[valid]
    s = torch.bincount(t, weights=ce[valid], minlength=n_old + 1)[1: n_old + 1]
    c = torch.bincount(t, minlength=n_old + 1)[1: n_old + 1]
    return s.tolist(), c.tolist()


def select_frames(H, ce_sum, ce_cnt, pool, seq_of, pos, origin_classes, quota, weights, budget, spacing, lam):
    """Greedy fixed-budget selection.

    H, ce_sum, ce_cnt : [n_candidates, n_old] GT voxels / CE sum / CE voxel count per old class
    pool              : [n_candidates] origin step of each candidate
    origin_classes    : {step: bool[n_old]} the classes a frame from that step supervises; only these
                        count for its coverage and its hardness
    quota             : {step: number of frames to keep from that step}
    """
    cover = np.stack([origin_classes[int(k)] for k in pool]).astype(float)
    Hc = H * cover
    hard = (ce_sum * cover).sum(1) / np.maximum((ce_cnt * cover).sum(1), 1)
    rank = np.argsort(np.argsort(hard)) / max(len(hard) - 1, 1)

    acc = np.zeros(H.shape[1])
    feasible = np.ones(len(pool), dtype=bool)
    filled = {k: 0 for k in quota}
    selected = []
    for _ in range(budget):
        open_pool = np.isin(pool, [k for k in quota if filled[k] < quota[k]])
        mask = feasible & open_pool
        if not mask.any():
            break
        gain = (weights * (np.sqrt(acc + Hc) - np.sqrt(acc))).sum(1)
        score = np.where(mask, gain * (1.0 + lam * rank), -np.inf)
        i = int(np.argmax(score))
        selected.append((i, float(gain[i]), float(score[i])))
        acc += Hc[i]
        filled[int(pool[i])] += 1
        feasible[i] = False
        feasible[(seq_of == seq_of[i]) & (np.linalg.norm(pos - pos[i], axis=1) < spacing)] = False
    return selected, filled, Hc, hard, rank, acc


def main():
    from ssc_ia.data.build import build_data_loaders
    from ssc_ia.data.datasets.semantic_kitti import DISJOINT_SEQ
    from dump_teacher_pseudo import build_teacher, last_if_list, move_to_cuda

    ap = argparse.ArgumentParser()
    ap.add_argument("--for-step", type=int, required=True, help="training step that will use this buffer")
    ap.add_argument("--ckpt", required=True, help="checkpoint of the model that finished step for_step-1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev-buffer", default=None, help="buffer JSON used at step for_step-1 (required if for_step >= 2)")
    ap.add_argument("--budget", type=int, default=49)
    ap.add_argument("--spacing", type=float, default=25.0, help="min metres between frames of one sequence")
    ap.add_argument("--lam", type=float, default=1.0, help="hardness boost: score = gain * (1 + lam * hardness_rank)")
    ap.add_argument("--cache", default=None, help="per-frame scores cache (skips forward passes when re-running)")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    N = args.for_step
    assert N >= 1, "a replay buffer only exists from step 1 on"
    if N >= 2:
        assert args.prev_buffer, f"--prev-buffer (the step-{N - 1} buffer) is required for step {N}"
        with open(args.prev_buffer) as f:
            prev = json.load(f)
        assert prev["for_step"] == N - 1, f"--prev-buffer is for step {prev['for_step']}, expected {N - 1}"
    else:
        prev = None

    with initialize_config_dir(config_dir=osp.join(REPO_ROOT, "configs"), version_base=None):
        cfg = compose(config_name="config_sema")
    OmegaConf.set_struct(cfg, False)
    ds_cfg = cfg.data.datasets
    ds_cfg.continual_step = N  # meta.old_global = classes seen by the step N-1 model
    ds_cfg.splits = ["train"]
    ds_cfg.replay.enable = False
    ds_cfg.render2d.enable = False
    ds_cfg.render2d.replay_via_render_steps = []
    cfg.render2d.enable = False
    cfg.pseudo.enable = False
    cfg.teacher.ckpt_path = args.ckpt

    splits = [list(ds_cfg.continual_splits[k]) for k in sorted(ds_cfg.continual_splits, key=int)]
    quota = quotas_by_class_count(splits, N, args.budget)

    dls, meta = build_data_loaders(cfg.data)
    ds = dls[0].dataset
    old_global = list(meta["old_global_classes"])
    origin_classes = {k: np.array([g in splits[k] for g in old_global]) for k in range(N)}

    # ---- candidate pools ----
    candidates = []  # (sequence, frame_id, origin_step)
    for fr in ds._build_scans_for_sequences(DISJOINT_SEQ[N - 1], True):
        candidates.append((fr["sequence"], fr["frame_id"], N - 1))
    if prev is not None:
        for fr in prev["frames"]:
            if fr["origin_step"] < N - 1:
                candidates.append((fr["sequence"], fr["frame_id"], fr["origin_step"]))
    key = lambda s, f: f"{s}/{f}"  # noqa: E731

    # ---- per-frame old-class histogram + per-class loss (model forward) ----
    cache = {}
    if args.cache and osp.exists(args.cache):
        with open(args.cache) as f:
            c = json.load(f)
        if c.get("ckpt") == args.ckpt and c.get("for_step") == N:
            cache = {k: v for k, v in c["frames"].items() if "ce_sum" in v}  # older caches lack per-class loss
            print(f"[select] loaded {len(cache)} cached frame scores from {args.cache}")

    todo = [(s, f) for s, f, _ in candidates if key(s, f) not in cache]
    if todo:
        wanted = {}
        for s, f in todo:
            wanted.setdefault(s, set()).add(f)
        scans = []
        for s, ids in wanted.items():
            scans += [sc for sc in ds._build_scans_for_sequences([s], True) if sc["frame_id"] in ids]
        ds.scans = scans

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        teacher = build_teacher(cfg, meta, device)
        lut = torch.full((256,), 255, dtype=torch.long, device=device)
        lut[torch.tensor(old_global, device=device)] = torch.arange(1, len(old_global) + 1, device=device)

        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"scoring {len(scans)} frames"):
                gt = y["target_global"].long().to(device)
                logits = last_if_list(teacher(move_to_cuda(x, device))["ssc_logits"])
                target = lut[gt.clamp(0, 255)]
                s, c = per_class_ce(logits, target, len(old_global))
                hist = torch.bincount(gt[target != 255].flatten(), minlength=256)[old_global]
                cache[key(x["sequence"][0], x["frame_id"][0])] = {
                    "hardness": sum(s) / max(sum(c), 1),  # mean CE over all old classes
                    "hist": hist.tolist(), "ce_sum": s, "ce_cnt": c,
                }

        if args.cache:
            os.makedirs(osp.dirname(osp.abspath(args.cache)), exist_ok=True)
            with open(args.cache, "w") as f:
                json.dump({"ckpt": args.ckpt, "for_step": N, "frames": cache}, f)

    # ---- greedy selection ----
    H = np.array([cache[key(s, f)]["hist"] for s, f, _ in candidates], dtype=float)
    ce_sum = np.array([cache[key(s, f)]["ce_sum"] for s, f, _ in candidates], dtype=float)
    ce_cnt = np.array([cache[key(s, f)]["ce_cnt"] for s, f, _ in candidates], dtype=float)
    pool = np.array([o for _, _, o in candidates])
    seq_of = np.array([s for s, _, _ in candidates])
    pos = np.zeros((len(candidates), 3))
    for s in set(seq_of):
        xyz = load_poses(ds.data_root, s)
        for i in np.where(seq_of == s)[0]:
            pos[i] = xyz[int(candidates[i][1])]

    freq = np.asarray(meta["class_frequencies"], dtype=float)[old_global]
    w = 1.0 / np.sqrt(freq / freq.sum())
    selected, filled, Hc, hard, rank, acc = select_frames(
        H, ce_sum, ce_cnt, pool, seq_of, pos, origin_classes, quota, w, args.budget, args.spacing, args.lam)

    for k in quota:
        if filled[k] < quota[k]:
            print(f"[select] WARNING: step-{k} pool exhausted, filled {filled[k]}/{quota[k]} "
                  f"(too few candidates at {args.spacing} m spacing)")

    # ---- write buffer ----
    frames = [{
        "sequence": candidates[i][0], "frame_id": candidates[i][1], "origin_step": int(pool[i]),
        "hardness": float(hard[i]), "hardness_rank": float(rank[i]), "coverage_gain": g, "score": sc,
    } for i, g, sc in selected]
    names = meta["class_names"]
    share_buf = acc / max(acc.sum(), 1)
    share_pool = Hc.sum(0) / max(Hc.sum(), 1)
    out = {
        "for_step": N, "budget": args.budget, "spacing_m": args.spacing, "lambda_hardness": args.lam,
        "ckpt": args.ckpt, "prev_buffer": args.prev_buffer, "quota": {str(k): v for k, v in quota.items()},
        "old_global_classes": old_global, "scored_on": "origin-step classes only",
        "class_voxel_share": {names[g]: float(share_buf[j]) for j, g in enumerate(old_global)},
        "frames": frames,
    }
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n[select] wrote {len(frames)} frames -> {args.out}")
    print(f"[select] quota {quota}  filled {filled}")
    print("[select] per sequence:", {s: int(sum(1 for i, _, _ in selected if seq_of[i] == s)) for s in sorted(set(seq_of))})
    sel_idx = [i for i, _, _ in selected]
    print(f"[select] mean hardness (on supervised classes): buffer {hard[sel_idx].mean():.4f} vs candidates {hard.mean():.4f}")
    print(f"\n{'class':14s} {'candidates':>10s} {'buffer':>8s}   (supervised voxels only)")
    for j in np.argsort(share_pool):
        print(f"{names[old_global[j]]:14s} {100 * share_pool[j]:9.2f}% {100 * share_buf[j]:7.2f}%")


if __name__ == "__main__":
    main()
