import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

# --------- SAME COLORS ----------
COLORS = np.array([
    [0, 0, 0],            # empty (black)
    [100, 150, 245],
    [100, 230, 245],
    [30, 60, 150],
    [80, 30, 180],
    [100, 80, 250],
    [255, 30, 30],
    [255, 40, 200],
    [150, 30, 90],
    [255, 0, 255],
    [255, 150, 255],
    [75, 0, 75],
    [175, 0, 75],
    [255, 200, 0],
    [255, 120, 50],
    [0, 175, 0],
    [135, 60, 0],
    [150, 240, 80],
    [255, 240, 150],
    [255, 0, 0],
], dtype=np.uint8)

# 🔥 CHANGE THIS TO YOUR TIMESTAMP
BASE_DIR = "eval_outputs/SemanticKITTI/<run_timestamp>/test/sequences"

TAUS = [0.6, 0.7, 0.8, 0.9, 0.95]


def voxel_to_bev(vol):
    """
    Convert (L,W,H) voxel grid to BEV by taking highest occupied voxel along height.
    """
    L, W, H = vol.shape
    bev = np.zeros((L, W), dtype=np.uint8)

    for h in range(H - 1, -1, -1):  # top-down
        mask = (vol[:, :, h] > 0) & (bev == 0)
        bev[mask] = vol[:, :, h][mask]

    return bev


def colorize(bev):
    rgb = COLORS[bev]
    return rgb


def main():

    for seq in os.listdir(BASE_DIR):

        pred_dir = osp.join(BASE_DIR, seq, "predictions")
        if not osp.isdir(pred_dir):
            continue

        for file in track(os.listdir(pred_dir), description=f"Sequence {seq}"):

            if not file.endswith(".teacher_old.npz"):
                continue

            frame_id = file.replace(".teacher_old.npz", "")
            data = np.load(osp.join(pred_dir, file))

            conf = data["teacher_conf_old"]   # (L,W,H)
            pred = data["teacher_pred_old"]   # (L,W,H)

            for tau in TAUS:

                vol = pred.copy()
                vol[conf <= tau] = 0   # remove low-confidence

                bev = voxel_to_bev(vol)
                rgb = colorize(bev)

                save_path = osp.join(pred_dir, f"{frame_id}_tau{tau:.2f}.png")

                plt.figure(figsize=(6,6))
                plt.imshow(rgb)
                plt.axis("off")
                plt.title(f"{frame_id} | tau={tau}")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()

                coverage = np.mean(conf > tau)
                print(f"{frame_id} tau={tau} coverage={coverage:.4f}")


if __name__ == "__main__":
    main()