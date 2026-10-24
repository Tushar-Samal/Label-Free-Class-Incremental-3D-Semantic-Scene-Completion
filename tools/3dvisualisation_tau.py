import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rich.progress import track


# ---------- SemanticKITTI COLORS ----------
COLORS = np.array([
    [0, 0, 0],
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


# 🔥 UPDATE THIS PATH
BASE_DIR = "eval_outputs/SemanticKITTI/<run_timestamp>/test/sequences"

TAUS = [0.6, 0.7, 0.8, 0.9, 0.95]


def render_voxels(vol, save_path, title):

    # occupied voxels
    occupied = vol > 0

    colors = COLORS[vol]
    colors = colors / 255.0

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(
        occupied,
        facecolors=colors,
        edgecolor=None
    )

    # nicer 3D view
    ax.view_init(elev=65, azim=45)

    ax.set_axis_off()
    ax.set_title(title)

    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()


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

            conf = data["teacher_conf_old"]
            pred = data["teacher_pred_old"]

            for tau in TAUS:

                vol = pred.copy()
                vol[conf <= tau] = 0

                save_path = osp.join(
                    pred_dir,
                    f"{frame_id}_tau{tau:.2f}_3D.png"
                )

                render_voxels(
                    vol,
                    save_path,
                    f"{frame_id} | tau={tau}"
                )

                coverage = np.mean(conf > tau)
                print(f"{frame_id} tau={tau} coverage={coverage:.4f}")


if __name__ == "__main__":
    main()