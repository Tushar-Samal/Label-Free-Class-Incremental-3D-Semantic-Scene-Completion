import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectedPixOcc2DRenderer(nn.Module):
    """
    Render occupancy logits to image using precomputed projected_pix and fov_mask.

    occ_logits:    [B, C, X, Y, Z]
    projected_pix: [B, X*Y*Z, 2]
    fov_mask:      [B, X*Y*Z]
    img_hw:        (H, W)
    pix_z:         [B, X*Y*Z] optional per-voxel depth. When provided the
                   nearest voxel wins at each pixel; otherwise last-write-wins.

    output:
        rendered_logits: [B, 1, C, H, W]
    """

    def __init__(self):
        super().__init__()

    def forward(self, occ_logits, projected_pix, fov_mask, img_hw, pix_z=None):
        B, C, X, Y, Z = occ_logits.shape
        H, W = img_hw

        voxel_logits = occ_logits.permute(0, 2, 3, 4, 1).reshape(B, -1, C)

        projected_pix = projected_pix.long()
        fov_mask = fov_mask.bool()

        rendered = occ_logits.new_zeros((B, 1, C, H, W))

        for b in range(B):
            valid_idx = torch.where(fov_mask[b])[0]

            if valid_idx.numel() == 0:
                continue

            pix = projected_pix[b, valid_idx]
            x = pix[:, 0].long()
            y = pix[:, 1].long()

            in_img = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            if in_img.sum() == 0:
                continue

            valid_idx = valid_idx[in_img]
            x = x[in_img]
            y = y[in_img]
            logits = voxel_logits[b, valid_idx]  # [N, C]

            if pix_z is not None:
                # Z-buffer: sort by depth (nearest first), then group entries by
                # pixel with a stable sort so within each pixel the depth order is
                # preserved — the first entry per pixel is then the nearest voxel.
                # An adjacent-duplicate check directly on the depth-sorted order
                # keeps almost every entry (same-pixel voxels are rarely adjacent
                # after a global depth sort), leaving the winner to a duplicate
                # scatter that favours the farthest voxel.
                depths = pix_z[b, valid_idx]
                sort_order = torch.argsort(depths)

                x_s      = x[sort_order]
                y_s      = y[sort_order]
                logits_s = logits[sort_order]

                pixel_id = y_s * W + x_s
                pixel_id_sorted, group_order = torch.sort(pixel_id, stable=True)

                is_first = torch.ones(pixel_id_sorted.shape[0], dtype=torch.bool, device=pixel_id_sorted.device)
                is_first[1:] = pixel_id_sorted[1:] != pixel_id_sorted[:-1]

                keep = group_order[is_first]
                rendered[b, 0, :, y_s[keep], x_s[keep]] = logits_s[keep].transpose(0, 1)
            else:
                # Fallback: last-write-wins (original behavior).
                rendered[b, 0, :, y, x] = logits.transpose(0, 1)

        return rendered


class NewClassRenderLoss(nn.Module):
    def __init__(
        self,
        new_class_ids,
        ignore_index=255,
        loss_weight=1.0,
        class_weight_overrides=None,
    ):
        super().__init__()
        self.new_class_ids = new_class_ids
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        # {compact_id: weight} — e.g. gives CIL-new classes a higher CE
        # weight than old classes reinforced through the same render loss.
        # Compact ids not present here default to weight 1.0.
        self.class_weight_overrides = class_weight_overrides or {}

    def forward(self, rendered_logits, target_2d, valid_2d=None):
        """
        rendered_logits: [B, 1, C, H, W]
        target_2d:       [B, 1, H, W]
        valid_2d:        [B, 1, H, W]
        """
        B, N_cam, C, H, W = rendered_logits.shape

        logits = rendered_logits.reshape(B * N_cam, C, H, W)
        target = target_2d.reshape(B * N_cam, H, W).long()

        new_class_mask = torch.zeros_like(target, dtype=torch.bool)

        for cls_id in self.new_class_ids:
            new_class_mask |= target == cls_id

        if valid_2d is not None:
            valid = valid_2d.reshape(B * N_cam, H, W).bool()
            new_class_mask &= valid

        target_for_loss = target.clone()
        target_for_loss[~new_class_mask] = self.ignore_index

        if new_class_mask.sum() == 0:
            return logits.sum() * 0.0

        class_weight = None
        if self.class_weight_overrides:
            class_weight = logits.new_ones(C)
            for cls_id, w in self.class_weight_overrides.items():
                if 0 <= cls_id < C:
                    class_weight[cls_id] = w

        return self.loss_weight * F.cross_entropy(
            logits,
            target_for_loss,
            ignore_index=self.ignore_index,
            weight=class_weight,
        )