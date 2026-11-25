"""Images -> the patch layout the Integral vision tokenizer reads."""

import torch


def patchify(pixel_values: torch.Tensor, patch_size: int, merge_size: int, temporal_patch_size: int):
    """[B, 3, H, W] -> (patches [B, N, 3*T*p*p] in merge-block order, gh, gw): the image repeated over the
    `temporal_patch_size` frames, each `merge_size x merge_size` block of patches consecutive."""
    B, C, H, W = pixel_values.shape
    p, m, tp = patch_size, merge_size, temporal_patch_size
    gh, gw = H // p, W // p
    x = pixel_values.unsqueeze(1).repeat(1, tp, 1, 1, 1)  # [B, tp, C, H, W]
    x = x.reshape(B, tp, C, gh // m, m, p, gw // m, m, p)
    # -> (B, gh//m, gw//m, m, m, C, tp, p, p): merge-block order
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = x.reshape(B, gh * gw, C * tp * p * p)
    return flat, gh, gw


def patchify_batch(pixel_values: torch.Tensor, config):
    """[B, 3, H, W], already normalized -> (patches [B*N, 3*T*p*p], image_grid_thw [B, 3]) for a vision tokenizer
    config, every image the same size."""
    flat, gh, gw = patchify(pixel_values, config.patch_size, config.spatial_merge_size, config.temporal_patch_size)
    B = flat.shape[0]
    thw = torch.tensor([[1, gh, gw]] * B, device=pixel_values.device)
    return flat.reshape(B * gh * gw, -1), thw
