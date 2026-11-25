import torch

from .diffaug import DiffAug
from .discriminator import DinoDiscriminator
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS


def build_discriminator(
    ckpt_path,
    ks=9,
    norm_type='bn',
    using_spec_norm=True,
    recipe= 'S_8',
    device='cuda',
    aug_prob=1.0,
    aug_cutout=0.0
) -> tuple[DinoDiscriminator, DiffAug]:
    if not ckpt_path:
        raise ValueError("the DINO discriminator needs its weights: set disc_ckpt")
    disc = DinoDiscriminator(
        device=device,
        dino_ckpt_path=ckpt_path,
        ks=ks,
        key_depths=(2, 5, 8, 11),
        norm_type=norm_type,
        using_spec_norm=using_spec_norm,
        norm_eps=1e-6,
        recipe=recipe,
    ).to(device)
    augment = DiffAug(prob=aug_prob, cutout=aug_cutout)
    return disc, augment


__all__ = [
    "LPIPS",
    "DiffAug",
    "DinoDiscriminator",
    "hinge_d_loss",
    "vanilla_d_loss",
    "vanilla_g_loss",
    "build_discriminator",
]
