"""Aspect-ratio buckets for dynamic-resolution tokenizer training.

An image is never upscaled and never letterboxed: we pick a bucket it can fill, crop a region
at least as large as the bucket, and downscale that region onto the bucket. Which bucket it
lands in is random, so the same image is seen at several scales and aspect ratios over training.

Everything here is driven by the config passed to `BucketPlan`; there are no hardcoded bucket
lists or budgets.
"""

import math
from typing import NamedTuple, Optional

import numpy as np
import torch
from PIL import Image


class Assignment(NamedTuple):
    """What `assign` decided for one image, carried alongside the still-encoded bytes.

    `src_size` is the size read from the image header. `apply` re-checks it against the decoded
    image: PIL's `crop` pads out-of-bounds boxes with black instead of raising, so a header/decode
    mismatch would silently produce black bars in the training data.
    """

    bucket: tuple  # (H, W)
    crop: tuple  # (x, y, cw, ch) in source pixels
    src_size: tuple  # (w, h) as measured from the header


class BucketPlan:
    """Bucket set + the assignment policy.

    Args:
        bucket_classes: list of classes, each a list of (H, W). A class groups buckets of
            comparable token count; the class is drawn uniformly among those an image can fill,
            so the realized resolution mix follows what the data can actually supply.
        token_budget: per-GPU token budget. A bucket's batch size is `token_budget // tokens`.
        ar_temp: temperature of the within-class aspect-ratio choice. Small values stay close to
            the nearest aspect ratio (little cropping); larger values mix aspect ratios more.
        patch_size, merge_size: the vision tower's patch size and spatial merge size. Every bucket side must be a
            multiple of `patch_size * merge_size`.
    """

    def __init__(self, bucket_classes, token_budget, ar_temp, patch_size=16, merge_size=2):
        if not bucket_classes:
            raise ValueError("bucket_classes is empty")
        if ar_temp <= 0:
            raise ValueError(f"ar_temp must be > 0, got {ar_temp}")

        self.patch_size = patch_size
        self.merge_size = merge_size
        self.token_budget = token_budget
        self.ar_temp = ar_temp

        stride = patch_size * merge_size
        self.buckets = []
        self.class_of = {}
        for ci, group in enumerate(bucket_classes):
            for hw in group:
                b = (int(hw[0]), int(hw[1]))
                if b[0] % stride or b[1] % stride:
                    raise ValueError(f"bucket {b[0]}x{b[1]} is not a multiple of {stride} (patch {patch_size} x merge {merge_size})")
                if b in self.class_of:
                    raise ValueError(f"bucket {b[0]}x{b[1]} appears more than once")
                self.buckets.append(b)
                self.class_of[b] = ci
        self.buckets = tuple(self.buckets)

        self.tokens = {b: (b[0] // patch_size) * (b[1] // patch_size) for b in self.buckets}
        worst = max(self.tokens.values())
        if token_budget < worst:
            raise ValueError(f"token_budget {token_budget} < largest bucket's {worst} tokens; batch size would be 0")
        self.batch_size = {b: token_budget // self.tokens[b] for b in self.buckets}

    def describe(self) -> str:
        lines = [f"BucketPlan: {len(self.buckets)} buckets, token_budget={self.token_budget}, ar_temp={self.ar_temp}"]
        for b in self.buckets:
            gh, gw = b[0] // self.patch_size, b[1] // self.patch_size
            lines.append(
                f"  class {self.class_of[b]}  {b[0]:4d}x{b[1]:<4d}  grid {gh:3d}x{gw:<3d}"
                f"  tokens {self.tokens[b]:5d}  batch {self.batch_size[b]:3d}"
            )
        return "\n".join(lines)

    def assign(self, w: int, h: int, rng) -> Optional[Assignment]:
        """Choose a bucket and a crop box for a `w x h` source image, or None to drop it.

        An image is only eligible for buckets it can cover without upscaling, which is also what
        makes "images smaller than the smallest bucket" drop out — no separate size threshold.
        """
        feasible = [b for b in self.buckets if b[0] <= h and b[1] <= w]
        if not feasible:
            return None

        # Resolution class first, uniformly over what this image can reach.
        classes = sorted({self.class_of[b] for b in feasible})
        cls = rng.choice(classes)

        # Then the aspect ratio within that class, favouring the closest match.
        cand = [b for b in feasible if self.class_of[b] == cls]
        ar = w / h
        weights = [math.exp(-abs(math.log((b[1] / b[0]) / ar)) / self.ar_temp) for b in cand]
        H, W = rng.choices(cand, weights=weights)[0]

        # Zoom jitter: the crop is anywhere between the bucket itself (native pixels, most
        # zoomed in) and the largest box that still fits the source (most zoomed out). Sampling
        # uniformly in area rather than in scale matches RandomResizedCrop's convention.
        s_min = max(W / w, H / h)  # <= 1 because the bucket fits
        area = rng.uniform(W * H, W * H / (s_min * s_min))
        s = math.sqrt(W * H / area)  # in [s_min, 1]
        cw = min(max(round(W / s), W), w)  # >= bucket: never upscale.  <= source: never overrun
        ch = min(max(round(H / s), H), h)

        x = rng.randint(0, w - cw)
        y = rng.randint(0, h - ch)
        return Assignment(bucket=(H, W), crop=(x, y, cw, ch), src_size=(w, h))

    def apply(self, img: Image.Image, a: Assignment, hflip: bool) -> Optional[torch.Tensor]:
        """Crop, resize onto the bucket and return `[3, H, W]` in `[0, 1]`, or None to drop."""
        if img.size != a.src_size:
            # Header said one thing, the decoder another. Cropping with the stale box would pad
            # with black rather than fail, so drop instead.
            return None

        H, W = a.bucket
        x, y, cw, ch = a.crop
        img = img.convert("RGB").crop((x, y, x + cw, y + ch))
        img = img.resize((W, H), Image.BICUBIC)  # PIL takes (width, height)
        if hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        arr = np.asarray(img, dtype=np.uint8).copy()  # copy: asarray on a PIL image is read-only
        return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)


def plan_from_config(cfg: dict, patch_size: int = 16, merge_size: int = 2) -> BucketPlan:
    """Build a `BucketPlan` from the `bucket_config` block of a training config."""
    return BucketPlan(
        bucket_classes=cfg["bucket_classes"],
        token_budget=cfg["token_budget"],
        ar_temp=cfg["ar_temp"],
        patch_size=patch_size,
        merge_size=merge_size,
    )
