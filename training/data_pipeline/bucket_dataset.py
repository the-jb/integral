"""Dynamic-resolution training stream for the vision tokenizer.

Reads webdataset shards, assigns every image to an aspect-ratio bucket and emits ready-made same-shape batches.
Sources are declared in the training config by the names in `SOURCES`. The image size is read from the file header
before `decode`, so images that fit no bucket are rejected without decoding them.
"""

import glob
import io
import os
import random

import torch
import webdataset as wds
from PIL import Image

from .bucket_batcher import BucketBatcher
from .buckets import BucketPlan, plan_from_config
from .data_paths import dataset_path

# Training image sources: name -> (dataset directory, subdirectory holding the .tar shards).
SOURCES = {
    "datacomp": (dataset_path("datacomp-1b"), "shards"),
    "blip3o-long": (dataset_path("blip3o-long-caption"), ""),
    "rendered-text": (dataset_path("rendered-text"), ""),
}

# The eval set: ImageNet-1K validation, square centre crop.
IMAGENET_PATH = dataset_path("imagenet-1k")


def _tar_files(name: str) -> list[str]:
    root, shards = SOURCES[name]
    base = os.path.join(root, shards)
    tars = sorted(glob.glob(os.path.join(base, "*.tar")))
    if not tars:
        raise FileNotFoundError(f"No tar files found in {base}")
    return tars


class _WorkerRng:
    """A `random.Random` per process, seeded from (seed, rank, dataloader worker), created again after a fork."""

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = None
        self._pid = None

    def __call__(self) -> random.Random:
        pid = os.getpid()
        if self._rng is None or self._pid != pid:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info is not None else 0
            rank = int(os.environ.get("RANK", "0"))
            self._rng = random.Random(self.seed * 1000003 + rank * 1009 + worker_id)
            self._pid = pid
        return self._rng


def _make_assign(plan: BucketPlan, rng: _WorkerRng):
    """Runs BEFORE decode: reads the header only, and drops images that fit no bucket."""

    def assign(sample):
        raw = sample.get("jpg") or sample.get("png")
        if raw is None:
            return None
        try:
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size  # PIL is lazy: this parses the header, not the pixels
        except Exception:
            return None
        a = plan.assign(w, h, rng())
        if a is None:
            return None
        # Keys starting with "__" pass through the webdataset decoder untouched (they are only
        # utf-8 decoded when they hold bytes, and an Assignment is a tuple).
        sample["__assign__"] = a
        return sample

    return assign


def _make_apply(plan: BucketPlan, rng: _WorkerRng, allow_hflip: bool):
    """Runs AFTER decode: crops and resizes onto the assigned bucket."""

    def apply(sample):
        img = sample.get("jpg") or sample.get("png")
        if img is None:
            return None
        a = sample["__assign__"]
        flip = allow_hflip and rng().random() < 0.5
        pixel_values = plan.apply(img, a, flip)
        if pixel_values is None:
            return None
        return {"pixel_values": pixel_values, "bucket": a.bucket}

    return apply


def build_bucket_train_dataset(cfg: dict, seed: int, patch_size: int = 16, merge_size: int = 2):
    """Build the mixed, bucketed, pre-batched training stream from a `bucket_config` block."""
    plan = plan_from_config(cfg, patch_size=patch_size, merge_size=merge_size)
    rng = _WorkerRng(seed)
    shuffle_buffer = cfg.get("shuffle_buffer", 5000)

    datasets, weights, names = [], [], []
    for spec in cfg["sources"]:
        name = spec["name"]
        if name not in SOURCES:
            raise KeyError(f"unknown source '{name}'; known: {sorted(SOURCES)}")
        if not os.path.exists(SOURCES[name][0]):
            raise FileNotFoundError(f"source '{name}' path not found: {SOURCES[name][0]}")
        tars = _tar_files(name)

        # per source: mirrored text is not a valid target
        allow_hflip = bool(spec.get("hflip", True))
        share = float(spec["share"])

        ds = (
            wds.WebDataset(
                tars,
                resampled=True,
                nodesplitter=wds.split_by_node,
                shardshuffle=False,
                handler=wds.warn_and_continue,
            )
            .map(_make_assign(plan, rng), handler=wds.warn_and_continue)
            .select(lambda s: s is not None)
            .shuffle(shuffle_buffer)  # still encoded bytes here, so the buffer stays cheap
            .decode("pil", handler=wds.warn_and_continue)
            .map(_make_apply(plan, rng, allow_hflip), handler=wds.warn_and_continue)
            .select(lambda s: s is not None)
        )
        datasets.append(ds)
        weights.append(share)
        names.append(name)
        print(f"[bucket] {name}: {len(tars)} shards, share={share:.4g}, hflip={allow_hflip}")

    if len(datasets) == 1:
        stream = datasets[0]
    else:
        # `share` is the source's fraction of the images (normalized), drawn after each source's own filtering
        total = sum(weights)
        probs = [w / total for w in weights]
        for name, p in zip(names, probs):
            print(f"[bucket] {name}: prob={p:.4f}")
        stream = wds.RandomMix(datasets, probs=probs)

    print(plan.describe())
    return BucketBatcher(stream, plan)


def build_imagenet_eval(resolution: int):
    """ImageNet-1K validation, centre-cropped to `resolution`, with `img_labels`."""
    from datasets import load_dataset
    from torchvision import transforms

    from .crop import center_crop_arr

    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, resolution)),
            transforms.ToTensor(),
        ]
    )

    def process_eval(examples):
        examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["image"]]
        examples["img_labels"] = examples["label"]
        return examples

    return load_dataset(IMAGENET_PATH, split="validation").with_transform(process_eval)
