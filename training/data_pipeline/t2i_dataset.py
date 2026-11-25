"""The visual generation eval set, and `drop`, the counter every loader in this package reports rejected samples
through.

`build_t2i_eval_set` reads BLIP-3o webdataset shards (`<key>.jpg` + `<key>.txt`) and returns the first n samples as
`{pixel_values, caption}`.
"""

import glob
import os

import webdataset as wds
from torchvision import transforms

from .data_paths import dataset_path

# BLIP-3o series (WebDataset .tar, each sample = <key>.jpg + <key>.txt caption).
T2I_SOURCES = {
    "blip3o-long": dataset_path("blip3o-long-caption"),
    "blip3o-short": dataset_path("blip3o-short-caption"),
}


def build_t2i_eval_set(sources, n: int, resolution: int = 256) -> list:
    """The first `n` samples of the sorted shards of `sources` (names of `T2I_SOURCES` or directories), resized and
    centre-cropped to `resolution` -> a list of `{pixel_values, caption}`. Every rank builds the same list."""
    tf = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )
    urls = []
    for s in sources:
        d = T2I_SOURCES.get(s, s)
        urls += sorted(glob.glob(os.path.join(d, "*.tar")))

    def preprocess(sample):
        cap = sample["txt"]
        if isinstance(cap, bytes):
            cap = cap.decode("utf-8")
        return {"pixel_values": tf(sample["jpg"].convert("RGB")),
                "caption": cap.strip()}

    ds = (
        # every rank reads the same samples; the Trainer's eval sampler splits them across ranks
        wds.WebDataset(urls, resampled=False, shardshuffle=False, nodesplitter=lambda s: s,
                       handler=wds.warn_and_continue)
        .decode("pil", handler=wds.warn_and_continue)
        .map(preprocess, handler=wds.warn_and_continue)
    )
    out = []
    for s in ds:
        out.append(s)
        if len(out) >= n:
            break
    return out


_DROPS: dict = {}


def drop(where: str, why: str) -> None:
    """Report a rejected sample as `[drop] where: why (xN)`: the first 5 of each (where, why), then every
    10,000th."""
    n = _DROPS[(where, why)] = _DROPS.get((where, why), 0) + 1
    if n <= 5 or n % 10_000 == 0:
        print(f"[drop] {where}: {why} (x{n})", flush=True)
