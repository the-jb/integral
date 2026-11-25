"""Per-layer, per-channel statistics of the vision tower's hidden states, for the abstraction fusion.

Runs ImageNet-train images (the first `--images` of a seed-0 shuffle, resized and centre-cropped to `--resolution`)
through the Qwen3.5 vision tower and accumulates, in float64 over (images x patches), the sum and sum of squares of
every hidden state the abstraction fusion reads. Saves `layer_mean` / `layer_std` [num_vision_hidden_states,
vision_hidden_size] (std clamped at 1e-4), which stage `main` of train_vision_tokenizer.py sets with
`abstraction_fusion.set_layer_statistics`.

    python tools/layer_statistics.py --config configs/<tokenizer config> --out <file>.pt
"""

import argparse
import os
import sys

import torch
import yaml
from datasets import load_dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integral.configuration_integral import IntegralVisionTokenizerConfig  # noqa: E402
from integral.modeling_integral import IntegralVisionTokenizer  # noqa: E402
from data_pipeline.data_paths import dataset_path  # noqa: E402
from data_pipeline.patches import patchify_batch  # noqa: E402
from train_vision_tokenizer import load_vision_tower, normalize  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", type=str, required=True,
               help="a stage-main config: its model_config and vision_model_path are read")
p.add_argument("--images", type=int, default=50000)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--data", type=str, default=dataset_path("imagenet-1k"))
p.add_argument("--split", type=str, default="train")
p.add_argument("--resolution", type=int, default=256)
p.add_argument("--out", type=str, required=True)
args = p.parse_args()
assert not os.path.exists(args.out), f"{args.out} exists; refusing to overwrite it"

with open(args.config) as f:
    run = yaml.safe_load(f)
config = IntegralVisionTokenizerConfig(**run["model_config"])

device = "cuda"
tokenizer = IntegralVisionTokenizer(config)
load_vision_tower(tokenizer, run["vision_model_path"])
vision_tower = tokenizer.visual.to(device).eval()
del tokenizer

tf = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ]
)
ds = load_dataset(args.data, split=args.split).shuffle(seed=0).select(range(args.images))

s1 = s2 = None  # sized once the number of hidden states is known
count = 0

with torch.no_grad():
    for i in range(0, args.images, args.batch_size):
        batch = torch.stack([tf(r["image"].convert("RGB"))
                             for r in ds.select(range(i, min(i + args.batch_size, args.images)))])
        patches, thw = patchify_batch(normalize(config, batch.to(device)), config)
        hidden_states, _ = vision_tower(patches, thw)
        lv = torch.stack(list(hidden_states), dim=1).double()  # [B*N, L, D]
        if s1 is None:
            L, D = lv.shape[1], lv.shape[2]
            s1 = torch.zeros(L, D, dtype=torch.float64, device=device)
            s2 = torch.zeros(L, D, dtype=torch.float64, device=device)
        s1 += lv.sum(dim=0)
        s2 += (lv * lv).sum(dim=0)
        count += lv.shape[0]
        if (i // args.batch_size) % 50 == 0:
            print(f"{i}/{args.images} images ({count} vectors/layer)", flush=True)

mean = (s1 / count).float()
std = ((s2 / count) - (s1 / count) ** 2).clamp(min=0).sqrt().clamp(min=1e-4).float()

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
torch.save({"layer_mean": mean.cpu(), "layer_std": std.cpu(), "images": args.images, "split": args.split,
            "resolution": args.resolution, "vision_model_path": run["vision_model_path"]}, args.out)
print(f"saved {args.out}: {count} vectors/layer from {args.images} {args.split} images")
for layer in range(L):
    print(f"  layer {layer}: mean|.|={mean[layer].abs().mean():.4f} "
          f"std range {std[layer].min():.4f}..{std[layer].max():.4f}")
