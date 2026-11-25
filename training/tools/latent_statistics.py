"""Per-channel statistics of the vision tokenizer's latent (the VAE posterior mean), written into the tokenizer.

Iterates ImageNet-train images through a trained IntegralVisionTokenizer, accumulates sum / sum of squares of the
un-standardized mu in float64 over (images x patches), sets them as the VAE's `latent_mean` / `latent_std`
(`set_latent_statistics`) and saves the tokenizer to a new directory. Every latent the model exchanges -- the context
image's, and the visual generation model's targets -- is standardized with them.

    python tools/latent_statistics.py --tokenizer <tokenizer dir> --out <new tokenizer dir>
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from integral.modeling_integral import IntegralVisionTokenizer  # noqa: E402
from data_pipeline.data_paths import dataset_path  # noqa: E402
from data_pipeline.patches import patchify_batch  # noqa: E402
from train_vision_tokenizer import normalize  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", type=str, required=True, help="an IntegralVisionTokenizer directory")
p.add_argument("--images", type=int, default=50000)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--data", type=str, default=dataset_path("imagenet-1k"))
p.add_argument("--split", type=str, default="train")
p.add_argument("--resolution", type=int, default=256)
p.add_argument("--out", type=str, required=True, help="the tokenizer with its statistics set; must not exist")
args = p.parse_args()
assert not os.path.exists(args.out), f"{args.out} exists; refusing to write into it"

device = "cuda"
model = IntegralVisionTokenizer.from_pretrained(args.tokenizer, dtype=torch.float32).to(device).eval()
config = model.config

tf = transforms.Compose(
    [
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ]
)
ds = load_dataset(args.data, split=args.split).shuffle(seed=0).select(range(args.images))

s1 = s2 = None
count = 0

with torch.no_grad():
    for i in range(0, args.images, args.batch_size):
        batch = torch.stack([tf(r["image"].convert("RGB"))
                             for r in ds.select(range(i, min(i + args.batch_size, args.images)))])
        patches, thw = patchify_batch(normalize(config, batch.to(device)), config)
        # the posterior mean before standardization, as `encode` computes it
        hidden_states, _ = model.visual(patches, thw)
        x = model.bottleneck(model.abstraction_fusion(hidden_states))
        mu = model.bottleneck.vae.encoder(x).chunk(2, dim=-1)[0].double()  # [B*N, D]
        if s1 is None:
            D = mu.shape[-1]
            s1 = torch.zeros(D, dtype=torch.float64, device=device)
            s2 = torch.zeros(D, dtype=torch.float64, device=device)
        s1 += mu.sum(dim=0)
        s2 += (mu * mu).sum(dim=0)
        count += mu.shape[0]
        if (i // args.batch_size) % 50 == 0:
            print(f"{i}/{args.images} images ({count} vectors)", flush=True)

mean = (s1 / count).float()
std = ((s2 / count) - (s1 / count) ** 2).clamp(min=0).sqrt().clamp(min=1e-4).float()

model.bottleneck.vae.set_latent_statistics(mean, std)
model.save_pretrained(args.out, max_shard_size="50GB")     # one model.safetensors
print(f"saved {args.out}: statistics from {count} vectors of {args.images} {args.split} images")
print(f"  mean|.|={mean.abs().mean():.4f} std range {std.min():.4f}..{std.max():.4f}")
