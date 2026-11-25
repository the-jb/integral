"""Training of the Integral vision tokenizer (`IntegralVisionTokenizer`), in two stages.

    accelerate launch --config_file configs/<accelerate config> train_vision_tokenizer.py main --config_path configs/<config>
    torchrun --nproc_per_node=8 train_vision_tokenizer.py gan --config_path configs/<config>

Stage `main` (HF Trainer). The vision tower is Qwen3.5's, loaded from `vision_model_path` and frozen. The abstraction
fusion, bottleneck and VAE, merger and pixel decoder train; the visual generation adapter stays frozen. The objective
is

    recl1_weight * L1(recon, image) + lpips_weight * LPIPS(recon, image) + kl_weight * KL
      + local_distill_weight * (raw MSE + ln_mse_ratio * normalized MSE) / (1 + ln_mse_ratio)

where the distillation compares the merger's output with Qwen3.5's own patch merger applied to the vision tower's last
block output (`teacher_merger`, frozen). `layer_statistics` (from `tools/layer_statistics.py`) sets the abstraction
fusion's per-layer standardization. Batches are aspect-ratio buckets (`bucket_config`); eval is ImageNet-1K validation
(PSNR, SSIM, rFID, and the distillation MSE / cosine).

Stage `gan` (a DDP loop: the pixel decoder and the discriminator, two optimizers updated in alternation). Starts from a
trained tokenizer directory (`hf_pretrained`) and trains the pixel decoder alone against a DINO discriminator, with
L1 + LPIPS + an adaptively weighted GAN loss. Every other module is frozen and in eval mode, so the latent is the
posterior mean and the merger's output equals the starting tokenizer's; the eval's distillation MSE / cosine check that.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image, PngImagePlugin
from simple_parsing import Serializable, parse
from torch.utils.data import DataLoader
from torchmetrics.image import FrechetInceptionDistance, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from transformers import TrainerCallback, TrainingArguments

from data_pipeline.bucket_dataset import build_bucket_train_dataset, build_imagenet_eval
from data_pipeline.build_datasets import build_datasets
from data_pipeline.patches import patchify_batch
from integral.configuration_integral import IntegralVisionTokenizerConfig
from integral.modeling_integral import IntegralLatentVAE, IntegralVisionTokenizer
from losses import LPIPS, build_discriminator, hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from utils import distributed as misc
from utils.trainer import CustomLoggingTrainer

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = None


def image_block(config) -> int:
    """Both image sides must be a multiple of this: the merger folds `spatial_merge_size` patches per side."""
    return config.patch_size * config.spatial_merge_size


def load_teacher_merger(vision_model_path: str) -> nn.Module:
    """Qwen3.5's own vision patch merger, frozen: the distillation target, applied to the vision tower's last block
    output (`vision_last_hidden_state`). Weights `model.visual.merger.*` of the Qwen3.5 checkpoint."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchMerger

    vision_config = AutoConfig.from_pretrained(vision_model_path).vision_config
    merger = Qwen3_5VisionPatchMerger(vision_config)
    state = _read_prefixed(vision_model_path, "model.visual.merger.")
    merger.load_state_dict(state, strict=True)
    merger.requires_grad_(False)
    return merger.eval()


def load_vision_tower(model: IntegralVisionTokenizer, vision_model_path: str) -> None:
    """Qwen3.5's vision tower weights (`model.visual.*` without the patch merger) into `model.visual`, strictly."""
    state = {k: v for k, v in _read_prefixed(vision_model_path, "model.visual.").items()
             if not k.startswith("merger.")}
    model.visual.load_state_dict(state, strict=True)


def _read_prefixed(path: str, prefix: str) -> dict:
    """The tensors under `prefix` in a (possibly sharded) safetensors checkpoint, prefix stripped. `path` is a local
    directory or a Hugging Face model id, of which only the shards holding `prefix` are downloaded."""
    import json

    from safetensors import safe_open

    def local(name):
        if os.path.isdir(path):
            return os.path.join(path, name)
        from huggingface_hub import hf_hub_download
        return hf_hub_download(path, name)

    try:
        index = local("model.safetensors.index.json")
    except Exception:  # noqa: BLE001 -- an unsharded hub checkpoint has no index
        index = None
    if index is not None and os.path.exists(index):
        with open(index) as f:
            files = sorted({v for k, v in json.load(f)["weight_map"].items() if k.startswith(prefix)})
    else:
        files = ["model.safetensors"]
    out = {}
    for name in files:
        with safe_open(local(name), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = f.get_tensor(k)
    assert out, f"no tensor under {prefix} in {path}"
    return out


def check_patch_grid(config, pixel_values: torch.Tensor):
    """Assert that the batch's patch grid divides by `spatial_merge_size` on both sides."""
    gh, gw = pixel_values.shape[-2] // config.patch_size, pixel_values.shape[-1] // config.patch_size
    d = config.spatial_merge_size
    assert gh % d == 0 and gw % d == 0, f"patch grid {gh}x{gw} must divide by spatial_merge_size {d}"


def normalize(config, pixel_values: torch.Tensor) -> torch.Tensor:
    """[0, 1] pixels -> the vision tokenizer's input normalization (`image_mean` / `image_std`)."""
    mean = torch.tensor(config.image_mean, dtype=pixel_values.dtype, device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor(config.image_std, dtype=pixel_values.dtype, device=pixel_values.device).view(1, 3, 1, 1)
    return (pixel_values - mean) / std


def load_demo_image(path: str, demo_resolution, block: int) -> Optional[torch.Tensor]:
    """The demo reconstruction input `[1, 3, H, W]` at `demo_resolution` ([width, height]): the image at `path`
    scaled until it covers that size, then centre-cropped. None when `path` is None (no demo)."""
    if path is None:
        return None
    img = Image.open(path).convert("RGB")
    demo_w, demo_h = demo_resolution
    assert demo_w % block == 0 and demo_h % block == 0, (
        f"demo_resolution {demo_w}x{demo_h} is not a multiple of the {block}px image block")
    w, h = img.size
    scale = max(demo_w / w, demo_h / h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - demo_w) // 2, (h - demo_h) // 2
    img = img.crop((left, top, left + demo_w, top + demo_h))
    return transforms.ToTensor()(img).unsqueeze(0)


def collate_fn(batch):
    """Eval examples -> `pixel_values` [B, 3, H, W]."""
    return {"pixel_values": torch.stack([item["pixel_values"] for item in batch])}


@dataclass
class VisionTokenizerTrainingArguments(TrainingArguments, Serializable):
    """Training arguments for stage `main`."""

    job_name: str = None
    # `IntegralVisionTokenizerConfig` fields. A dict, not the config class: simple_parsing would try to decompose a
    # PretrainedConfig into its dataclass fields.
    model_config: dict = None
    # The Qwen3.5 model whose vision tower the tokenizer runs on: the vision tower's weights and the distillation
    # target's patch merger come from here.
    vision_model_path: str = None
    # Per-layer statistics of the vision tower's hidden states, from tools/layer_statistics.py.
    layer_statistics: str = None

    # Aspect-ratio buckets: token_budget, ar_temp, shuffle_buffer, bucket_classes, sources [{name, share, hflip}].
    bucket_config: dict = None
    resolution: int = 256

    # Warm starts
    hf_pretrained: str = None  # an IntegralVisionTokenizer directory; its tensors (but the vision tower's) of matching shape load
    pretrained_model: str = None  # a checkpoint in this TrainingModel's own namespace, loaded non-strictly

    # Loss weights
    recl1_weight: float = field(default=1.0, metadata={"help": "L1 reconstruction loss weight"})
    lpips_weight: float = field(default=1.0, metadata={"help": "LPIPS loss weight"})
    local_distill_weight: float = field(
        default=0.0,
        metadata={"help": "Distillation of the merger output to Qwen3.5's patch merger output. 0 = disabled."},
    )
    distill_norm: str = field(
        default="ln",
        metadata={
            "help": "The normalization of the second distill term: 'ln' = LayerNorm (2(1 - mean-removed cosine)), "
            "'rms' = RMS normalization (2(1 - cosine), the quantity eval reports as distill_cos)."
        },
    )
    ln_mse_ratio: float = field(
        default=1.0,
        metadata={
            "help": "Distill = (raw_MSE + ln_mse_ratio * normalized_MSE) / (1 + ln_mse_ratio). ratio=1.0 -> "
            "(raw+norm)/2 (scale-matched to a single MSE)."
        },
    )
    train_modules: list = field(
        default=None,
        metadata={
            "help": "Train only the parameters whose name starts with one of these prefixes; freeze everything else. "
            "Prefixes are matched against TrainingModel.named_parameters(), e.g. ['model.abstraction_fusion.', "
            "'model.bottleneck.']. None = no effect. Every prefix must hit at least one parameter, or it raises."
        },
    )
    kl_weight: float = field(default=1e-3, metadata={"help": "KL divergence loss weight for the VAE"})

    evaluate_gen: bool = field(default=False, metadata={"help": "Evaluate only and exit"})
    # An image reconstructed every `demo_steps` (saved under <output_dir>/demo/ and logged); None = no demo.
    demo_image: str = None
    demo_steps: int = 500
    demo_resolution: list = field(
        default_factory=lambda: [256, 256],
        metadata={"help": "Demo reconstruction size as [width, height]."},
    )

    # Default overrides
    output_dir: str = field(default="./ckpts")
    num_train_epochs: int = field(default=800)
    per_device_train_batch_size: int = field(default=16)
    remove_unused_columns: bool = False
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = "constant"
    accelerator_config: dict[str] = field(
        default_factory=lambda: {
            "dispatch_batches": False,
            "split_batches": False,
        },
    )

    report_to: str = "none"
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 2503
    save_steps: int = 2503
    max_steps: int = 150000
    logging_steps: int = 10

    max_grad_norm: float = 1.0
    save_safetensors: bool = False
    reset_state: bool = field(default=False, metadata={"help": "Reset optimizer/scheduler/random on resume"})

    def __post_init__(self):
        if self.job_name is not None:
            self.run_name = f"{self.job_name}-{self.run_name}"
            output_path = Path(self.output_dir)
            self.output_dir = output_path.with_name(f"{self.job_name}-{output_path.name}")
        super().__post_init__()


class TrainingModel(nn.Module):
    def __init__(self, args: VisionTokenizerTrainingArguments):
        super().__init__()

        self.args = args
        self.use_local_distill = args.local_distill_weight > 0

        config = IntegralVisionTokenizerConfig(**args.model_config)
        self.model = IntegralVisionTokenizer(config)
        # The vision tower is Qwen3.5's, always frozen (IntegralVisionTokenizer freezes it).
        load_vision_tower(self.model, args.vision_model_path)
        assert args.layer_statistics, "layer_statistics is required: the abstraction fusion standardizes every layer"
        statistics = torch.load(args.layer_statistics, map_location="cpu", weights_only=True)
        self.model.abstraction_fusion.set_layer_statistics(statistics["layer_mean"], statistics["layer_std"])
        # the visual generation adapter is trained by the visual generation training
        self.model.visual_generation_adapter.requires_grad_(False)

        if args.hf_pretrained:
            print(f"Warm-starting the vision tokenizer from: {args.hf_pretrained}")
            from safetensors.torch import load_file

            ck = load_file(str(Path(args.hf_pretrained) / "model.safetensors"))
            msd = self.model.state_dict()
            filt = {k: v for k, v in ck.items()
                    if not k.startswith("visual.") and k in msd and msd[k].shape == v.shape}
            self.model.load_state_dict(filt, strict=False)
            print(f"[warm-start] loaded {len(filt)}/{len(ck)} keys (vision tower excluded)")

        # The distillation target: Qwen3.5's own patch merger on the vision tower's last block output (frozen).
        self.teacher_merger = load_teacher_merger(args.vision_model_path) if self.use_local_distill else None

        self.lpips = LPIPS().eval()
        self.lpips.requires_grad_(False)

        if args.pretrained_model:
            ckpt_dir = Path(args.pretrained_model)
            st_path = ckpt_dir / "model.safetensors"
            if st_path.exists():
                from safetensors.torch import load_file

                ckpt = load_file(str(st_path))
            else:
                ckpt = torch.load(ckpt_dir / "pytorch_model.bin", weights_only=False, map_location="cpu")

            self.load_state_dict(ckpt, strict=False)
            print(f"[warm-start] loaded {len(ckpt)} keys from {ckpt_dir.name}")

        if args.train_modules:
            self._freeze_all_but(args.train_modules)

    def _freeze_all_but(self, prefixes):
        """Freeze every parameter whose name starts with none of `prefixes`; a prefix that matches nothing raises."""
        prefixes = tuple(prefixes)
        hit = {p: 0 for p in prefixes}
        frozen, trainable = 0, 0
        for name, p in self.named_parameters():
            match = next((pre for pre in prefixes if name.startswith(pre)), None)
            if match is not None:
                hit[match] += p.numel()
                trainable += p.numel()
            else:
                p.requires_grad_(False)
                frozen += p.numel()
        missed = [pre for pre, n in hit.items() if n == 0]
        if missed:
            raise ValueError(f"train_modules: these prefixes matched no parameter (typo?): {missed}")
        print(
            f"[train_modules] trainable={trainable / 1e6:.1f}M  frozen={frozen / 1e6:.1f}M  ("
            + ", ".join(f"{pre}{n / 1e6:.1f}M" for pre, n in hit.items())
            + ")"
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_eval_outputs: bool = False,
    ):
        """`pixel_values` [B, 3, H, W] in [0, 1] -> `(loss, [total, rec, lpips, kl, local_distill, distill_raw,
        distill_ln])`, or with `return_eval_outputs` `(reconstruction, merger output, distillation target)`."""
        config = self.model.config
        check_patch_grid(config, pixel_values)
        patches, thw = patchify_batch(normalize(config, pixel_values), config)
        model_out = self.model(patches, thw, with_reconstruction=True, with_merger=self.use_local_distill)
        reconstructed = torch.stack(model_out.reconstructed_images)
        orig_dtype = pixel_values.dtype

        target = None
        if self.use_local_distill:
            with torch.no_grad():
                target = self.teacher_merger(model_out.vision_last_hidden_state.to(
                    self.teacher_merger.linear_fc1.weight.dtype))

        if return_eval_outputs:
            # eval goes through this forward: FSDP2 unshards the parameters in its forward hooks
            distill_pred = model_out.merger_output.float() if self.use_local_distill else None
            return reconstructed, distill_pred, target

        # LPIPS takes [-1, 1]
        images_normed = pixel_values * 2.0 - 1.0
        recon_normed = reconstructed * 2.0 - 1.0

        rec_loss = F.l1_loss(reconstructed, pixel_values)

        # LPIPS in fp32
        with torch.autocast(device_type=pixel_values.device.type, enabled=False):
            lpips_loss = self.lpips(images_normed.float(), recon_normed.float()).mean()
        lpips_loss = lpips_loss.to(orig_dtype)

        # Distillation (fp32): the MSE to the patch merger's output itself, and the MSE after `distill_norm` (which
        # removes the magnitude), both also logged on their own (loss_distill_raw / loss_distill_ln).
        loss_local_distill = 0
        distill_raw = distill_ln = 0
        if self.use_local_distill:
            with torch.autocast(device_type=pixel_values.device.type, enabled=False):
                r = self.args.ln_mse_ratio
                _rms = (lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
                _norm = _rms if self.args.distill_norm == "rms" else (
                    lambda x: F.layer_norm(x, x.shape[-1:])
                )

                pred = model_out.merger_output.float()
                tgt = target.float().detach()
                distill_raw = (pred - tgt).pow(2).mean(-1).mean()   # per-token mean squared error, averaged
                distill_ln = F.mse_loss(_norm(pred), _norm(tgt))
                loss_local_distill = (distill_raw + r * distill_ln) / (1.0 + r)
            loss_local_distill = loss_local_distill.to(orig_dtype)

        loss_kl = model_out.kl_loss

        loss = (
            self.args.recl1_weight * rec_loss
            + self.args.lpips_weight * lpips_loss
            + self.args.local_distill_weight * loss_local_distill
            + self.args.kl_weight * loss_kl
        )

        def _t(v):
            return v.detach() if torch.is_tensor(v) else torch.tensor(float(v), device=loss.device)

        # Order: [total, rec, lpips, kl, local_distill, distill_raw, distill_ln]
        log_tensor = torch.stack([_t(loss), _t(rec_loss), _t(lpips_loss), _t(loss_kl), _t(loss_local_distill),
                                  _t(distill_raw), _t(distill_ln)])
        return loss, log_tensor


class VisionTokenizerTrainer(CustomLoggingTrainer, TrainerCallback):
    args: VisionTokenizerTrainingArguments

    def __init__(self, args: VisionTokenizerTrainingArguments):
        geom = IntegralVisionTokenizerConfig(**(args.model_config or {}))
        assert args.bucket_config, "bucket_config is required: batches are aspect-ratio buckets"
        train_ds = build_bucket_train_dataset(
            args.bucket_config,
            seed=args.seed,
            patch_size=geom.patch_size,
            merge_size=geom.spatial_merge_size,
        )
        val_ds = build_imagenet_eval(args.resolution)
        model = TrainingModel(args)

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            callbacks=[self],
        )
        if self.is_world_process_zero():
            print(f"{args.world_size=}, {args.local_rank=}")
            print(f"{args=}")
            print(f"{model=}")
            print(f"{train_ds=}, {val_ds=}")
            config_path = os.path.join(self.args.output_dir, "training_config.yaml")
            if not os.path.exists(config_path):
                os.makedirs(self.args.output_dir, exist_ok=True)
                with open(config_path, "w") as fp:
                    args.dump_yaml(fp)

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.args.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.args.device)
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(self.args.device)
        # per-batch distillation MSE / cosine on the eval set
        self.eval_distill_mse = []
        self.eval_distill_cos = []

        self.demo_input_transform = load_demo_image(args.demo_image, args.demo_resolution, image_block(geom))

        # Bucket batches hold different numbers of images, so `compute_loss` weighs each microbatch by its image count
        # (`_get_num_items_in_batch`); `model_accepts_loss_kwargs` stops `training_step` dividing by the accumulation
        # steps.
        self._num_images_local = None
        self.model_accepts_loss_kwargs = True

    def _get_num_items_in_batch(self, batch_samples, device):
        """The images in the accumulation window, summed over the ranks (this rank's count kept for logging)."""
        local = sum(b["pixel_values"].shape[0] for b in batch_samples)
        self._num_images_local = local
        total = torch.tensor(float(local), device=device)
        if self.args.world_size > 1:
            total = self.accelerator.gather(total).sum()
        return total

    def get_train_dataloader(self):
        """A plain DataLoader (`batch_size=None`) over the bucket stream, which batches and splits across ranks itself;
        not passed through `accelerator.prepare`."""
        kwargs = {
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(self.train_dataset, batch_size=None, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch: torch.Tensor = None,
    ):
        total_loss, log_tensor = model(**inputs)
        log_dict = {
            "loss_rec": log_tensor[1].item(),
            "loss_lpips": log_tensor[2].item(),
            "loss_kl": log_tensor[3].item(),
            "loss_local_distill": log_tensor[4].item(),
            "loss_distill_raw": log_tensor[5].item(),
            "loss_distill_ln": log_tensor[6].item(),
        }

        if num_items_in_batch is not None:
            n_images = inputs["pixel_values"].shape[0]
            # the per-image mean over the accumulation window and the ranks (`world_size` cancels the gradient
            # reduction's mean over ranks)
            total_loss = total_loss * (n_images * self.args.world_size / num_items_in_batch)
            # the logged terms weighted the same way (custom_log averages over microbatches)
            if self._num_images_local:
                w = n_images * self.current_gradient_accumulation_steps / self._num_images_local
                log_dict = {k: v * w for k, v in log_dict.items()}

        self.custom_log(log_dict)

        return (total_loss, None) if return_outputs else total_loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            pixel_values = inputs["pixel_values"]
            recon, pred, tgt = model(pixel_values=pixel_values, return_eval_outputs=True)

            recon = recon.detach()
            real = pixel_values.detach()

            # the eval loss is the L1 reconstruction error
            loss = F.l1_loss(recon, real)

            self.psnr_metric.update(recon, real)
            self.ssim_metric.update(recon, real)

            # FID on uint8 images, clamped to [0, 1] first
            recon_uint8 = (recon.clamp(0, 1) * 255).to(torch.uint8)
            real_uint8 = (real.clamp(0, 1) * 255).to(torch.uint8)
            self.fid_metric.update(real_uint8, real=True)
            self.fid_metric.update(recon_uint8, real=False)

            # the distillation's prediction against its target: MSE and cosine
            if pred is not None and tgt is not None:
                tgt = tgt.float()
                self.eval_distill_mse.append(F.mse_loss(pred, tgt).item())
                self.eval_distill_cos.append(F.cosine_similarity(pred, tgt, dim=-1).mean().item())

        return (loss, None, None)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )
        eval_metrics = {
            "psnr": self.psnr_metric.compute().item(),
            "ssim": self.ssim_metric.compute().item(),
            "fid": self.fid_metric.compute().item(),
            "distill_mse": sum(self.eval_distill_mse) / len(self.eval_distill_mse) if self.eval_distill_mse else 0.0,
            "distill_cos": sum(self.eval_distill_cos) / len(self.eval_distill_cos) if self.eval_distill_cos else 0.0,
        }
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.fid_metric.reset()

        self.eval_distill_mse.clear()
        self.eval_distill_cos.clear()

        for k, v in eval_metrics.items():
            output.metrics[f"{metric_key_prefix}_{k}"] = v
        return output

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # every rank runs the demo forward (an FSDP all-gather); rank 0 writes the file
        if self.demo_input_transform is not None and state.global_step % args.demo_steps == 0:
            self._generate_demo(model, args.output_dir, state.global_step, args.device)

    def _generate_demo(self, model, output_dir, step, device):
        model.eval()
        with torch.no_grad():
            demo_in = self.demo_input_transform.to(device)
            recon, _, _ = model(pixel_values=demo_in, return_eval_outputs=True)
            sampled = recon.float()
        model.train()

        if not self.is_world_process_zero():
            return

        save_folder = os.path.join(output_dir, "demo")
        os.makedirs(save_folder, exist_ok=True)
        recon_arr = (sampled.clip(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        recon_img = Image.fromarray(recon_arr)
        recon_img.save(os.path.join(save_folder, f"{step}.png"))

        if wandb.run is not None:
            wandb.log({"demo/reconstruction": wandb.Image(recon_img)}, step=step)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if self.args.reset_state:
            return

        return super()._load_optimizer_and_scheduler(checkpoint)

    def _load_rng_state(self, checkpoint):
        if self.args.reset_state:
            return

        return super()._load_rng_state(checkpoint)


def main(args: VisionTokenizerTrainingArguments):
    ckpt_dir = None
    if args.resume_from_checkpoint:
        ckpt_dir = args.resume_from_checkpoint
        # the bucket stream resamples its shards, so a resume does not fast-forward it; it starts again with seed 84
        args.ignore_data_skip = True
        args.seed = 84

    trainer = VisionTokenizerTrainer(args)

    if args.evaluate_gen:
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            for key in sorted(metrics):
                print(f"{key}: {metrics[key]}", flush=True)
        return

    trainer.train(resume_from_checkpoint=ckpt_dir)


@dataclass
class GANTrainingArguments(Serializable):
    """Arguments of stage `gan` (a plain dataclass; field names match stage `main` where the meaning is the same)."""

    job_name: str = None

    resolution: int = 256

    # The tokenizer being tuned: an IntegralVisionTokenizer directory.
    hf_pretrained: str = None
    # The Qwen3.5 model whose patch merger is the target of the eval's distillation check.
    vision_model_path: str = None
    resume: str = None

    # Reconstruction
    recl1_weight: float = 1.0
    lpips_weight: float = 1.0

    # GAN
    gan_weight: float = 0.5
    gan_start_step: int = 0
    disc_upd_start_step: int = 0
    adaptive_weight: bool = True
    max_d_weight: float = 1.0e4
    disc_loss: str = "hinge"
    gen_loss: str = "vanilla"
    disc_ckpt: str = None  # the discriminator's DINO ViT-S/8 weights: a local file (the launcher downloads a HF repo id)
    disc_recipe: str = "S_8"
    disc_ks: int = 9
    disc_lr: float = 2.0e-5
    disc_weight_decay: float = 0.0
    disc_aug_prob: float = 1.0
    disc_aug_cutout: float = 0.0

    # Optimization (no gradient accumulation: the discriminator updates once per generator step)
    per_device_train_batch_size: int = 16
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    lr_scheduler_type: str = "constant"
    lr_min: float = 1.0e-6
    max_grad_norm: float = 1.0
    max_steps: int = 50000
    seed: int = 0
    bf16: bool = True

    # Latent noise in training only: z = mu + latent_noise_scale * sigma * eps (per patch and channel). 0 = mu, the
    # inference path; 1 = the VAE's own reparameterization. Eval always uses mu.
    latent_noise_scale: float = 0.0

    # Eval / logging / saving
    eval_steps: int = 5000
    save_steps: int = 5000
    logging_steps: int = 10
    # An image reconstructed every `demo_steps` (saved under <output_dir>/demo/ and logged); None = no demo.
    demo_image: str = None
    demo_steps: int = 500
    demo_resolution: list = field(default_factory=lambda: [256, 256])
    eval_on_start: bool = False
    evaluate_gen: bool = False

    dataloader_num_workers: int = 4
    dataloader_prefetch_factor: int = 4

    # None = `<hf_pretrained>/gan-<YYYYmmdd_HHMMSS>`, rank 0's time broadcast to every rank (`resolve_output_dir`).
    output_dir: str = None
    run_name: str = "vision-tokenizer-gan"
    project: str = "vision-tokenizer"
    wandb_entity: str = None
    wandb_id: str = None

    # set by distributed.init_distributed_mode from torchrun's environment
    world_size: int = 1
    local_rank: int = -1
    dist_url: str = "env://"
    device: str = "cuda"

    def __post_init__(self):
        if self.job_name is not None:
            # job_name names the run only; the output directory is under the tokenizer being tuned
            self.run_name = f"{self.job_name}-{self.run_name}"


class ScaledNoiseLatentVAE(IntegralLatentVAE):
    """`IntegralLatentVAE` whose reparameterization noise is scaled by `noise_scale` and switched by `noise_active`
    (not by `self.training`: the module is held in eval). Installed by setting `__class__` on the existing instance,
    so the parameters and state-dict keys are unchanged."""

    noise_scale: float = 0.0
    noise_active: bool = False

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        if self.noise_active and self.noise_scale > 0:
            z = mu + self.noise_scale * (0.5 * logvar).exp() * torch.randn_like(mu)
        else:
            z = mu
        z_hat = self.decoder(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return z_hat, kl, (mu - self.latent_mean) / self.latent_std


class GANTrainingModel(nn.Module):
    """The generator: the vision tokenizer with only its pixel decoder trained; returns the reconstruction (the
    training loop computes the losses)."""

    DECODER_PREFIXES = ("model.pixel_decoder.",)

    def __init__(self, args: GANTrainingArguments):
        super().__init__()
        self.args = args
        assert args.hf_pretrained, "hf_pretrained is required: the vision tokenizer being tuned"
        self.model, info = IntegralVisionTokenizer.from_pretrained(
            args.hf_pretrained, dtype=torch.float32, output_loading_info=True)
        # every key must load
        bad = {k: v for k, v in info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
        if bad:
            raise RuntimeError(f"{args.hf_pretrained} did not load cleanly: {bad}")
        self.teacher_merger = load_teacher_merger(args.vision_model_path)

        self._freeze_all_but_decoder()

        vae = self.model.bottleneck.vae
        vae.__class__ = ScaledNoiseLatentVAE
        vae.noise_scale = float(args.latent_noise_scale)
        vae.noise_active = False
        if vae.noise_scale > 0:
            print(f"[latent-noise] z = mu + {vae.noise_scale} * sigma * eps during training "
                  f"(per patch token, per channel); eval stays at mu")
        else:
            print("[latent-noise] off: z = mu everywhere")

    def _freeze_all_but_decoder(self):
        self._trainable_names, frozen = [], []
        for name, p in self.named_parameters():
            if name.startswith(self.DECODER_PREFIXES):
                p.requires_grad_(True)
                self._trainable_names.append(name)
            else:
                p.requires_grad_(False)
                frozen.append(name)
        if not self._trainable_names:
            raise RuntimeError(f"no parameter matched {self.DECODER_PREFIXES} -- nothing would train")

        sizes = dict(self.named_parameters())
        n_tr = sum(sizes[n].numel() for n in self._trainable_names)
        n_fr = sum(sizes[n].numel() for n in frozen)
        print(f"[decoder-only] trainable={n_tr / 1e6:.1f}M ({len(self._trainable_names)} tensors)  "
              f"frozen={n_fr / 1e6:.1f}M ({len(frozen)} tensors)")

    def _frozen_modules(self):
        m = self.model
        return [m.visual, m.abstraction_fusion, m.bottleneck, m.merger, m.visual_generation_adapter,
                self.teacher_merger]

    def train(self, mode: bool = True):
        """The frozen modules stay in eval mode; `mode` switches the latent noise (`ScaledNoiseLatentVAE`)."""
        super().train(mode)
        for m in self._frozen_modules():
            m.eval()
        self.model.bottleneck.vae.noise_active = bool(mode)
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def last_layer(self):
        """The layer the adaptive GAN weight is measured at -- the pixel decoder's final projection."""
        return self.model.pixel_decoder.conv_out.weight

    def forward(self, pixel_values: torch.Tensor, return_eval_outputs: bool = False):
        """-> (reconstruction in [0, 1], merger output, teacher merger output); the last two only with
        `return_eval_outputs` (eval's distillation check), else None."""
        config = self.model.config
        check_patch_grid(config, pixel_values)
        patches, thw = patchify_batch(normalize(config, pixel_values), config)
        out = self.model(patches, thw, with_reconstruction=True, with_merger=return_eval_outputs)
        recon = torch.stack(out.reconstructed_images)
        if not return_eval_outputs:
            return recon, None, None
        with torch.no_grad():
            target = self.teacher_merger(out.vision_last_hidden_state.to(self.teacher_merger.linear_fc1.weight.dtype))
        return recon, out.merger_output, target


def calculate_adaptive_weight(recon_loss, gan_loss, layer, max_d_weight=1e4):
    """The GAN loss weight |grad recon| / |grad gan| at `layer`, clamped to [0, max_d_weight].

    `torch.autograd.grad` does not accumulate into `.grad`, so DDP's reducer is not triggered.
    """
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    return torch.clamp(d_weight, 0.0, max_d_weight).detach()


def build_loaders(args: GANTrainingArguments):
    train_ds, val_ds = build_datasets(resolution=args.resolution)

    kw = dict(num_workers=args.dataloader_num_workers, pin_memory=True, collate_fn=collate_fn)
    if args.dataloader_num_workers > 0:
        kw["prefetch_factor"] = args.dataloader_prefetch_factor

    # the training stream splits itself across ranks and never ends
    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size,
                              drop_last=True, **kw)

    world = misc.get_world_size()
    n = len(val_ds)
    # DistributedSampler would pad the eval set by repeating images
    if n % world != 0:
        raise ValueError(f"eval set has {n} images, not divisible by world_size {world}; "
                         f"the sampler would pad and the metrics would double-count")
    val_sampler = torch.utils.data.DistributedSampler(
        val_ds, num_replicas=world, rank=misc.get_rank(), shuffle=False, drop_last=False
    )
    val_loader = DataLoader(val_ds, sampler=val_sampler,
                            batch_size=args.per_device_train_batch_size, drop_last=False, **kw)
    return train_loader, val_loader, n


@torch.no_grad()
def evaluate(model, loader, args, device):
    """PSNR, SSIM, FID and the distillation MSE / cosine on the eval set, over all ranks (`model` unwrapped)."""
    was_training = model.training
    model.eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    acc = torch.zeros(3, device=device)  # [distill_mse, distill_cos, n_batches]
    n_images = torch.zeros((), device=device)

    for batch in loader:
        real = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
            recon, pred, tgt = model(real, return_eval_outputs=True)
        recon = recon.float().detach()

        psnr.update(recon, real)
        ssim.update(recon, real)
        # FID on uint8 images, clamped to [0, 1] first
        fid.update((real.clamp(0, 1) * 255).to(torch.uint8), real=True)
        fid.update((recon.clamp(0, 1) * 255).to(torch.uint8), real=False)
        n_images += real.shape[0]

        if pred is not None and tgt is not None:
            pred, tgt = pred.float(), tgt.float()
            acc[0] += F.mse_loss(pred, tgt)
            acc[1] += F.cosine_similarity(pred, tgt, dim=-1).mean()
            acc[2] += 1

    if misc.is_dist_avail_and_initialized():
        dist.all_reduce(acc)
        dist.all_reduce(n_images)

    metrics = {
        "eval/psnr": psnr.compute().item(),
        "eval/ssim": ssim.compute().item(),
        "eval/fid": fid.compute().item(),
        "eval/n": n_images.item(),
    }
    if acc[2] > 0:
        metrics["eval/distill_mse"] = (acc[0] / acc[2]).item()
        metrics["eval/distill_cos"] = (acc[1] / acc[2]).item()

    model.train(was_training)
    return metrics


@torch.no_grad()
def generate_demo(model, demo_input, output_dir, step, device, args):
    was_training = model.training
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16):
        recon, _, _ = model(demo_input.to(device))
    model.train(was_training)

    if not misc.is_main_process():
        return
    folder = os.path.join(output_dir, "demo")
    os.makedirs(folder, exist_ok=True)
    arr = (recon.float().clip(0, 1)[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(os.path.join(folder, f"{step}.png"))
    if wandb.run is not None:
        wandb.log({"demo/reconstruction": wandb.Image(img)}, step=step)


def save_checkpoint(path, model, optimizer, scheduler, disc, disc_optimizer, disc_scheduler,
                    step, args):
    blob = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "discriminator": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict(),
        "step": step,
        "args": args.to_dict(),
    }
    misc.save_on_master(blob, path)
    if misc.is_main_process():
        print(f"[save] {path}", flush=True)


def resolve_resume(resume: str) -> Optional[Path]:
    """A checkpoint file, or the highest-step `checkpoint-<step>.pth` in a directory; None if there is none."""
    p = Path(resume)
    if p.is_file():
        return p
    if p.is_dir():
        cands = []
        for c in p.glob("checkpoint-*.pth"):
            tail = c.stem.rsplit("-", 1)[-1]
            if tail.isdigit():
                cands.append((int(tail), c))
        if cands:
            return max(cands)[1]
    return None


STAMP_FMT = "%Y%m%d_%H%M%S"
STAMP_LEN = 15  # fixed width, so the broadcast buffer below has a constant size


def resolve_output_dir(args, device):
    """`args.output_dir`, or `<hf_pretrained>/gan-<YYYYmmdd_HHMMSS>` with rank 0's time broadcast to every rank (so
    it runs after the process group is up)."""
    if args.output_dir:
        return args.output_dir

    stamp = time.strftime(STAMP_FMT)
    assert len(stamp) == STAMP_LEN, f"stamp {stamp!r} is not {STAMP_LEN} chars"
    if misc.is_dist_avail_and_initialized():
        buf = torch.tensor(list(stamp.encode()), dtype=torch.uint8, device=device)
        dist.broadcast(buf, src=0)
        stamp = bytes(buf.cpu().tolist()).decode()

    args.output_dir = str(Path(args.hf_pretrained) / f"gan-{stamp}")
    args.run_name = f"{args.run_name}-{stamp}"
    return args.output_dir


def build_scheduler(optimizer, args):
    if args.lr_scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    if args.lr_scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_steps, eta_min=args.lr_min)
    raise ValueError(f"lr_scheduler_type must be 'constant' or 'cosine', got {args.lr_scheduler_type!r}")


def gan_main(args: GANTrainingArguments):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if args.disc_loss not in ("hinge", "vanilla"):
        raise ValueError(f"disc_loss must be 'hinge' or 'vanilla', got {args.disc_loss!r}")
    if args.gen_loss != "vanilla":
        raise ValueError(f"gen_loss must be 'vanilla', got {args.gen_loss!r}")
    disc_loss_fn = hinge_d_loss if args.disc_loss == "hinge" else vanilla_d_loss

    train_loader, val_loader, n_eval = build_loaders(args)

    model = GANTrainingModel(args).to(device)
    model.train()
    geom = model.model.config
    lpips = LPIPS().to(device).eval()
    lpips.requires_grad_(False)

    disc, disc_aug = build_discriminator(
        ckpt_path=args.disc_ckpt, ks=args.disc_ks, recipe=args.disc_recipe, device=device,
        aug_prob=args.disc_aug_prob, aug_cutout=args.disc_aug_cutout,
    )

    ddp_model = model
    ddp_disc = disc
    if misc.is_dist_avail_and_initialized():
        # every trainable parameter (the pixel decoder's) is used by the reconstruction
        ddp_model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        ddp_disc = nn.parallel.DistributedDataParallel(
            disc, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    disc_params = [p for p in disc.parameters() if p.requires_grad]
    disc_optimizer = torch.optim.AdamW(disc_params, lr=args.disc_lr,
                                       betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.disc_weight_decay)
    disc_scheduler = build_scheduler(disc_optimizer, args)

    output_dir = Path(resolve_output_dir(args, device))
    if misc.is_main_process():
        print(f"[output] {output_dir}", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = output_dir / "training_config.yaml"
        if not cfg_path.exists():
            with open(cfg_path, "w") as fp:
                args.dump_yaml(fp)

    start_step = 0
    ckpt_path = None
    if args.resume:
        ckpt_path = resolve_resume(args.resume)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"--resume {args.resume} matched no checkpoint. Give a checkpoint-*.pth file, or a "
                f"directory containing one."
            )
    if ckpt_path is not None:
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing:
            raise RuntimeError(f"{len(missing)} parameters missing from {ckpt_path}: {missing[:10]}")
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        disc.load_state_dict(ck["discriminator"])
        disc_optimizer.load_state_dict(ck["disc_optimizer"])
        disc_scheduler.load_state_dict(ck["disc_scheduler"])
        start_step = ck["step"]
        if misc.is_main_process():
            print(f"[resume] {ckpt_path} at step {start_step}, ignored {len(unexpected)} extra keys")
        del ck

    if misc.is_main_process():
        wandb.init(project=args.project, entity=args.wandb_entity, name=args.run_name,
                   config=args.to_dict(), id=args.wandb_id,
                   resume="must" if args.wandb_id else None)
        print(f"world_size={misc.get_world_size()}  eval images={n_eval}")
        print(f"[schedule] GAN on generator from step {args.gan_start_step}, "
              f"D updates from step {args.disc_upd_start_step}, gan_weight={args.gan_weight}, "
              f"adaptive={args.adaptive_weight}")

    demo_input = load_demo_image(args.demo_image, args.demo_resolution, image_block(geom))

    if args.evaluate_gen:
        metrics = evaluate(model, val_loader, args, device)
        if misc.is_main_process():
            for k in sorted(metrics):
                print(f"{k}: {metrics[k]}", flush=True)
        return

    if args.eval_on_start:
        metrics = evaluate(model, val_loader, args, device)
        if misc.is_main_process():
            print(f"[eval @ {start_step}] " + "  ".join(f"{k}={v:.6g}" for k, v in metrics.items()),
                  flush=True)
            wandb.log(metrics, step=start_step)

    data_iter = iter(train_loader)
    log_acc = torch.zeros(8, device=device)
    log_n = 0
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        batch = next(data_iter)
        images = batch["pixel_values"].to(device, non_blocking=True)
        images_normed = images * 2.0 - 1.0

        use_gan = step >= args.gan_start_step and args.gan_weight > 0
        train_disc = step >= args.disc_upd_start_step and args.gan_weight > 0

        # generator step; a fresh autocast context per block
        def ac():
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16)

        with ac():
            recon, _, _ = ddp_model(images)
        recon = recon.float()
        recon_normed = recon * 2.0 - 1.0

        rec_loss = F.l1_loss(recon, images)
        with torch.autocast(device_type=device.type, enabled=False):
            lpips_loss = lpips(images_normed.float(), recon_normed).mean()
        recon_loss = args.recl1_weight * rec_loss + args.lpips_weight * lpips_loss

        if use_gan:
            # the discriminator passes gradient to the image only; the module itself, not its DDP wrapper
            disc.requires_grad_(False)
            with ac():
                logits_fake = disc.classify(disc_aug.aug(recon_normed))
            gan_loss = vanilla_g_loss(logits_fake.float())
            if args.adaptive_weight:
                adapt_w = calculate_adaptive_weight(recon_loss, gan_loss, model.last_layer,
                                                    args.max_d_weight)
            else:
                adapt_w = torch.ones((), device=device)
            loss = recon_loss + args.gan_weight * adapt_w * gan_loss
        else:
            gan_loss = torch.zeros((), device=device)
            adapt_w = torch.zeros((), device=device)
            loss = recon_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # discriminator step
        if train_disc:
            disc.requires_grad_(True)
            disc_optimizer.zero_grad(set_to_none=True)
            # the fake is clamped to [-1, 1] here (not in the generator step, where it would cut the gradient)
            fake_detached = recon_normed.detach().clamp(-1.0, 1.0)
            with ac():
                logits_fake_d, logits_real = ddp_disc(disc_aug.aug(fake_detached),
                                                      disc_aug.aug(images_normed.float()))
            logits_fake_d, logits_real = logits_fake_d.float(), logits_real.float()
            d_loss = disc_loss_fn(logits_real, logits_fake_d)
            d_loss.backward()
            disc_optimizer.step()
            disc_scheduler.step()
            # the fraction of pairs where the real image scores above the fake
            d_acc = (logits_real > logits_fake_d).float().mean()
        else:
            d_loss = torch.zeros((), device=device)
            d_acc = torch.zeros((), device=device)

        log_acc += torch.stack([
            loss.detach(), rec_loss.detach(), lpips_loss.detach(), gan_loss.detach(),
            d_loss.detach(), d_acc.detach(), adapt_w.detach().float(), grad_norm.detach(),
        ])
        log_n += 1

        if (step + 1) % args.logging_steps == 0:
            v = log_acc / log_n
            if misc.is_dist_avail_and_initialized():
                dist.all_reduce(v)
                v /= misc.get_world_size()
            log_acc.zero_()
            log_n = 0
            if misc.is_main_process():
                v = v.tolist()
                rec = {
                    "train/loss": v[0], "train/loss_rec": v[1], "train/loss_lpips": v[2],
                    "train/loss_gan": v[3], "train/loss_disc": v[4], "train/disc_accuracy": v[5],
                    "train/adaptive_weight": v[6], "train/grad_norm": v[7],
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/disc_lr": disc_optimizer.param_groups[0]["lr"],
                    "train/steps_per_sec": args.logging_steps / (time.time() - t0),
                }
                wandb.log(rec, step=step + 1)
                print(f"step {step + 1}  loss {v[0]:.5f}  rec {v[1]:.5f}  lpips {v[2]:.5f}  "
                      f"gan {v[3]:.4f}  d {v[4]:.4f}  d_acc {v[5]:.3f}  aw {v[6]:.4g}",
                      flush=True)
            t0 = time.time()

        if demo_input is not None and (step + 1) % args.demo_steps == 0:
            generate_demo(model, demo_input, args.output_dir, step + 1, device, args)

        if (step + 1) % args.eval_steps == 0:
            metrics = evaluate(model, val_loader, args, device)
            if misc.is_main_process():
                print(f"[eval @ {step + 1}] "
                      + "  ".join(f"{k}={x:.6g}" for k, x in metrics.items()), flush=True)
                wandb.log(metrics, step=step + 1)

        if (step + 1) % args.save_steps == 0 or (step + 1) == args.max_steps:
            save_checkpoint(output_dir / f"checkpoint-{step + 1}.pth", model, optimizer, scheduler,
                            disc, disc_optimizer, disc_scheduler, step + 1, args)

    if misc.is_main_process():
        wandb.finish()


STAGES = {"main": (VisionTokenizerTrainingArguments, main), "gan": (GANTrainingArguments, gan_main)}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        raise SystemExit(f"usage: train_vision_tokenizer.py {{{'|'.join(STAGES)}}} --config_path <config>")
    arg_cls, entry = STAGES[sys.argv.pop(1)]
    entry(parse(arg_cls, add_config_path_arg=True))
