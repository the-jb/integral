"""Visual generation training of an Integral model on packed interleaved sequences.

    accelerate launch --config_file configs/<accelerate config> train_visual_generation.py --config_path configs/<config>

Data: `interleaved` configures `data_pipeline.stream.build_interleaved_stream`. One sample is one whole conversation
(an edit chain is one sequence with one flow target per generated image), and several samples are packed into one
row of `row_len` tokens. A row is the microbatch, so `per_device_train_batch_size` is 1.

Losses: cross-entropy over the supervised text and the generation markers, and flow matching over every generated
image in the row. Each is summed per row, divided by its count summed over the ranks and the accumulation window
(`_get_num_items_in_batch`), and the two are combined with `w_text` / `w_img` (`_interleaved_loss`). Cross-entropy is
computed only when the language model trains (`train_language_model`).

Forward: the trainer's model is the `IntegralForConditionalGeneration` itself (the FSDP root).
`VisualGenerationSteps.training_forward_packed` runs the row through its `forward` once; every target's visual
generation model reads that pass's key/values over its own condition, `[sample start, gen_point)`. Checkpoints are
the model's `save_pretrained` output with the starting model's processor.

Packing: the sample boundaries reach the full-attention layers as `cu_seq_lens_q/k` (flash-attention only) and the
linear-attention layers as `seq_idx` / `cu_seqlens`. `assert_packing_supported` stops if a kernel that reads them
is missing.

Trained modules: the visual generation model and the vision tokenizer's visual generation adapter; with
`train_language_model` also the language model, its head and the vision tokenizer's merger. The vision tower,
abstraction fusion, bottleneck and pixel decoder are frozen: they produce the flow targets. The launcher lists the
frozen modules in FSDP_IGNORED_MODULES.

Eval: each image of the eval set (`t2i_sources`, at `resolution`) is put in the context and asked for back with
`instruction`. Metrics: the flow loss at one timestep per image, `x0_mse` (one denoise from pure noise),
`velocity_mse` (velocity MSE on a fixed timestep grid) and `nll_bpd` (bits per dimension of the latent).
"""

import dataclasses
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin
from simple_parsing import Serializable, parse
from transformers import TrainerCallback, TrainingArguments

from data_pipeline.packing import (image_offsets, packing_kwargs, sample_position_ids, slot_latents,
                                   target_condition)
from data_pipeline.segments import IGNORE as IGNORE_INDEX
from data_pipeline.segments import Geometry, PackedBatch
from data_pipeline.stream import build_interleaved_stream
from data_pipeline.t2i_dataset import build_t2i_eval_set
from data_pipeline.patches import patchify
from utils.trainer import CustomLoggingTrainer

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = None

@dataclass
class VisualGenerationArguments(TrainingArguments, Serializable):
    job_name: str = None
    # The starting model: an Integral directory (vision tokenizer, language model, visual generation model and the
    # text tokenizer files).
    model_path: str = None
    # the eval images' size
    resolution: int = 256

    # The eval set: comma-separated names of `data_pipeline.t2i_dataset.T2I_SOURCES`.
    t2i_sources: str = None

    # The training stream (`data_pipeline.stream.build_interleaved_stream`). Keys: row_len, sources [{name, share}],
    # max_area, ar_clamp, max_targets, max_target_latents, supervise_open, report_every.
    interleaved: dict = None

    # Weights of the two losses, each already divided by its own count (supervised tokens, targets).
    w_text: float = 1.0
    w_img: float = 1.0

    # Also train the language model, its head and the vision tokenizer's merger, and add cross-entropy.
    train_language_model: bool = False

    # EMA of the trained weights, the language model and its head excluded (`EMA_EXCLUDE`).
    #   ema_decay      0 = no EMA
    #   ema_warmup     decay min(ema_decay, (1 + step) / (10 + step))
    #   ema_eval       eval on the EMA weights (the trained weights are swapped back afterwards)
    #   ema_eval_live  eval twice: the trained weights as `eval_live_*`, then the EMA as `eval_*`
    ema_decay: float = 0.0
    ema_warmup: bool = False
    ema_eval: bool = True
    ema_eval_live: bool = False

    # The prompt.
    #   system_prompt  the system turn
    #   gen_marker     the text right before a generated image, `{h}` / `{w}` filled with its size; every target's
    #                  condition ends after it
    #   user_template  the user turn around the caption, with `{h}` / `{w}` / `{caption}`
    system_prompt: str = None
    gen_marker: str = None
    user_template: str = None

    # Checkpoint retention (on when all three are set; transformers' own rotation must be off: save_total_limit 0).
    #   keep_latest      the newest N checkpoints are kept whole
    #   keep_steps       older checkpoints survive only at multiples of this, as the model alone; a multiple of
    #                    save_steps
    #   optimizer_steps  older checkpoints at multiples of this also keep their optimizer; a multiple of keep_steps
    keep_latest: int = None
    keep_steps: int = None
    optimizer_steps: int = None

    # `nll_bpd`: Euler steps of the change-of-variables integral (0 = not computed) and Hutchinson samples per step.
    nll_steps: int = 0
    nll_hutchinson: int = 1

    # The eval request: each eval image is put in the context and asked for back with this.
    instruction: str = "Reproduce this image exactly."

    # Training timesteps (t=0 clean, t=1 noise): t = sigmoid(N(logitnorm_mean, logitnorm_std)), then
    # time_shift * t / (1 + (time_shift - 1) * t).
    logitnorm_mean: float = 0.0
    logitnorm_std: float = 1.0
    time_shift: float = 4.0

    # The probability of dropping the last request's caption (classifier-free guidance).
    caption_dropout: float = 0.1

    # FSDP2 keeps the parameters and the gradient reduction in fp32; compute stays bf16.
    fsdp_fp32_params: bool = False

    # On resume, load the weights and the step counter but start the optimizer, scheduler and RNG fresh.
    reset_state: bool = field(default=False,
                              metadata={"help": "Reset optimizer/scheduler/random on resume"})

    output_dir: str = field(default="./ckpts")
    per_device_train_batch_size: int = 4
    remove_unused_columns: bool = False
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "constant"
    accelerator_config: dict = field(default_factory=lambda: {"dispatch_batches": False, "split_batches": False})
    report_to: str = "none"
    eval_strategy: str = "steps"
    eval_steps: int = 5000
    eval_images: int = 64
    per_device_eval_batch_size: int = 8
    ignore_data_skip: bool = True
    save_strategy: str = "steps"
    save_steps: int = 5000
    save_total_limit: int = 5
    max_steps: int = 200000
    logging_steps: int = 50
    max_grad_norm: float = 1.0
    save_safetensors: bool = False

    def __post_init__(self):
        if self.job_name is not None:
            self.run_name = f"{self.job_name}-{self.run_name}"
            output_path = Path(self.output_dir)
            self.output_dir = str(output_path.with_name(f"{self.job_name}-{output_path.name}"))
        super().__post_init__()


# Captions longer than this are cut; only the user turn is shortened.
MAX_CAPTION_TOKENS = 512


def truncate_caption(tokenizer, caption: str) -> str:
    ids = tokenizer(caption, add_special_tokens=False)["input_ids"]
    if len(ids) <= MAX_CAPTION_TOKENS:
        return caption
    return tokenizer.decode(ids[:MAX_CAPTION_TOKENS])


def _apply_time_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Timestep shift s*t/(1+(s-1)t); shift=1.0 is a no-op."""
    return t if shift == 1.0 else shift * t / (1 + (shift - 1) * t)


def sample_timesteps(n, device, dtype, logitnorm_mean, logitnorm_std, time_shift, generator=None):
    """t = sigmoid(N(logitnorm_mean, logitnorm_std)), then the time shift. t=0 is clean, t=1 noise."""
    z = torch.randn(n, device=device, dtype=dtype, generator=generator)
    t = torch.sigmoid(z * logitnorm_std + logitnorm_mean)
    return _apply_time_shift(t, time_shift)


def load_model(args: VisualGenerationArguments):
    """The starting Integral model with the trained modules set: the visual generation model and the vision
    tokenizer's visual generation adapter, with `train_language_model` also the language model, its head and the
    merger. The vision tower and the rest of the vision tokenizer stay frozen (they produce the flow targets)."""
    from integral.modeling_integral import IntegralForConditionalGeneration

    # Flash-attention on both sides: a packed row's sample boundaries reach the language model's full-attention
    # layers as `cu_seq_lens_q/k`, which only the flash path reads; the visual generation model runs varlen over
    # the targets' real patches.
    # A frozen language model and head are loaded in bf16; the model's `_keep_in_fp32_modules_strict` keeps the
    # Gated DeltaNet layers' `A_log` and `norm.weight` (fp32 in Qwen3.5), the vision tokenizer and the visual
    # generation model in fp32, and the rotary buffers stay fp32. A trained language model is loaded fp32 whole, so
    # its FSDP2 shards and AdamW moments are fp32.
    model = IntegralForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.float32 if args.train_language_model else torch.bfloat16,
        attn_implementation={"text_config": "flash_attention_2",
                             "visual_generation_config": "flash_attention_2"})
    vision_tokenizer = model.model.vision_tokenizer
    fp32 = {"vision tokenizer": vision_tokenizer, "visual generation model": model.model.visual_generation_model}
    bad = {name: sorted({str(p.dtype) for p in m.parameters()}) for name, m in fp32.items()
           if any(p.dtype != torch.float32 for p in m.parameters())}
    assert not bad, f"not fp32 after the load: {bad}"
    model.requires_grad_(False)
    model.eval()
    model.model.visual_generation_model.requires_grad_(True)
    vision_tokenizer.visual_generation_adapter.requires_grad_(True)
    if args.train_language_model:
        model.model.language_model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
        vision_tokenizer.merger.requires_grad_(True)
        bad = {p.dtype for m in (model.model.language_model, model.lm_head) for p in m.parameters()}
        assert bad == {torch.float32}, f"the trained language model is {sorted(map(str, bad))}, not fp32"
    n_vis = sum(p.numel() for p in vision_tokenizer.visual.parameters() if p.requires_grad)
    assert n_vis == 0, f"the vision tower is trainable ({n_vis:,} params); it produces the flow target"

    from data_pipeline.packing import assert_packing_supported
    assert_packing_supported(model.config.text_config)
    return model


class VisualGenerationSteps:
    """What the training step and the eval compute around the Integral model: the flow targets, the packed row's
    losses, the eval prompt and the eval metrics.

    Not a module: the model is the trainer's (the FSDP root) and every method takes it. Everything that touches
    its parameters goes through `IntegralForConditionalGeneration.forward`, except the vision tokenizer's
    `encode` (frozen, outside FSDP) and, on a row without an image span, the merger and the adapter after the forward
    (`training_forward_packed`).

    Training reads one packed row per step (`training_row`). Eval reads fixed-resolution image/caption batches with
    the eval image in the context and `instruction` as the text (`eval_loss`, `x0_at`, `flow_vmse`, `flow_bpd`).
    """

    def __init__(self, args: VisualGenerationArguments, model):
        self.args = args
        from transformers import AutoTokenizer

        vision_tokenizer = model.model.vision_tokenizer
        # Cross-entropy reaches a trainable parameter only through the language model, its head or the merger
        # that writes the image embeddings; with all three frozen it is not computed.
        self.ce_on = any(p.requires_grad for m in (model.model.language_model, model.lm_head, vision_tokenizer.merger)
                         for p in m.parameters())

        config = model.config
        vt_config = config.vision_tokenizer_config
        self.patch_size = vt_config.patch_size
        self.merge_size = vt_config.spatial_merge_size
        self.temporal_patch_size = vt_config.temporal_patch_size
        self.image_mean = torch.tensor(vt_config.image_mean)
        self.image_std = torch.tensor(vt_config.image_std)
        self.image_token_id = config.image_token_id

        # the eval prompt: one image of `resolution` in the user turn, then the generation marker
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        assert args.system_prompt and args.gen_marker and args.user_template, \
            "system_prompt, gen_marker and user_template are required"
        self.grid = args.resolution // self.patch_size
        self.n_slots = (self.grid // self.merge_size) ** 2
        self.system_prompt = args.system_prompt
        self._tail_ids = self._marker_ids(args.resolution, args.resolution)
        print(f"[model] {args.model_path}: language model {'trained' if args.train_language_model else 'frozen'}, "
              f"cross-entropy {'on' if self.ce_on else 'off'}")

    def _marker_ids(self, h: int, w: int) -> list:
        """The generation marker's ids at size `h x w`."""
        return self.tokenizer(self.args.gen_marker.format(h=h, w=w), add_special_tokens=False)["input_ids"]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """[..., 3, H, W] in [0, 1] -> the vision tokenizer's input normalization (`image_mean`, `image_std`)."""
        mean = self.image_mean.to(x.device).view(3, 1, 1)
        std = self.image_std.to(x.device).view(3, 1, 1)
        return (x - mean) / std

    def _prefix_ids(self, text: str) -> tuple:
        """-> (the eval prompt's ids, the index its image span starts at).

        system, then user (the image, then `user_template` around the text), the generation prompt, and the
        generation marker. The image placeholder is widened to `n_slots` tokens.
        """
        h = w = self.args.resolution
        body = self.args.user_template.format(h=h, w=w, caption=truncate_caption(self.tokenizer, text))
        msgs = [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": body}]}]
        s = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        n = ids.count(self.image_token_id)
        assert n == 1, f"the template emitted {n} image placeholders, expected 1"
        k = ids.index(self.image_token_id)
        ids = ids[:k] + [self.image_token_id] * self.n_slots + ids[k + 1:]
        return ids + self._tail_ids, k

    def _prompt(self, captions: list, device):
        """-> (ids, attention mask, image span mask), each [B, width], right-padded.

        Padding goes after everything real: between the text and the image span it would be consumed by the
        linear-attention layers as time steps, whatever the mask says.
        """
        built = [self._prefix_ids(c) for c in captions]
        B, width = len(built), max(len(i) for i, _ in built)
        ids = torch.full((B, width), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        attn = torch.zeros(B, width, dtype=torch.long, device=device)
        image = torch.zeros(B, width, dtype=torch.bool, device=device)
        for b, (i, k) in enumerate(built):
            ids[b, :len(i)] = torch.tensor(i, device=device)
            attn[b, :len(i)] = 1
            image[b, k:k + self.n_slots] = True
        return ids, attn, image

    def _check_tokenizer_dtype(self, model):
        # the vision tokenizer produces the flow target; computing it in a lower precision moves the target
        w = next(model.model.vision_tokenizer.parameters())
        assert w.dtype == torch.float32, f"the vision tokenizer is {w.dtype}, not fp32"

    def _targets(self, model, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [0, 1] -> standardized latents [B, gh*gw, latent], row-major."""
        self._check_tokenizer_dtype(model)
        x = self._normalize(pixel_values)
        patches, gh, gw = patchify(x.float(), self.patch_size, self.merge_size, self.temporal_patch_size)
        B = patches.shape[0]
        thw = torch.tensor([[1, gh, gw]] * B, device=x.device)
        z = model.model.vision_tokenizer.encode(patches.reshape(B * gh * gw, -1), thw)
        return z.float().reshape(B, gh * gw, -1)

    def _targets_list(self, model, images: list):
        """uint8 `[3, h, w]` at any sizes -> (standardized latents [total, latent], `(gh, gw)` per image).

        One vision tokenizer forward over the distinct images. The latents are flat, in image order, row-major
        within each image. An image that repeats an earlier one (same size, same pixels; a reconstruction sample
        has its image twice) reuses that one's latents.
        """
        self._check_tokenizer_dtype(model)
        distinct, index = [], []
        for im in images:
            j = next((u for u, seen in enumerate(distinct)
                      if seen.shape == im.shape and torch.equal(seen, im)), None)
            if j is None:
                distinct.append(im)
                j = len(distinct) - 1
            index.append(j)
        patches, grids = [], []
        for im in distinct:
            # `images` are views on the uint8 buffer: 0..255, so `/ 255` before the normalization
            x = self._normalize(im.float() / 255.0)
            f, gh, gw = patchify(x[None], self.patch_size, self.merge_size, self.temporal_patch_size)
            patches.append(f[0])
            grids.append((gh, gw))
        thw = torch.tensor([[1, gh, gw] for gh, gw in grids], device=patches[0].device)
        z = model.model.vision_tokenizer.encode(torch.cat(patches), thw).float()
        if len(distinct) == len(images):
            return z, grids
        off = image_offsets(grids)
        return torch.cat([z[off[j]:off[j + 1]] for j in index]), [grids[j] for j in index]

    def training_row(self, model, fields: dict):
        """One packed row -> `(ce_sum, ce_count, flow_sum, flow_count)`.

        `fields` is a `PackedBatch` as a dict, on this rank's device, with the pixels as uint8. The vision
        tokenizer encodes every image of the row (context and target) in fp32 outside autocast; the model runs
        under bf16 autocast, which FSDP2 does not install by itself.
        """
        batch = PackedBatch.from_dict(fields)
        with torch.no_grad():
            latents, grids = self._targets_list(model, batch.images())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(self.args.bf16)):
            return self.training_forward_packed(model, batch, latents, grids)

    def training_forward_packed(self, model, batch, latents: torch.Tensor, grids: list, *, generator=None):
        """One packed row -> `(ce_sum, ce_count, flow_sum, flow_count)`.

        Args:
            model: the Integral model.
            batch: the packed row.
            latents: `[total_latents, latent]`, every image of the row in sequence order.
            grids: image i's `(gh, gw)`.
            generator: the RNG for the noise and the timesteps (default: the global one).

        Sums and counts, not means: the trainer divides by the counts summed over the ranks and the accumulation
        window. The language model runs once; every target reads its key/values over `[sample start, gen_point)`.
        """
        n_tgt = batch.n_targets
        assert len(grids) > 0 and latents.dim() == 2, (
            f"latents must be flat [total, dim] with one grid per image; got "
            f"{tuple(latents.shape)} and {len(grids)} grids")
        device = latents.device
        off = image_offsets(grids)
        gi = batch.gen_image.tolist()
        assert len(gi) == n_tgt, f"{len(gi)} target images for {n_tgt} targets"

        # The targets padded to the longest one; `keep` marks the real columns. Target r's column c is latent
        # `off[gi[r]] + c` where `c < n_lat[r]`, and zero past it.
        tgt_grids = [grids[i] for i in gi]
        n_lat = torch.tensor([gh * gw for gh, gw in tgt_grids], device=device)
        T = int(n_lat.max())
        starts = torch.tensor([off[i] for i in gi], device=device)
        cols = torch.arange(T, device=device)
        keep = cols[None, :] < n_lat[:, None]
        zk = torch.where(keep[..., None], latents[torch.where(keep, starts[:, None] + cols[None, :], 0)],
                         latents.new_zeros(()))

        eps = torch.randn(zk.shape, device=zk.device, dtype=zk.dtype, generator=generator)
        t = sample_timesteps(n_tgt, zk.device, zk.dtype, self.args.logitnorm_mean, self.args.logitnorm_std,
                             self.args.time_shift, generator)
        x_t = (1 - t.view(-1, 1, 1)) * zk + t.view(-1, 1, 1) * eps
        x_t = x_t * keep.unsqueeze(-1)          # padding stays exactly zero going in

        slots = batch.image_slots
        # The model finds image positions by token id. A span cut at the row boundary leaves `<|image_pad|>` tokens
        # outside `image_slots` at the end of its sample; nothing reads them, so they become padding.
        ids = batch.input_ids.masked_fill((batch.input_ids == self.image_token_id) & ~slots,
                                          self.tokenizer.pad_token_id)
        span_latents = slot_latents(batch, latents, grids)
        image_inputs = {}
        if span_latents:
            image_inputs = dict(image_latents=torch.cat(span_latents),
                                image_grid_thw=torch.cat([torch.ones_like(batch.slot_grid[:, :1]), batch.slot_grid],
                                                         dim=-1))
        # Position i's hidden state predicts `labels[i + 1]`; every sample's first label is IGNORE, so nothing is
        # predicted across a packing boundary. The head runs on the supervised positions only (`logits_to_keep`;
        # none without cross-entropy).
        assert batch.input_ids.shape[0] == 1, f"a packed batch is one row, got {batch.input_ids.shape[0]}"
        tgt_ids = batch.labels.to(device)[:, 1:].reshape(-1)
        sup = ((tgt_ids != IGNORE_INDEX).nonzero(as_tuple=True)[0] if self.ce_on
               else torch.zeros(0, dtype=torch.long, device=device))

        out = model(input_ids=ids,
                    position_ids=sample_position_ids(batch, model.model.get_rope_index),
                    mm_token_type_ids=slots.int(),
                    visual_noisy_latents=x_t, visual_timesteps=t,
                    visual_grid_thw=torch.tensor([[1, gh, gw] for gh, gw in tgt_grids], device=device),
                    visual_batch_index=torch.zeros(n_tgt, dtype=torch.long, device=device),
                    visual_attention_mask=target_condition(batch),
                    logits_to_keep=sup, use_cache=False, **image_inputs, **packing_kwargs(batch))
        pred = out.visual_velocity
        target = eps - zk
        # The padding is not a target: both sides are zeroed there.
        pred, target = pred * keep.unsqueeze(-1), target * keep.unsqueeze(-1)
        # Per-target mean over that target's own latents, summed over targets; the count is the target count.
        se = ((pred - target).pow(2) * keep.unsqueeze(-1)).sum(dim=(1, 2))
        flow_sum = (se / (keep.sum(1) * target.shape[-1]).to(se.dtype)).sum()
        flow_count = int(target.shape[0])
        if not image_inputs:
            # A row with no image span reaches neither the merger nor the adapter, but every rank has to take part
            # in their FSDP2 gradient reduce-scatter: both run on image 0 and are added times zero. After the forward,
            # whose root hook has gathered their parameters (the root does not reshard after forward).
            vision_tokenizer = model.model.vision_tokenizer
            gh0, gw0 = grids[0]
            z0 = latents[:gh0 * gw0]
            thw0 = torch.tensor([[1, gh0, gw0]], device=device)
            touch = vision_tokenizer.merge(z0, thw0).sum() + sum(
                k.sum() + v.sum() for k, v in vision_tokenizer.adapt(z0, thw0))
            flow_sum = flow_sum + 0.0 * touch

        if sup.numel() == 0:
            return flow_sum.new_zeros(()), 0, flow_sum, flow_count
        ce_sum = F.cross_entropy(out.logits[0].float(), tgt_ids.index_select(0, sup), reduction="sum",
                                 ignore_index=IGNORE_INDEX)
        return ce_sum, int(sup.numel()), flow_sum, flow_count

    def _denoise(self, model, x_t: torch.Tensor, t: torch.Tensor, captions: list,
                 read: torch.Tensor) -> torch.Tensor:
        """The velocity at `x_t` [B, gh*gw, latent], with `read` [B, gh*gw, latent] in the context. No guidance."""
        B = x_t.shape[0]
        device = x_t.device
        ids, attn, image = self._prompt(captions, device)
        thw = torch.tensor([[1, self.grid, self.grid]] * B, device=device)
        with torch.no_grad():
            pos, _ = model.model.get_rope_index(ids, image.int(), image_grid_thw=thw, attention_mask=attn)
        out = model(input_ids=ids, attention_mask=attn, position_ids=pos, mm_token_type_ids=image.int(),
                    image_latents=read.reshape(-1, read.shape[-1]), image_grid_thw=thw,
                    visual_noisy_latents=x_t, visual_timesteps=t, visual_grid_thw=thw,
                    logits_to_keep=torch.zeros(0, dtype=torch.long, device=device), use_cache=False)
        return out.visual_velocity

    @torch.no_grad()
    def x0_at(self, model, z: torch.Tensor, captions: list, t: float = 1.0,
              read: torch.Tensor = None) -> torch.Tensor:
        """One denoise of a noised real latent -> the x0 estimate. t=1.0 is pure noise, where sampling starts."""
        B = z.shape[0]
        x_t = (1 - t) * z + t * torch.randn_like(z)
        pred = self._denoise(model, x_t, z.new_full((B,), t), list(captions), read)
        return x_t - t * pred

    @torch.no_grad()
    def flow_vmse(self, model, z: torch.Tensor, captions: list, n_t: int = 8, read: torch.Tensor = None) -> float:
        """Velocity MSE over a fixed uniform timestep grid. No ODE, no guidance, no time shift."""
        B = z.shape[0]
        caps = list(captions)
        total = 0.0
        for t in torch.linspace(0.05, 0.95, n_t, device=z.device, dtype=z.dtype):
            eps = torch.randn_like(z)
            x_t = (1 - t) * z + t * eps
            v_hat = self._denoise(model, x_t, t.expand(B), caps, read).float()
            total += (v_hat - (eps - z).float()).pow(2).mean().item()
        return total / n_t

    @torch.no_grad()
    def flow_bpd(self, model, z: torch.Tensor, captions: list, read: torch.Tensor = None,
                 steps: int = 32, n_hutch: int = 1, eps: float = 1e-3) -> float:
        """Bits per dimension of the standardized latent `z` under the model's flow (can be negative).

        The change of variables d/dt log p(x_t) = -div v(x_t, t) is integrated with `steps` Euler steps from the
        data (t=0) to the noise (t=1), where p_1 = N(0, I). The divergence is Hutchinson's estimator with `n_hutch`
        Rademacher vectors and the directional derivative by central differences of size `eps`, so it needs model
        forwards only (`1 + 2 * n_hutch` per step). No guidance, no time shift.
        """
        caps = list(captions)
        B = z.shape[0]
        x = z.float()
        D = x[0].numel()
        logdet = torch.zeros(B, device=x.device, dtype=torch.float32)
        grid = torch.linspace(0.0, 1.0, steps + 1, device=x.device, dtype=torch.float32)

        def field(xi: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self._denoise(model, xi.to(z.dtype), t.expand(B).to(z.dtype), caps, read).float()

        for i in range(steps):
            t0 = grid[i]
            dt = grid[i + 1] - t0
            v = field(x, t0)
            div = torch.zeros(B, device=x.device, dtype=torch.float32)
            for _ in range(n_hutch):
                eta = torch.randint(0, 2, x.shape, device=x.device, dtype=torch.float32) * 2 - 1
                jvp = (field(x + eps * eta, t0) - field(x - eps * eta, t0)) / (2 * eps)
                div = div + (jvp * eta).flatten(1).sum(1)
            logdet = logdet + dt * (div / n_hutch)
            x = x + dt * v
        # p_1 = N(0, I) at the end of the trajectory
        log_p1 = -0.5 * (x.flatten(1).pow(2).sum(1) + D * math.log(2 * math.pi))
        return float((-(log_p1 + logdet) / (D * math.log(2))).mean().item())

    def eval_loss(self, model, pixel_values: torch.Tensor, captions: list):
        """The flow loss of an eval batch of `pixel_values` [B, 3, H, W] and `captions` at one training-distribution
        timestep per image, with the image itself in the context."""
        with torch.no_grad():
            z = self._targets(model, pixel_values)
        B = z.shape[0]
        eps = torch.randn(z.shape, device=z.device, dtype=z.dtype)
        t = sample_timesteps(B, z.device, z.dtype, self.args.logitnorm_mean, self.args.logitnorm_std,
                             self.args.time_shift)
        x_t = (1 - t.view(B, 1, 1)) * z + t.view(B, 1, 1) * eps
        pred = self._denoise(model, x_t, t, list(captions), z)
        flow_loss = (pred - (eps - z)).pow(2).mean()
        return flow_loss, flow_loss.detach()


def collate_fn(batch):
    """Eval examples -> `pixel_values` [B, 3, H, W] and `captions`."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }


def _packed_to_dict(batch):
    """A `PackedBatch` -> the dict `Trainer._prepare_inputs` moves to the device (module level: picklable)."""
    return batch.as_dict()


# The prefix activation checkpointing inserts into a parameter's name.
_NAME_WRAPPERS = ("_checkpoint_wrapped_module.",)


def _clean_param_name(name: str) -> str:
    """A state-dict key as the unwrapped model spells it."""
    for junk in _NAME_WRAPPERS:
        name = name.replace(junk, "")
    return name


def ema_decay_at(decay: float, step: int, warmup: bool) -> float:
    """The EMA decay at EMA step `step`: `decay`, or with warmup `min(decay, (1 + step) / (10 + step))`."""
    return decay if not warmup else min(decay, (1.0 + step) / (10.0 + step))


def ema_reshard(model) -> int:
    """Reshard every FSDP2 module (`FSDPModule.reshard()`, not recursive, no collective). Returns the count.

    With `reshard_after_forward` false, an eval forward (no backward) leaves the parameters unsharded: the module
    holds a tensor over the all-gather buffer instead of the sharded DTensor the optimizer steps and `state_dict()`
    reads. The EMA swap has to write the sharded parameters, so this runs before it.
    """
    from torch.distributed.fsdp import FSDPModule

    n = 0
    for m in model.modules():
        if isinstance(m, FSDPModule):
            m.reshard()
            n += 1
    return n


def ema_copy_(dst, src) -> None:
    """`dst.copy_(src)` for two tensors of the same layout (both DTensors or both plain); raises otherwise."""
    from torch.distributed.tensor import DTensor

    if isinstance(dst, DTensor) != isinstance(src, DTensor):
        raise RuntimeError(
            f"EMA copy across layouts: dst is {type(dst).__name__}, src is {type(src).__name__}. "
            "The parameters were not resharded before the swap, and a copy that crosses this "
            "boundary does not reach the weights the optimizer holds.")
    dst.copy_(src.to(dst.dtype))


@torch.no_grad()
def ema_apply(ema: dict, named_params, decay: float) -> None:
    """One EMA step in place, on each rank's own shards (`lerp_` of same-placement DTensors, no communication)."""
    for n, p in named_params:
        ema[n].lerp_(p.detach().float(), 1.0 - decay)


def _resume_step(resume_from_checkpoint) -> int:
    """The `global_step` in the checkpoint's `trainer_state.json`, or 0.

    A resume does not fast-forward the data stream (`ignore_data_skip`); the stream starts again with
    `seed + this step`.
    """
    if not resume_from_checkpoint:
        return 0
    state = os.path.join(str(resume_from_checkpoint), "trainer_state.json")
    if not os.path.exists(state):
        return 0
    with open(state) as fp:
        return int(json.load(fp).get("global_step", 0))


def _patch_fsdp_state_dict_options():
    """Save and load only the trainable parameters in FSDP checkpoints (the frozen ones come from `model_path`).

    `accelerate.utils.fsdp_utils._prepare_sd_options` builds the state-dict options of all four FSDP checkpoint
    paths (save / load, model / optimizer); it is wrapped to add `ignore_frozen_params=True` and `strict=False`.
    """
    from accelerate.utils import fsdp_utils

    if getattr(fsdp_utils._prepare_sd_options, "_trainable_only", False):
        return
    original = fsdp_utils._prepare_sd_options

    def trainable_only(fsdp_plugin):
        return dataclasses.replace(original(fsdp_plugin), ignore_frozen_params=True, strict=False)

    trainable_only._trainable_only = True
    fsdp_utils._prepare_sd_options = trainable_only


class _StepInjector:
    """The wandb module with `log` bound to one step. Everything else is forwarded untouched."""

    def __init__(self, wb, step: int):
        self._wb, self._step = wb, step

    def log(self, data, **kw):
        kw.setdefault("step", self._step)
        return self._wb.log(data, **kw)

    def __getattr__(self, name):
        return getattr(self._wb, name)


def _attach_wandb_step(handler):
    """Make transformers' wandb callback log at the training step.

    `WandbCallback.on_log` calls `wandb.log` without a step, so wandb's own step counts calls. For the duration of
    each `on_log` the callback's wandb handle is replaced by one whose `log` passes `step=state.global_step`.
    """
    try:
        from transformers.integrations import WandbCallback
    except ImportError:
        return
    for cb in handler.callbacks:
        if isinstance(cb, WandbCallback) and not getattr(cb, "_step_attached", False):
            cb._step_attached = True
            inner = cb.on_log

            def on_log(args, state, control, model=None, logs=None, _cb=cb, _inner=inner, **kw):
                if _cb._wandb is None:
                    return
                if not _cb._initialized:          # setup() must run against the real module
                    _cb.setup(args, state, model)
                real = _cb._wandb
                _cb._wandb = _StepInjector(real, state.global_step)
                try:
                    return _inner(args, state, control, model=model, logs=logs, **kw)
                finally:
                    _cb._wandb = real

            cb.on_log = on_log


class VisualGenerationTrainer(CustomLoggingTrainer, TrainerCallback):
    args: VisualGenerationArguments

    def __init__(self, args: VisualGenerationArguments):
        from accelerate import PartialState

        _patch_fsdp_state_dict_options()

        # A resume restarts the data stream with `seed + step`. Set before the stream is built and before
        # `Trainer.__init__` seeds from `args.seed`.
        say = print if args.process_index == 0 else (lambda *a, **k: None)

        step = _resume_step(args.resume_from_checkpoint)
        if step:
            args.seed += step
            say(f"[resume] step {step}: seed {args.seed - step} -> {args.seed}", flush=True)

        # Checkpoint retention (`on_save`); `_keep_full` is the number of newest checkpoints kept whole.
        self._save_failures = 0
        self._keep_full = None
        # The EMA is allocated on the first optimizer step, once the parameters are FSDP2 DTensors (`_init_ema`).
        # `_ema_backup` holds the trained weights while eval runs on the EMA.
        self._ema = None
        self._ema_backup = None
        self._ema_steps = 0
        if args.keep_latest or args.keep_steps or args.optimizer_steps:
            assert args.save_strategy == "steps" and args.save_steps, \
                "keep_latest / keep_steps / optimizer_steps need step-based saving"
            assert args.keep_latest and args.keep_steps and args.optimizer_steps, \
                "keep_latest, keep_steps and optimizer_steps are set together"
            # transformers' own rotation deletes checkpoints inside `_save_checkpoint`, before `on_save`. It is off
            # at `save_total_limit: 0` (a YAML `null` parses back to the default).
            assert not args.save_total_limit, (
                f"save_total_limit is {args.save_total_limit}; set it to 0 with keep_latest / keep_steps / "
                f"optimizer_steps, or transformers deletes the checkpoints they keep")
            assert args.keep_steps % args.save_steps == 0, (
                f"keep_steps {args.keep_steps} is not a multiple of save_steps {args.save_steps}, "
                f"so no written checkpoint is ever on the keep grid and every older one would go")
            assert args.optimizer_steps % args.keep_steps == 0, (
                f"optimizer_steps {args.optimizer_steps} is not a multiple of keep_steps "
                f"{args.keep_steps}, so no surviving checkpoint would ever keep its optimizer")
            self._keep_full = max(1, args.keep_latest)
            say(f"[ckpt] written every {args.save_steps} steps; the newest {self._keep_full} stay "
                f"resumable; older ones survive only on the {args.keep_steps} grid, as the model "
                f"alone, and keep their optimizer on the {args.optimizer_steps} grid.", flush=True)

        assert args.t2i_sources, "t2i_sources (the eval set) is required"
        # The loader yields one packed row at a time; `interleaved.row_len` sizes a step.
        assert args.per_device_train_batch_size == 1, (
            f"per_device_train_batch_size is {args.per_device_train_batch_size}; a packed row is the microbatch, "
            f"so it must be 1 (`interleaved.row_len` sizes a step)")
        assert args.system_prompt and args.gen_marker and args.user_template, \
            "system_prompt, gen_marker and user_template are required"
        sources = [s.strip() for s in args.t2i_sources.split(",") if s.strip()]
        eval_ds = build_t2i_eval_set(sources, args.eval_images, resolution=args.resolution)
        from transformers import AutoTokenizer

        from integral.configuration_integral import IntegralConfig
        # The prompt keys are top-level arguments shared by the training stream and the eval prompt.
        owned = ({"gen_marker", "user_template", "caption_dropout", "instruction", "system",
                  "resolution", "text_loss"} & set(args.interleaved))
        assert not owned, (
            f"`interleaved` names {sorted(owned)}; these are set from the top-level arguments (gen_marker, "
            f"user_template, caption_dropout, instruction, system_prompt, resolution) or from train_language_model "
            f"(text_loss). Remove them from the block.")
        cfg = dict(args.interleaved,
                   gen_marker=args.gen_marker,
                   user_template=args.user_template,
                   caption_dropout=args.caption_dropout,
                   instruction=args.instruction,
                   system=args.system_prompt,
                   resolution=args.resolution,
                   # without cross-entropy each sample is cut after its last target's condition
                   text_loss=args.train_language_model)
        # With the language model trained, `lm_head`'s only gradient is cross-entropy: a row with no supervised
        # token would skip its FSDP2 reduce-scatter. `supervise_open` labels every generation marker.
        assert cfg.get("supervise_open", True) or not args.train_language_model, (
            "interleaved.supervise_open: false with train_language_model: a row of image-only turns has no "
            "cross-entropy and its rank skips lm_head's gradient reduce-scatter. Set supervise_open: true.")
        geom = Geometry(patch=IntegralConfig.from_pretrained(args.model_path).vision_tokenizer_config.patch_size)
        train_ds = build_interleaved_stream(
            cfg, AutoTokenizer.from_pretrained(args.model_path),
            args.seed, geom=geom)
        where = (f"interleaved ({len(cfg['sources'])} sources), "
                 f"row {cfg['row_len']} tokens, eval={','.join(sources)}")
        # Every rank reads the same model files; rank 0 goes first.
        with PartialState().main_process_first():
            model = load_model(args)
        self._steps = VisualGenerationSteps(args, model)
        # saved with every checkpoint (`Trainer._save`), next to the model's `save_pretrained` output
        from integral.processing_integral import IntegralProcessor
        processor = IntegralProcessor.from_pretrained(args.model_path)
        super().__init__(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         data_collator=collate_fn, processing_class=processor, callbacks=[self])
        if args.fsdp_fp32_params:
            self._force_fp32_fsdp_policy()
        # Per-step loss counts (supervised tokens, targets) over all ranks and the accumulation window; see
        # `_get_num_items_in_batch`. `model_accepts_loss_kwargs` stops `training_step` dividing by the accumulation steps.
        self._ce_items = self._flow_items = None
        # [flow_sum, ce_sum, flow_count, ce_count, rows] since the last training log; see `log`
        self._loss_sums = None
        self._ce_on = bool(self._steps.ce_on)
        assert self._ce_on == cfg["text_loss"], (
            f"the stream cuts samples for text_loss={cfg['text_loss']} but the model's cross-entropy "
            f"is {'on' if self._ce_on else 'off'}")
        self.model_accepts_loss_kwargs = True
        _attach_wandb_step(self.callback_handler)
        if self.is_world_process_zero():
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"trainable: {n / 1e6:.1f}M  | data={where}")
            os.makedirs(self.args.output_dir, exist_ok=True)
            cp = os.path.join(self.args.output_dir, "training_config.yaml")
            if not os.path.exists(cp):
                with open(cp, "w") as fp:
                    args.dump_yaml(fp)

    def get_train_dataloader(self):
        """A plain DataLoader over the stream, not passed through `accelerator.prepare`.

        The stream splits itself across ranks and dataloader workers. accelerate would treat an IterableDataset as
        the same stream on every rank and either dispatch rank 0's batches or let every rank build and drop the
        others' samples. The batch reaches the device through `Trainer._prepare_inputs`.
        """
        from torch.utils.data import DataLoader

        # The stream splits on the global rank, which equals accelerate's data-parallel index only without tensor or
        # context parallelism.
        mesh = getattr(self.accelerator.state, "device_mesh", None)
        dims = tuple(getattr(mesh, "mesh_dim_names", ()) or ())
        assert not ({"tp", "cp"} & set(dims)), (
            f"the loader shards on the GLOBAL rank, which stops matching accelerate's "
            f"data-parallel index once the mesh has tp or cp; mesh dims are {dims}")

        kwargs = {
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        # `Trainer._prepare_inputs` moves the tensors of dicts and lists, so the row crosses as a dict.
        kwargs["collate_fn"] = _packed_to_dict
        return self._check_unwrapped(DataLoader(self.train_dataset, batch_size=None, **kwargs))

    def _get_num_items_in_batch(self, batch_samples, device):
        """[supervised tokens, flow targets] of the accumulation window, summed over the ranks.

        Read off the rows before any forward: the labels the shift scores (`labels[..., 1:]`) and the targets
        (`gen_point`). `compute_loss` divides each loss by its count; the gradient reduction averages over the ranks,
        so every token and every target weighs the same wherever it landed.
        """
        ce = (sum(int((b["labels"][..., 1:] != IGNORE_INDEX).sum()) for b in batch_samples)
              if self._ce_on else 0)
        tgt = sum(int(b["gen_point"].numel()) for b in batch_samples)
        counts = torch.tensor([float(ce), float(tgt)], device=device)
        if self.args.world_size > 1:
            counts = self.accelerator.reduce(counts, reduction="sum")
        # the packer puts at least one target in every row
        assert counts[1] > 0, "the accumulation window holds no flow target on any rank"
        # every target's generation marker is supervised
        assert not self._ce_on or counts[0] > 0, \
            "the accumulation window holds no supervised token on any rank"
        self._ce_items, self._flow_items = counts[0], counts[1]
        return counts

    def _load_optimizer_and_scheduler(self, checkpoint):
        if self.args.reset_state:
            return

        return super()._load_optimizer_and_scheduler(checkpoint)

    def _load_rng_state(self, checkpoint):
        if self.args.reset_state:
            return

        return super()._load_rng_state(checkpoint)

    def _force_fp32_fsdp_policy(self):
        """Set the FSDP2 mixed-precision policy's gradient reduction to fp32, the rest unchanged.

        The parameters stay bf16 when all-gathered (flash-attention takes bf16 / fp16 only); the sharded master
        parameters keep the module's own dtype (fp32 for every trained module).
        """
        from torch.distributed.fsdp import MixedPrecisionPolicy

        plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if plugin is None:
            print("[fsdp_fp32_params] no FSDP plugin — ignored", flush=True)
            return
        before = plugin.mixed_precision_policy
        if before is None:
            print("[fsdp_fp32_params] no mixed-precision policy — nothing to raise", flush=True)
            return
        plugin.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype=before.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=before.output_dtype,
            cast_forward_inputs=before.cast_forward_inputs)
        if self.is_world_process_zero():
            print(f"[fsdp_fp32_params] {before}\n                -> "
                  f"{plugin.mixed_precision_policy}", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        """After accelerate has applied FSDP2 (and before the first eval or step): `_uncast_root`."""
        super().on_train_begin(args, state, control, **kwargs)
        self._uncast_root()

    def _uncast_root(self):
        """Stop the FSDP2 root (the Integral model) from casting its forward inputs and outputs.

        The mixed-precision policy casts every FSDP unit's floating-point forward inputs to bf16
        (`cast_forward_inputs`) and its outputs to `output_dtype`. The root's forward is the model's own, so the fp32
        timesteps, noisy latents and image latents the step builds would reach the model in bf16 (the timestep
        embedding takes `sin / cos(1000 t)` of the rounded t), and the velocity would leave it in bf16. The wrapped
        blocks keep the policy. accelerate gives every unit the same policy, so the root's is replaced on its FSDP
        state, a private attribute of torch's FSDP2 (`FSDPModule._get_fsdp_state()._mp_policy`, read only by the
        input / output casts; the parameter casts are the parameter group's).
        """
        if not self.is_fsdp_enabled:
            return
        from torch.distributed.fsdp import FSDPModule

        assert isinstance(self.model, FSDPModule), f"the FSDP root is {type(self.model).__name__}"
        state = self.model._get_fsdp_state()
        before = state._mp_policy
        state._mp_policy = dataclasses.replace(before, cast_forward_inputs=False, output_dtype=None)
        if self.is_world_process_zero():
            print(f"[fsdp root] {type(self.model).__name__}: {before}\n                -> {state._mp_policy}",
                  flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        """Set `epoch` to `global_step / max_steps` (the stream has no length, so one epoch is the whole run; the
        trainer's own value restarts at a resume), then take the EMA step."""
        if args.max_steps > 0:
            state.epoch = state.global_step / args.max_steps
        self._ema_update()

    # EMA of the trained weights: the visual generation model and the vision tokenizer's trained modules (the
    # adapter, and the merger with `train_language_model`), never the language model or its head.
    EMA_EXCLUDE = ("model.language_model.", "lm_head.")

    def _ema_params(self):
        """`(name, parameter)` of the parameters the EMA covers, off the unwrapped model.

        The update, the swap, the save and the load all use this set. The names are cleaned of activation
        checkpointing's wrapper prefix, so they match the saved EMA file across resumes.
        """
        model = self.accelerator.unwrap_model(self.model)
        trainable = [(_clean_param_name(n), p) for n, p in model.named_parameters()
                     if p.requires_grad]
        return [(n, p) for n, p in trainable if not n.startswith(self.EMA_EXCLUDE)]

    def _ema_decay_now(self) -> float:
        return ema_decay_at(self.args.ema_decay, self._ema_steps, self.args.ema_warmup)

    def _ema_reshard_checked(self) -> int:
        """`ema_reshard`, then check that every EMA parameter is an FSDP2 DTensor. Returns the module count."""
        n_fsdp = ema_reshard(self.model)
        if not self.is_fsdp_enabled:
            return n_fsdp
        from torch.distributed.tensor import DTensor

        stray = [n for n, p in self._ema_params() if not isinstance(p, DTensor)]
        if not n_fsdp or stray:
            raise RuntimeError(
                f"[ema] the reshard did not take: {n_fsdp} FSDP modules walked, {len(stray)} of "
                f"{len(self._ema_params())} parameters still unsharded (first: {stray[:3]}); "
                "a swap now would not reach the sharded parameters.")
        return n_fsdp

    @torch.no_grad()
    def _ema_update(self):
        """One EMA step on this rank's shards. The first call allocates the EMA (or loads it on resume)."""
        if not self.args.ema_decay:
            return
        if self._ema is None:
            # Allocated on the first optimizer step: the parameters are sharded DTensors by now, so the EMA is
            # sharded the same way and `_ema_load` can distribute a saved one onto this run's mesh.
            n_fsdp = self._ema_reshard_checked()
            if self.args.resume_from_checkpoint:
                self._ema_load(self.args.resume_from_checkpoint)
            if self._ema is None:
                self._ema = {n: p.detach().clone().float() for n, p in self._ema_params()}
            # `numel()` of a DTensor is the global count; this rank holds its local shard
            g = sum(t.numel() for t in self._ema.values())
            loc = sum((t.to_local() if hasattr(t, "to_local") else t).numel()
                      for t in self._ema.values())
            if self.args.process_index == 0:
                n_train = sum(p.numel() for p in
                              self.accelerator.unwrap_model(self.model).parameters()
                              if p.requires_grad)
                print(f"[ema] decay={self.args.ema_decay} warmup={self.args.ema_warmup} "
                      f"eval={self.args.ema_eval}: "
                      f"{len(self._ema)} tensors, {g:,} parameters, "
                      f"{loc:,} on this rank ({loc * 4 / 2**30:.2f} GiB fp32), "
                      f"reshard covers {n_fsdp} FSDP modules", flush=True)
                # the EMA set by module: `model.<module>`, one level further under `vision_tokenizer`
                by_group = {}
                for n, p in self._ema_params():
                    parts = n.split(".")
                    top = (parts[2] if parts[:2] == ["model", "vision_tokenizer"] else
                           parts[1] if parts[0] == "model" else parts[0])
                    by_group[top] = by_group.get(top, 0) + p.numel()
                groups = ", ".join(f"{k} {v / 1e6:.1f}M"
                                   for k, v in sorted(by_group.items(), key=lambda x: -x[1]))
                print(f"[ema] covers {g / max(n_train, 1):.1%} of the {n_train:,} trainable "
                      f"parameters: {groups} -- the language model and its head are excluded by name.", flush=True)
        ema_apply(self._ema, self._ema_params(), self._ema_decay_now())
        self._ema_steps += 1

    @torch.no_grad()
    def _ema_swap_in(self) -> bool:
        """Trained weights -> backup, EMA -> parameters (after a reshard). Returns whether anything was swapped.

        Used by eval (when `ema_eval`) and by the EMA save.
        """
        if not self.args.ema_decay or self._ema is None:
            return False
        self._ema_reshard_checked()
        self._ema_backup = {}
        for n, p in self._ema_params():
            self._ema_backup[n] = p.detach().clone()
            ema_copy_(p.detach(), self._ema[n])
        return True

    @torch.no_grad()
    def _ema_swap_out(self):
        """Backup -> parameters, then check each local shard equals its backup (per rank, no collective)."""
        if self._ema_backup is None:
            return
        self._ema_reshard_checked()
        bad = []
        for n, p in self._ema_params():
            ema_copy_(p.detach(), self._ema_backup[n])
            now, was = p.detach(), self._ema_backup[n]
            now = now.to_local() if hasattr(now, "to_local") else now
            was = was.to_local() if hasattr(was, "to_local") else was
            if not torch.equal(now, was):
                bad.append(n)
        self._ema_backup = None
        if bad:
            raise RuntimeError(
                f"[ema] the swap-out did not restore {len(bad)} of {len(self._ema)} parameters "
                f"(first: {bad[:3]}); the model holds the EMA weights.")

    def evaluate(self, *a, **kw):
        """One eval pass, or with `ema_eval_live` two: the trained weights as `eval_live_*`, then the EMA as `eval_*`."""
        if not (self.args.ema_eval_live and self.args.ema_eval and self.args.ema_decay):
            return self._evaluate_swapped(*a, **kw)
        assert len(a) < 3, \
            "metric_key_prefix arrived positionally; the two passes set it by keyword"
        live = super().evaluate(*a, **{**kw, "metric_key_prefix": "eval_live"})
        ema = self._evaluate_swapped(*a, **kw)
        return {**live, **ema}

    def _evaluate_swapped(self, *a, **kw):
        """Eval on the EMA weights (when `ema_eval`); the trained weights are restored afterwards, also on failure.

        The swap copies in both directions; the parameters are resharded first so the copy lands in the sharded
        parameters.
        """
        swapped = self._ema_swap_in() if self.args.ema_eval else False
        if self.args.ema_decay and not swapped and self.args.process_index == 0:
            print(f"[ema] eval at step {self.state.global_step} did NOT swap "
                  f"(ema_eval={self.args.ema_eval}, ema allocated={self._ema is not None})",
                  flush=True)
        if swapped and self.args.process_index == 0:
            # Mean |EMA - trained| over rank 0's local shards. Local only: a reduction over a DTensor is a
            # collective, which only rank 0 would issue here.
            def _local(t):
                return t.to_local() if hasattr(t, "to_local") else t

            gap = sum(_local(self._ema[n] - self._ema_backup[n].float()).abs().mean().item()
                      for n, _ in self._ema_params()) / max(len(self._ema), 1)
            print(f"[ema] eval at step {self.state.global_step} on the averaged weights "
                  f"(rank0-shard mean |ema - live| = {gap:.3e}, decay {self._ema_decay_now()})",
                  flush=True)
        try:
            return super().evaluate(*a, **kw)
        finally:
            if swapped:
                self._ema_swap_out()

    def _ema_save(self, checkpoint_dir: str):
        """Write the model with the EMA weights to `<checkpoint>/ema/` (swapped in, then `save_model`); the weights
        outside the EMA set are the current ones."""
        if not self.args.ema_decay or self._ema is None:
            return
        d = os.path.join(checkpoint_dir, "ema")
        swapped = self._ema_swap_in()
        try:
            self.save_model(d, _internal_call=True)
        finally:
            if swapped:
                self._ema_swap_out()
        if self.args.process_index == 0:
            print(f"[ema] wrote {d} at step {self.state.global_step}", flush=True)

    @torch.no_grad()
    def _ema_load(self, checkpoint_dir: str):
        """Load `<checkpoint>/ema/` into this rank's shards. Without one, the EMA starts from the resumed weights."""
        if not self.args.ema_decay:
            return
        from torch.distributed.tensor import DTensor, distribute_tensor

        from safetensors import safe_open

        say = print if self.args.process_index == 0 else (lambda *a, **k: None)
        d = os.path.join(checkpoint_dir, "ema")
        index = os.path.join(d, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index) as fp:
                files = sorted(set(json.load(fp)["weight_map"].values()))
        elif os.path.isfile(os.path.join(d, "model.safetensors")):
            files = ["model.safetensors"]
        else:
            say(f"[ema] no EMA in {d}: the average starts from the resumed weights", flush=True)
            return
        # The directory is a saved Integral model, keyed like the EMA set; only the EMA set's tensors are read.
        want = {n for n, _ in self._ema_params()}
        full = {}
        for f in files:
            with safe_open(os.path.join(d, f), framework="pt") as fh:
                for k in fh.keys():
                    if k in want:
                        full[k] = fh.get_tensor(k)
        ema, missing = {}, []
        for n, p in self._ema_params():
            if n not in full:
                missing.append(n)
                ema[n] = p.detach().clone().float()
                continue
            t = full[n].to(device=p.device, dtype=torch.float32)
            # the file is whole; shard it onto this run's mesh
            ema[n] = (distribute_tensor(t, p.device_mesh, p.placements)
                      if isinstance(p, DTensor) else t)
        self._ema = ema
        say(f"[ema] loaded {d}: {len(ema) - len(missing)}/{len(ema)} tensors"
            + (f", {len(missing)} missing and seeded from the weights" if missing else ""),
            flush=True)

    # The files only a resume needs; the saved model (`config.json`, `model*.safetensors`) stays.
    _RESUME_ONLY = ("optimizer.bin", "pytorch_model_fsdp.bin", "scheduler.pt")

    def on_save(self, args, state, control, **kwargs):
        """Checkpoint retention, after every save. Checkpoints are ordered by the step in their name.

        The newest `keep_latest` stay whole. Of the older ones, a step on the `optimizer_steps` grid stays whole, a
        step on the `keep_steps` grid keeps only the model (the resume-only files and RNG states are removed), and
        every other one is removed. Every removal is printed.
        """
        if not args.should_save or not args.keep_steps:
            return
        import shutil

        run_dir = args.output_dir
        found = []
        for name in os.listdir(run_dir):
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            if m and os.path.isdir(os.path.join(run_dir, name)):
                found.append((int(m.group(1)), os.path.join(run_dir, name)))
        found.sort()
        for step, path in found[:-self._keep_full]:
            if step % args.optimizer_steps == 0:
                continue
            if step % args.keep_steps == 0:
                dropped = [fn for fn in sorted(os.listdir(path))
                           if fn in self._RESUME_ONLY or fn.startswith("rng_state_")]
                for fn in dropped:
                    os.remove(os.path.join(path, fn))
                if dropped:
                    print(f"[ckpt] thinned {os.path.basename(path)}: dropped {len(dropped)} "
                          f"resume-only files, kept the model", flush=True)
            else:
                shutil.rmtree(path)
                print(f"[ckpt] removed {path}", flush=True)

    # A failed checkpoint is skipped; this many in a row stop the run.
    _MAX_CONSECUTIVE_SAVE_FAILURES = 3

    def _save(self, output_dir=None, state_dict=None):
        """Rank 0's write, retried 3 times in place. It runs after the FSDP gather, so no other rank waits on it."""
        last = None
        for attempt in range(1, 4):
            try:
                return super()._save(output_dir, state_dict)
            except Exception as exc:                      # noqa: BLE001 -- transient by assumption
                last = exc
                print(f"[ckpt] write to {output_dir} failed on attempt {attempt}/3 "
                      f"({type(exc).__name__}: {exc})", flush=True)
                time.sleep(5 * attempt)
        raise last

    def _save_checkpoint(self, model, trial=None, **kwargs):
        """Save a checkpoint (and the EMA); a failure is skipped unless it is the
        `_MAX_CONSECUTIVE_SAVE_FAILURES`-th in a row."""
        try:
            out = super()._save_checkpoint(model, trial, **kwargs)
        except Exception as exc:                          # noqa: BLE001
            self._save_failures += 1
            print(f"[ckpt] SAVE FAILED at step {self.state.global_step} "
                  f"({self._save_failures}/{self._MAX_CONSECUTIVE_SAVE_FAILURES} in a row): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if self._save_failures >= self._MAX_CONSECUTIVE_SAVE_FAILURES:
                raise
            return None
        self._save_failures = 0
        self._ema_save(os.path.join(self.args.output_dir,
                                    f"checkpoint-{self.state.global_step}"))
        return out

    def _check_unwrapped(self, loader):
        """Assert that the train loader is not wrapped by accelerate's dispatcher or sharder."""
        bad = ("DataLoaderDispatcher", "IterableDatasetShard", "DataLoaderShard")
        chain = [type(loader).__name__, type(getattr(loader, "dataset", None)).__name__]
        assert not (set(chain) & set(bad)), \
            f"the loader was wrapped by accelerate after all: {chain}"
        if self.is_world_process_zero():
            print(f"[loader] {chain[0]} over {chain[1]}  (no accelerate wrapper)", flush=True)
        return loader

    def _verify_placement_once(self, model, inputs):
        """On the first step, assert that every tensor of the batch is on this rank's device (the loader is not
        prepared by accelerate; `_prepare_inputs` moves the batch)."""
        if getattr(self, "_placement_ok", False):
            return
        self._placement_ok = True
        want = self.args.device
        p = next((q for q in model.parameters() if q.requires_grad), None)
        bad = {k: str(v.device) for k, v in inputs.items()
               if torch.is_tensor(v) and v.device.type != want.type}
        assert not bad, (
            f"the batch reached compute_loss off-device: {bad} (expected {want}). "
            f"`accelerator.prepare` is skipped on the loader, so `_prepare_inputs` is the only "
            f"thing that moves it")
        assert p is None or p.device.type == want.type, \
            f"trainable parameters are on {p.device}, the batch on {want}"
        shown = {k: (f"{tuple(v.shape)} {v.dtype} {v.device}" if torch.is_tensor(v)
                     else f"{type(v).__name__}[{len(v)}]" if isinstance(v, (list, tuple)) else type(v).__name__)
                 for k, v in inputs.items()}
        print(f"[placement] rank {self.args.process_index}/{self.args.world_size} device={want} "
              f"params={None if p is None else p.device} batch={shown}", flush=True)

    def _interleaved_loss(self, model, inputs, return_outputs):
        """One packed row -> the loss the backward runs on.

        loss = w_img * flow_sum * world / targets + w_text * ce_sum * world / supervised tokens, with both counts over
        all ranks and the accumulation window (`_get_num_items_in_batch`); `world` cancels the gradient reduction's
        mean over ranks. The row's unweighted sums and counts are also accumulated for `log`.
        """
        ce_sum, ce_count, flow_sum, flow_count = self._steps.training_row(model, inputs)
        w = self.args.world_size
        loss = self.args.w_img * flow_sum * (w / self._flow_items)
        if self._ce_on:
            loss = loss + self.args.w_text * ce_sum * (w / self._ce_items)
        sums = torch.cat([torch.stack([flow_sum.detach().float(), ce_sum.detach().float()]),
                          torch.tensor([flow_count, ce_count, 1], dtype=torch.float32,
                                       device=flow_sum.device)])
        self._loss_sums = sums if self._loss_sums is None else self._loss_sums + sums
        return (loss, None) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        """Add the interval's flow and cross-entropy means, over every rank, to a training log.

        A training log is the one that carries `learning_rate`; every rank reaches it at the same
        step (`_maybe_log_save_evaluate` gathers the loss right before), so the all-reduce here is
        issued by all of them.
        """
        if "learning_rate" in logs and self._loss_sums is not None:
            t = self._loss_sums
            if self.args.world_size > 1:
                t = self.accelerator.reduce(t, reduction="sum")
            flow_sum, ce_sum, flow_count, ce_count, rows = t.tolist()
            logs["loss_mu_flow"] = flow_sum / max(flow_count, 1)
            logs["n_targets"] = flow_count / max(rows, 1)
            if self._ce_on:
                logs["loss_ce"] = ce_sum / max(ce_count, 1)
                logs["n_ce_tokens"] = ce_count / max(rows, 1)
            self._loss_sums = None
        return super().log(logs, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._verify_placement_once(model, inputs)
        return self._interleaved_loss(model, inputs, return_outputs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """The eval loss of one batch; the metrics are collected in `self._m`.

        The eval image is in the context and the text is `instruction` (the reconstruction request): the target is
        the context image. `x0_mse` is one denoise from t=1.0 (pure noise).
        """
        inputs = self._prepare_inputs(inputs)
        base = model.module if hasattr(model, "module") else model
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            px, caps = inputs["pixel_values"], inputs["captions"]
            text = [self.args.instruction] * len(caps)
            loss, _ = self._steps.eval_loss(base, px, text)
            z_real = self._steps._targets(base, px)

            read = z_real
            z_gen = self._steps.x0_at(base, z_real, text, t=1.0, read=read)
            self._m["x0_mse"].append(F.mse_loss(z_gen.float(), z_real.float()).item())

            self._m["velocity_mse"].append(self._steps.flow_vmse(base, z_real, text, read=read))
            if self.args.nll_steps:
                self._m["nll_bpd"].append(self._steps.flow_bpd(
                    base, z_real, text, read=read, steps=self.args.nll_steps,
                    n_hutch=self.args.nll_hutchinson))
        return (loss, None, None)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        self._m = {k: [] for k in ("x0_mse", "velocity_mse", "nll_bpd")}

        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys,
                                         metric_key_prefix)
        # The eval set is sharded across ranks: each metric is the mean over every rank's batches.
        for k, vals in self._m.items():
            tot = torch.tensor([float(sum(vals)), float(len(vals))], device=self.args.device)
            if self.args.world_size > 1:
                torch.distributed.all_reduce(tot, op=torch.distributed.ReduceOp.SUM)
            if tot[1] > 0:
                output.metrics[f"{metric_key_prefix}_{k}"] = (tot[0] / tot[1]).item()
        return output


def main(args: VisualGenerationArguments):
    trainer = VisualGenerationTrainer(args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    # `kill -USR1 <pid>` dumps every thread's Python stack to stderr (for a rank stalled in a collective).
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    main(parse(VisualGenerationArguments, add_config_path_arg=True))
