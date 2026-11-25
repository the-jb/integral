"""Processor of the Integral model: the image processor and the tokenizer for understanding inputs, and the prompts
of image generation."""

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin


class IntegralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
    }


class IntegralProcessor(ProcessorMixin):
    r"""
    Constructs an Integral processor from an image processor and a tokenizer.

    Args:
        image_processor ([`BaseImageProcessor`], *optional*):
            Produces `pixel_values` and `image_grid_thw` in the vision tokenizer's input format.
        tokenizer ([`PreTrainedTokenizerBase`], *optional*):
            The language model's tokenizer.
        chat_template (`str`, *optional*):
            The chat template.
        image_generation_system_prompt (`str`, *optional*):
            The system turn of image generation prompts.
        image_generation_user_template (`str`, *optional*):
            The user turn of image generation prompts, with `{h}`, `{w}` (the image size in pixels) and `{caption}`.
        image_generation_marker (`str`, *optional*):
            The text after the assistant turn opens and before the generated image, with `{h}` and `{w}`.
    """

    valid_processor_kwargs = IntegralProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None,
                 image_generation_system_prompt=None, image_generation_user_template=None,
                 image_generation_marker=None, **kwargs):
        self.image_token = getattr(tokenizer, "image_token", None) or "<|image_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_generation_system_prompt = image_generation_system_prompt
        self.image_generation_user_template = image_generation_user_template
        self.image_generation_marker = image_generation_marker

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        # transformers >= 5.16 passes the processing kwargs too; they change nothing here
        merge_length = self.image_processor.merge_size**2
        return self.image_token * int(image_inputs["image_grid_thw"][image_idx].prod() // merge_length)

    @property
    def model_input_names(self):
        return super().model_input_names + ["mm_token_type_ids"]

    def image_generation_prompt_ids(self, caption: str, height: int, width: int,
                                    image_tokens: int | list[int] = 0) -> list[int]:
        """
        The token ids of the prompt asking for one `height x width` image of `caption`: the system turn, the user
        turn (the input images, in order, then `image_generation_user_template` around the caption), the opened
        assistant turn, then `image_generation_marker`.

        Args:
            caption (`str`):
                The image description or, with input images, the instruction; `""` gives the unconditional prompt
                of classifier-free guidance.
            height (`int`), width (`int`):
                The size of the generated image in pixels.
            image_tokens (`int` or `list[int]`, *optional*, defaults to 0):
                The image tokens of each input image, in order (an `int` for one image); 0 or `[]` for none.

        Returns:
            `list[int]`.
        """
        if None in (self.image_generation_system_prompt, self.image_generation_user_template,
                    self.image_generation_marker):
            raise ValueError("this processor has no image generation prompt (image_generation_system_prompt, "
                             "image_generation_user_template, image_generation_marker)")
        image_tokens = ([image_tokens] if image_tokens else []) if isinstance(image_tokens, int) else list(image_tokens)
        request = self.image_generation_user_template.format(h=height, w=width, caption=caption)
        content = ([{"type": "image"}] * len(image_tokens) + [{"type": "text", "text": request}] if image_tokens
                   else request)
        messages = [
            {"role": "system", "content": self.image_generation_system_prompt},
            {"role": "user", "content": content},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if image_tokens:
            pads = [k for k, t in enumerate(ids) if t == self.image_token_id]
            if len(pads) != len(image_tokens):
                raise ValueError(f"the chat template emitted {len(pads)} image tokens for {len(image_tokens)} images")
            # widened from the back so the earlier positions stay valid
            for k, n in zip(reversed(pads), reversed(image_tokens)):
                ids = ids[:k] + [self.image_token_id] * n + ids[k + 1:]
        marker = self.tokenizer(self.image_generation_marker.format(h=height, w=width),
                                add_special_tokens=False)["input_ids"]
        return ids + marker

    def image_generation_pixel_values(self, images: list) -> BatchFeature:
        """
        The input images of image generation as the vision tokenizer reads them, at their own size: pixels
        `uint8 / 255`, normalized with the image processor's `image_mean` and `image_std`, then cut into patches.

        Args:
            images (`list`):
                `PIL.Image.Image`s or `torch.Tensor`s of shape `(3, height, width)` in `uint8`; height and width
                multiples of `patch_size * merge_size`.

        Returns:
            [`BatchFeature`] with `pixel_values` and `image_grid_thw` (`(num_images, 3)`).
        """
        unit = self.image_processor.patch_size * self.image_processor.merge_size
        mean = torch.tensor(self.image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(self.image_processor.image_std).view(3, 1, 1)
        normalized = []
        for image in images:
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
            if image.dtype != torch.uint8 or image.dim() != 3 or image.shape[0] != 3:
                raise ValueError(f"an input image must be a PIL image or a (3, H, W) uint8 tensor, got "
                                 f"{tuple(image.shape)} {image.dtype}")
            if image.shape[1] % unit or image.shape[2] % unit:
                raise ValueError(f"input image sides must be multiples of {unit}, got {tuple(image.shape[1:])}")
            normalized.append((image.float().div(255) - mean) / std)
        return self.image_processor(images=normalized, do_resize=False, do_rescale=False, do_normalize=False,
                                    return_tensors="pt")

    def prepare_image_generation(self, prompts: str | list[str], height: int = 512, width: int = 512,
                                 images: list | None = None, guidance: bool = True,
                                 image_guidance: bool | None = None) -> BatchFeature:
        """
        Builds the inputs of `IntegralForConditionalGeneration.prepare_image_generation_cache` / `generate_image`.

        Args:
            prompts (`str` or `list[str]`):
                The image descriptions, or with `images` the instructions.
            height (`int`, *optional*, defaults to 512), width (`int`, *optional*, defaults to 512):
                The size of the generated images in pixels, a multiple of `patch_size * merge_size`.
            images (`list`, *optional*):
                The input images of each prompt (`image_generation_pixel_values`): one image, or a list of one or
                more read in order in the same user turn, before the instruction. The unconditional rows keep them and
                drop the instruction.
            guidance (`bool`, *optional*, defaults to `True`):
                Whether to add the unconditional rows classifier-free guidance needs.
            image_guidance (`bool`, *optional*):
                Whether to add the image-free rows of image guidance (`generate_image(image_guidance_scale=...)`):
                each prompt's instruction with no input image, the prompt of text-to-image. Needs `images` and
                `guidance`. Defaults to `True` with `images` and `guidance` (editing samples with image guidance by
                default), `False` otherwise.

        Returns:
            [`BatchFeature`] with, for `B` prompts:
            - `input_ids`, `attention_mask`, `mm_token_type_ids` (1 at image tokens) of shape `(B, sequence_length)`,
              `(2B, sequence_length)` with guidance, `(3B, sequence_length)` with image guidance: the prompts, then
              their unconditional rows, then their image-free rows, each in prompt order. Padded on the right.
            - `visual_grid_thw` of shape `(B, 3)`: the patch grid `(1, height // patch_size, width // patch_size)`.
            - with `images`: `pixel_values` and `image_grid_thw` of the input images, in row order (the prompts' images,
              then the unconditional rows' copies; the image-free rows have none).
        """
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        patch_size, merge_size = self.image_processor.patch_size, self.image_processor.merge_size
        unit = patch_size * merge_size
        if height % unit or width % unit:
            raise ValueError(f"height and width must be multiples of {unit}, got {height}x{width}")
        if image_guidance is None:
            image_guidance = images is not None and guidance
        if image_guidance and (images is None or not guidance):
            raise ValueError("image guidance needs input images and the unconditional rows (`guidance=True`)")
        captions = prompts + ([""] * len(prompts) if guidance else []) + (prompts if image_guidance else [])
        image_inputs, image_tokens = None, [[] for _ in captions]
        if images is not None:
            if len(images) != len(prompts):
                raise ValueError(f"input images for {len(images)} prompts, {len(prompts)} prompts")
            per_prompt = [list(i) if isinstance(i, (list, tuple)) else [i] for i in images]
            if not all(per_prompt):
                raise ValueError("a prompt given `images` needs at least one image")
            image_inputs = self.image_generation_pixel_values([i for row in per_prompt for i in row])
            counts = (image_inputs["image_grid_thw"].prod(dim=-1) // merge_size**2).tolist()
            tokens, k = [], 0
            for row in per_prompt:
                tokens.append(counts[k:k + len(row)])
                k += len(row)
            image_tokens = tokens + (tokens if guidance else []) + ([[]] * len(prompts) if image_guidance else [])
        rows = [self.image_generation_prompt_ids(caption, height, width, n)
                for caption, n in zip(captions, image_tokens)]
        length = max(len(ids) for ids in rows)
        input_ids = torch.full((len(rows), length), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), length, dtype=torch.long)
        for i, ids in enumerate(rows):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attention_mask[i, :len(ids)] = 1
        visual_grid_thw = torch.tensor([[1, height // patch_size, width // patch_size]] * len(prompts))
        outputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": ((input_ids == self.image_token_id) & (attention_mask == 1)).int(),
            "visual_grid_thw": visual_grid_thw,
        }
        if image_inputs is not None:
            repeat = 2 if guidance else 1
            outputs["pixel_values"] = torch.cat([image_inputs["pixel_values"]] * repeat)
            outputs["image_grid_thw"] = torch.cat([image_inputs["image_grid_thw"]] * repeat)
        return BatchFeature(outputs)


__all__ = ["IntegralProcessor"]
