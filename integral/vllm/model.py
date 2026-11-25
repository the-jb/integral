"""Integral in vLLM: image understanding and text generation.

The language model is vLLM's Qwen3.5 (`Qwen3_5ForCausalLM`): Integral's language model and head are Qwen3.5-9B's,
with the same config. Images are read by the vision tokenizer's understanding path, as
`IntegralModel.get_image_features` reads them: `vision_tokenizer.encode_and_merge` (vision tower, abstraction fusion,
bottleneck -> mu, VAE decoder, merger -> language model input embeddings), placed at the image tokens (no deepstack, as Qwen3.5-9B). The inputs come from `IntegralProcessor`. The
visual generation side (the visual generation model, the visual generation adapter, the pixel decoder) is not
loaded: image generation is `IntegralForConditionalGeneration.generate_image` in transformers.

Precision, as the transformers model loaded with `dtype=torch.bfloat16` (`_keep_in_fp32_modules_strict`):
  - the language model and its head in the model dtype (bf16), except the Gated DeltaNet layers' `A_log` (vLLM keeps
    it in fp32) and `linear_attn.norm.weight` (made fp32 here; vLLM would build it in the model dtype)
  - the vision tokenizer in fp32: built under fp32 default dtype and run on fp32 inputs; its output is cast to the
    model dtype at the image tokens, as the transformers model casts it to `inputs_embeds`' dtype
  - the pixels: the processor resizes and patches them without rescaling or normalizing (`IntegralMultiModalProcessor`;
    vLLM casts a processor's floating outputs to the model dtype), so they arrive as uint8, and
    `IntegralVisionEncoder` normalizes them in fp32 as the image processor does
"""

from collections.abc import Iterable, Mapping

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.torch_utils import set_default_torch_dtype

from ..configuration_integral import IntegralVisionTokenizerConfig
from ..modeling_integral import (
    IntegralAbstractionFusion,
    IntegralBottleneck,
    IntegralVisionPatchMerger,
    IntegralVisionTokenizer,
    IntegralVisionTower,
)
from ..processing_integral import IntegralProcessor
from .config import IntegralVLLMConfig


class IntegralProcessingInfo(Qwen3_5ProcessingInfo):
    """Qwen3-VL's image processing with the Integral config and `IntegralProcessor` (its understanding inputs are
    Qwen3-VL's: the same image processor, image tokens and chat template). Images only."""

    def get_hf_config(self):
        return self.ctx.get_hf_config(IntegralVLLMConfig)

    def get_hf_processor(self, **kwargs: object) -> IntegralProcessor:
        return self.ctx.get_hf_processor(IntegralProcessor, **kwargs)

    def get_video_processor(self, **kwargs: object):
        raise ValueError("Integral does not take video input")

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len: int, mm_counts: Mapping[str, int]) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}


class IntegralDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    """Qwen3-VL's profiling inputs, images only."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|vision_start|><|image_pad|><|vision_end|>" * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int], mm_options) -> dict:
        width, height = self.info.get_image_size_with_most_features()
        return {"image": self._get_dummy_images(width=width, height=height, num_images=mm_counts.get("image", 0),
                                                overrides=mm_options.get("image"))}


class IntegralMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Qwen3-VL's processing with the image processor's rescale and normalize off: the patches stay uint8 (vLLM would
    cast normalized floating patches to the model dtype) and `IntegralVisionEncoder` normalizes them."""

    def _call_hf_processor(self, prompt: str, mm_data: Mapping[str, object], mm_kwargs: Mapping[str, object],
                           tok_kwargs: Mapping[str, object]) -> BatchFeature:
        return super()._call_hf_processor(prompt, mm_data, {**mm_kwargs, "do_rescale": False, "do_normalize": False},
                                          tok_kwargs)


class IntegralVisionEncoder(nn.Module):
    """The vision tokenizer's understanding path: its vision tower, abstraction fusion, bottleneck and merger, with
    `IntegralVisionTokenizer.encode_and_merge` itself. Parameter names are the vision tokenizer's.

    The pixels arrive as the uint8 patches of `IntegralMultiModalProcessor` (resized and patched, not rescaled or
    normalized) and are normalized here in fp32 with the image processor's own arithmetic (`rescale_and_normalize`:
    `(x - mean / rescale_factor) / (std / rescale_factor)`), so they equal the image processor's fp32 `pixel_values`."""

    encode_and_merge = IntegralVisionTokenizer.encode_and_merge

    def __init__(self, config: IntegralVisionTokenizerConfig, output_dtype: torch.dtype, image_processor):
        super().__init__()
        # Built on the CPU in fp32, then moved to the device vLLM builds the model on: the buffers computed at
        # construction and not in the checkpoint (the vision tower's rotary `inv_freq`) are then the CPU values
        # transformers computes (vLLM's GPU device context would compute them on the GPU, a last bit apart).
        device = torch.get_default_device()
        with torch.device("cpu"), set_default_torch_dtype(torch.float32):
            self.visual = IntegralVisionTower(config)
            self.abstraction_fusion = IntegralAbstractionFusion(config)
            self.bottleneck = IntegralBottleneck(config)
            self.merger = IntegralVisionPatchMerger(config)
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.merger_out_hidden_size
        # the dtype Qwen3-VL casts the pixels to before calling this module (uint8 values are exact in fp32), and the
        # dtype the embeddings are cast to
        self.dtype = torch.float32
        self.output_dtype = output_dtype
        if not (image_processor.do_rescale and image_processor.do_normalize):
            raise ValueError(f"the image processor must rescale and normalize, got {image_processor}")
        scale = 1.0 / image_processor.rescale_factor
        with torch.device("cpu"):   # as the image processor computes them
            self.register_buffer("pixel_mean", torch.tensor(image_processor.image_mean, dtype=torch.float32) * scale,
                                 persistent=False)
            self.register_buffer("pixel_std", torch.tensor(image_processor.image_std, dtype=torch.float32) * scale,
                                 persistent=False)
        self.to(device)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        return self.embed(pixel_values, grid_thw).to(self.output_dtype)

    def embed(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """The fp32 language model input embeddings of the images from their unnormalized patches."""
        # vLLM keeps image_grid_thw on the CPU; the vision tower builds its position indices from it
        grid_thw = grid_thw.to(self.pixel_mean.device)
        pixel_values = pixel_values.to(self.pixel_mean.device)
        # each patch is (channel, temporal, height, width), channel first
        x = pixel_values.reshape(pixel_values.shape[0], self.pixel_mean.numel(), -1).to(torch.float32)
        x = x.sub(self.pixel_mean[None, :, None]).div(self.pixel_std[None, :, None])
        return self.encode_and_merge(x.reshape(pixel_values.shape[0], -1), grid_thw)[1]


@MULTIMODAL_REGISTRY.register_processor(
    IntegralMultiModalProcessor,
    info=IntegralProcessingInfo,
    dummy_inputs=IntegralDummyInputsBuilder,
)
class IntegralForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        # the generation side: not loaded
        "model.visual_generation_model.": None,
        "model.vision_tokenizer.visual_generation_adapter.": None,
        "model.vision_tokenizer.pixel_decoder.": None,
        "model.vision_tokenizer.": "visual.",
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # Qwen3_5ForConditionalGeneration.__init__ with the vision tokenizer in place of Qwen's vision tower
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config: IntegralVLLMConfig = model_config.hf_config
        multimodal_config = model_config.multimodal_config
        if vllm_config.quant_config is not None:
            raise ValueError("Integral in vLLM supports no quantization")

        self.config = config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        if self.use_data_parallel:
            raise ValueError("Integral's vision tokenizer runs whole on every rank; use mm_encoder_tp_mode='weights'")
        self.is_multimodal_pruning_enabled = False
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(model_config)

        # attributes the Qwen3-VL methods read
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        image_processor = cached_processor_from_config(model_config, processor_cls=IntegralProcessor).image_processor
        with self._mark_tower_model(vllm_config, {"image"}):
            self.visual = IntegralVisionEncoder(config.vision_tokenizer_config, model_config.dtype, image_processor)

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model"))
        self._keep_gated_delta_net_norms_fp32()

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _keep_gated_delta_net_norms_fp32(self):
        """`linear_attn.norm.weight` in fp32, as Qwen3.5 stores it and the transformers model keeps it (vLLM builds
        it in the model dtype; its gated RMSNorm kernels take an fp32 weight)."""
        text = self.config.text_config
        expected = sum(t == "linear_attention" for t in text.layer_types)
        found = 0
        for name, module in self.language_model.named_modules():
            if name.endswith("linear_attn.norm"):
                weight = nn.Parameter(module.weight.data.to(torch.float32), requires_grad=False)
                set_weight_attrs(weight, {"dummy_weight_value": 1.0})
                module.weight = weight
                found += 1
        if found != expected:
            raise RuntimeError(f"{found} Gated DeltaNet norms made fp32, the config has {expected} linear-attention "
                               f"layers")

    def get_encoder_cudagraph_config(self):
        # Qwen3-VL's encoder CUDA graphs capture its vision tower's buffers, which the vision tokenizer does not have
        raise ValueError("Integral's vision tokenizer does not run under encoder CUDA graphs "
                         "(compilation_config.cudagraph_mm_encoder)")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model", connector=["visual.merger"], tower_model="visual.")
