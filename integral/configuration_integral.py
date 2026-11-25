# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig, remap_legacy_layer_types
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen3.5-27B")
@strict
class IntegralTextConfig(PreTrainedConfig):
    r"""
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_key_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each key head in linear attention.
    linear_value_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each value head in linear attention.
    linear_num_key_heads (`int`, *optional*, defaults to 16):
        Number of key heads used in linear attention layers.
    linear_num_value_heads (`int`, *optional*, defaults to 32):
        Number of value heads used in linear attention layers.
    """

    model_type = "integral_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkv": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_z": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_b": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_a": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = None  # no Moe

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = remap_legacy_layer_types(self.layer_types)

        super().__post_init__(**kwargs)


class IntegralVisionTokenizerConfig(PreTrainedConfig):
    r"""
    Configuration for [`IntegralVisionTokenizer`].

    Args:
        vision_num_hidden_layers (`int`, *optional*, defaults to 27):
            Number of blocks in the vision tower.
        vision_hidden_size (`int`, *optional*, defaults to 1152):
            Hidden size of the vision tower.
        vision_intermediate_size (`int`, *optional*, defaults to 4304):
            MLP size of the vision tower.
        vision_num_attention_heads (`int`, *optional*, defaults to 16):
            Attention heads of the vision tower.
        vision_num_position_embeddings (`int`, *optional*, defaults to 2304):
            Size of the vision tower's position table, a square grid interpolated to each image.
        patch_size (`int`, *optional*, defaults to 16):
            Pixels per patch side.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            Frames per patch. A still image is repeated to fill them.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Patches merged per side into one language model token.
        image_mean (`list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Per-channel mean for normalizing RGB in [0, 1].
        image_std (`list[float]`, *optional*, defaults to `[0.5, 0.5, 0.5]`):
            Per-channel standard deviation for normalizing RGB in [0, 1].
        abstraction_fusion_group_size (`int`, *optional*, defaults to 4):
            Consecutive vision tower hidden states per abstraction group.
        abstraction_fusion_hidden_size (`int`, *optional*, defaults to 1152):
            Hidden size of abstraction fusion.
        abstraction_fusion_num_hidden_layers (`int`, *optional*, defaults to 1):
            Self-attention layers across the abstraction groups.
        abstraction_fusion_num_attention_heads (`int`, *optional*, defaults to 16):
            Attention heads of those layers.
        abstraction_fusion_intermediate_size (`int`, *optional*, defaults to 4608):
            MLP size of those layers.
        bottleneck_pre_intermediate_size (`int`, *optional*, defaults to 4096):
            Hidden size of the bottleneck's first SwiGLU.
        bottleneck_hidden_size (`int`, *optional*, defaults to 1152):
            Output size of the bottleneck's first SwiGLU.
        bottleneck_post_intermediate_size (`int`, *optional*, defaults to 4608):
            Hidden size of the bottleneck's second SwiGLU.
        vae_hidden_size (`int`, *optional*, defaults to 2304):
            Input and output size of the VAE.
        latent_size (`int`, *optional*, defaults to 256):
            Size of the latent.
        merger_intermediate_size (`int`, *optional*, defaults to 4608):
            Hidden size of the merger's input SwiGLU.
        merger_hidden_size (`int`, *optional*, defaults to 1152):
            Output size of the merger's input SwiGLU.
        merger_out_hidden_size (`int`, *optional*, defaults to 4096):
            Output size of the merger, the language model's hidden size.
        merger_layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon of the merger's LayerNorm.
        visual_generation_adapter_num_layers (`int`, *optional*, defaults to 8):
            Number of key/value deltas the visual generation adapter produces, one per visual generation
            block.
        visual_generation_adapter_intermediate_size (`int`, *optional*, defaults to 2304):
            Hidden size of the visual generation adapter.
        visual_generation_adapter_num_key_value_heads (`int`, *optional*, defaults to 4):
            Key/value heads of each delta.
        visual_generation_adapter_head_dim (`int`, *optional*, defaults to 256):
            Head size of each delta.
        pixel_decoder_in_intermediate_size (`int`, *optional*, defaults to 4608):
            Hidden size of the pixel decoder's input SwiGLU.
        pixel_decoder_in_hidden_size (`int`, *optional*, defaults to 1152):
            Output size of the pixel decoder's input SwiGLU.
        pixel_decoder_hidden_size (`int`, *optional*, defaults to 1152):
            Hidden size of the pixel decoder.
        pixel_decoder_num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of blocks in the pixel decoder.
        pixel_decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Attention heads of the pixel decoder.
        pixel_decoder_intermediate_size (`int`, *optional*, defaults to 3072):
            SwiGLU size of the pixel decoder's blocks.
        pixel_decoder_rope_theta (`float`, *optional*, defaults to 100.0):
            Base of the pixel decoder's 2D RoPE.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon of the LayerNorms in abstraction fusion, the bottleneck and the pixel decoder.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the normal initializer for linear, convolution and embedding
            weights.
    """

    model_type = "integral_vision_tokenizer"
    base_config_key = "vision_tokenizer_config"

    vision_num_hidden_layers: int = 27
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_attention_heads: int = 16
    vision_num_position_embeddings: int = 2304

    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    image_mean: list[float] | None = None
    image_std: list[float] | None = None

    abstraction_fusion_group_size: int = 4
    abstraction_fusion_hidden_size: int = 1152
    abstraction_fusion_num_hidden_layers: int = 1
    abstraction_fusion_num_attention_heads: int = 16
    abstraction_fusion_intermediate_size: int = 4608

    bottleneck_pre_intermediate_size: int = 4096
    bottleneck_hidden_size: int = 1152
    bottleneck_post_intermediate_size: int = 4608
    vae_hidden_size: int = 2304
    latent_size: int = 256

    merger_intermediate_size: int = 4608
    merger_hidden_size: int = 1152
    merger_out_hidden_size: int = 4096
    merger_layer_norm_eps: float = 1e-6

    visual_generation_adapter_num_layers: int = 8
    visual_generation_adapter_intermediate_size: int = 2304
    visual_generation_adapter_num_key_value_heads: int = 4
    visual_generation_adapter_head_dim: int = 256

    pixel_decoder_in_intermediate_size: int = 4608
    pixel_decoder_in_hidden_size: int = 1152
    pixel_decoder_hidden_size: int = 1152
    pixel_decoder_num_hidden_layers: int = 28
    pixel_decoder_num_attention_heads: int = 16
    pixel_decoder_intermediate_size: int = 3072
    pixel_decoder_rope_theta: float = 100.0

    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.image_mean is None:
            self.image_mean = [0.5, 0.5, 0.5]
        if self.image_std is None:
            self.image_std = [0.5, 0.5, 0.5]
        super().__post_init__(**kwargs)

    @property
    def num_vision_hidden_states(self) -> int:
        """Hidden states the vision tower emits: the patch embeddings and one per block."""
        return self.vision_num_hidden_layers + 1


class IntegralVisualGenerationConfig(PreTrainedConfig):
    r"""
    Configuration for [`IntegralVisualGenerationModel`]: the visual generation blocks, which read the key/value
    states of language model layers, and the flow head.

    Args:
        key_value_layer_indices (`list[int]`, *optional*, defaults to `[3, 7, 11, 15, 19, 23, 27, 31]`):
            Full-attention language model layers whose key/value states the visual generation blocks read, one
            block per layer, in increasing order.
        latent_size (`int`, *optional*, defaults to 256):
            Size of the image latent the model denoises.
        hidden_size (`int`, *optional*, defaults to 2048):
            Hidden size of the visual generation blocks.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Query heads of the visual generation blocks.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Key/value heads the blocks compute from their own hidden states.
        head_dim (`int`, *optional*, defaults to 256):
            Head size of the blocks, equal to the language model's.
        intermediate_size (`int`, *optional*, defaults to 8192):
            MLP size of the blocks.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon of the LayerNorms of the blocks and the flow head.
        timestep_embedding_size (`int`, *optional*, defaults to 512):
            Size of the sinusoidal timestep embedding of the blocks.
        timestep_scale (`float`, *optional*, defaults to 1000.0):
            Factor applied to the timestep in [0, 1] before the blocks' sinusoidal embedding.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5):
            Fraction of each head the blocks' 2D rotary embedding rotates.
        rope_theta (`float`, *optional*, defaults to 100.0):
            Base of the blocks' 2D rotary embedding.
        flow_head_hidden_size (`int`, *optional*, defaults to 3072):
            Hidden size of the flow head.
        flow_head_num_hidden_layers (`int`, *optional*, defaults to 2):
            Blocks in the flow head.
        flow_head_num_attention_heads (`int`, *optional*, defaults to 32):
            Attention heads of the flow head.
        flow_head_intermediate_size (`int`, *optional*, defaults to 12288):
            MLP size of the flow head.
        flow_head_rope_theta (`float`, *optional*, defaults to 100.0):
            Base of the flow head's 2D rotary embedding.
        time_shift (`float`, *optional*, defaults to 4.0):
            Shift `s` of the timestep schedule, `t -> s * t / (1 + (s - 1) * t)`, the model was trained with; the
            default of `generate_image`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the normal initializer for linear weights.
    """

    model_type = "integral_visual_generation"
    base_config_key = "visual_generation_config"

    key_value_layer_indices: list[int] | None = None
    latent_size: int = 256
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    intermediate_size: int = 8192
    layer_norm_eps: float = 1e-6
    timestep_embedding_size: int = 512
    timestep_scale: float = 1000.0
    partial_rotary_factor: float = 0.5
    rope_theta: float = 100.0
    flow_head_hidden_size: int = 3072
    flow_head_num_hidden_layers: int = 2
    flow_head_num_attention_heads: int = 32
    flow_head_intermediate_size: int = 12288
    flow_head_rope_theta: float = 100.0
    time_shift: float = 4.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.key_value_layer_indices is None:
            self.key_value_layer_indices = [3, 7, 11, 15, 19, 23, 27, 31]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Qwen/Qwen3.5-27B")
@strict
class IntegralConfig(PreTrainedConfig):
    r"""
    vision_tokenizer_config (`IntegralVisionTokenizerConfig` or `dict`, *optional*):
        Configuration of the vision tokenizer.
    visual_generation_config (`IntegralVisualGenerationConfig` or `dict`, *optional*):
        Configuration of the visual generation model.
    """

    model_type = "integral"
    sub_configs = {
        "vision_tokenizer_config": IntegralVisionTokenizerConfig,
        "text_config": IntegralTextConfig,
        "visual_generation_config": IntegralVisualGenerationConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_tokenizer_config: dict | PreTrainedConfig | None = None
    visual_generation_config: dict | PreTrainedConfig | None = None

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_tokenizer_config, dict):
            self.vision_tokenizer_config = self.sub_configs["vision_tokenizer_config"](**self.vision_tokenizer_config)
        elif self.vision_tokenizer_config is None:
            self.vision_tokenizer_config = self.sub_configs["vision_tokenizer_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        if isinstance(self.visual_generation_config, dict):
            self.visual_generation_config = self.sub_configs["visual_generation_config"](**self.visual_generation_config)
        elif self.visual_generation_config is None:
            self.visual_generation_config = self.sub_configs["visual_generation_config"]()

        tokenizer, text, generation = self.vision_tokenizer_config, self.text_config, self.visual_generation_config
        layers = list(generation.key_value_layer_indices)
        mismatched = {
            "merger_out_hidden_size vs hidden_size": (tokenizer.merger_out_hidden_size, text.hidden_size),
            "visual_generation_adapter_num_key_value_heads vs num_key_value_heads":
                (tokenizer.visual_generation_adapter_num_key_value_heads, text.num_key_value_heads),
            "visual_generation_adapter_head_dim vs head_dim": (tokenizer.visual_generation_adapter_head_dim, text.head_dim),
            "visual_generation_adapter_num_layers vs len(key_value_layer_indices)":
                (tokenizer.visual_generation_adapter_num_layers, len(layers)),
            "visual_generation_config.head_dim vs head_dim": (generation.head_dim, text.head_dim),
            "visual_generation_config.latent_size vs vision_tokenizer_config.latent_size":
                (generation.latent_size, tokenizer.latent_size),
            "key_value_layer_indices are increasing full-attention layers":
                (layers == sorted(set(layers)) and all(0 <= i < text.num_hidden_layers for i in layers)
                 and all(text.layer_types[i] == "full_attention" for i in layers if 0 <= i < text.num_hidden_layers),
                 True),
        }
        mismatched = {k: v for k, v in mismatched.items() if v[0] != v[1]}
        if mismatched:
            raise ValueError(f"the sub-configs disagree: {mismatched}")

        super().__post_init__(**kwargs)


__all__ = ["IntegralConfig", "IntegralTextConfig", "IntegralVisionTokenizerConfig", "IntegralVisualGenerationConfig"]
