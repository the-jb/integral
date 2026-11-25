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

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.integrations.accelerate import force_accelerate_hooks
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from transformers.utils.generic import (
    accepts_precomputed_kwargs,
    maybe_autocast,
    merge_with_config_defaults,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
from transformers.utils.output_capturing import capture_outputs
from .configuration_integral import (
    IntegralConfig,
    IntegralTextConfig,
    IntegralVisionTokenizerConfig,
    IntegralVisualGenerationConfig,
)
from .sampling import (
    classifier_free_guidance,
    dpm_solver_sample,
    dpm_solver_timesteps,
    euler_step,
    flow_timesteps,
)


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None

logger = logging.get_logger(__name__)


class IntegralTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: IntegralTextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: IntegralTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Integral has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class IntegralRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # for each chunk
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@use_kernel_forward_from_hub("IntegralGatedDeltaNet")
class IntegralGatedDeltaNet(nn.Module):
    def __init__(self, config: IntegralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = (
            IntegralRMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
            if FusedRMSNormGated is None
            else FusedRMSNormGated(
                self.head_v_dim,
                eps=self.layer_norm_epsilon,
                activation=self.activation,
            )
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of the required library is not installed. Falling back to "
                "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

        self.layer_type = config.layer_types[layer_idx]

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        # We have cached `conv_state` / `recurrent_state` to continue from. The two cached modes
        # (single-token decode and chunk-tokens continuation) share the state read here; they only
        # diverge in how the conv input is assembled and which kernel consumes the states below,
        # which we gate locally on `seq_len`.
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        # getting projected states from cache if it exists
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1:
            # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            # Multi-token forward (prefill, or chunked-tokens decode when the cache has prior state).
            if use_precomputed_states:
                # Cached chunked-tokens decode: prepend the cached conv context so the causal conv
                # sees the correct left-context rather than zero-padding. Dropped from the output
                # at the end of this branch.
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=kwargs.get("seq_idx"),
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                # The chunked FLA kernel takes a single `cu_seqlens` arg; for packed self-attention this matches q-side lengths.
                cu_seqlens=kwargs.get("cu_seq_lens_q"),
            )

        # Update cache
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Adapted from transformers.models.glm.modular_glm.apply_rotary_pos_emb
def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to a query or key tensor.

    Only the first `cos.shape[-1]` channels are rotated; the rest pass through.

    Args:
        x (`torch.Tensor` of shape `(batch_size, num_heads, seq_len, head_dim)`): The query or key tensor.
        cos (`torch.Tensor` of shape `(batch_size, seq_len, rotary_dim)`): The cosine part of the rotary embedding.
        sin (`torch.Tensor` of shape `(batch_size, seq_len, rotary_dim)`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which `cos` and `sin` are unsqueezed to broadcast against `x`.

    Returns:
        `torch.Tensor` of the same shape as `x`.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_embed, x_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class IntegralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: IntegralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = IntegralRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = IntegralRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    def normalize_key(
        self, key_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Applies `k_norm` and the rotary embedding to the output of `k_proj`.

        Args:
            key_states (`torch.Tensor` of shape `(batch_size, seq_len, num_key_value_heads * head_dim)`):
                Output of `k_proj`.
            position_embeddings (`tuple[torch.Tensor, torch.Tensor]`):
                `(cos, sin)` from `IntegralTextRotaryEmbedding`.

        Returns:
            `torch.Tensor` of shape `(batch_size, num_key_value_heads, seq_len, head_dim)`.
        """
        key_states = self.k_norm(key_states.view(*key_states.shape[:-1], -1, self.head_dim)).transpose(1, 2)
        cos, sin = position_embeddings
        return apply_rotary_pos_emb(key_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        output_visual_key_value_states: bool = False,
        visual_key_value_delta: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple:
        """
        Returns `(attn_output, attn_weights)`, and with `output_visual_key_value_states=True` also
        `(visual_key_states, visual_value_states)` as a third element: the key/value states the visual generation
        blocks read, each of shape `(batch_size, num_key_value_heads, seq_len, head_dim)`. They are computed
        separately from the key/value states this layer attends with and stores in `past_key_values`:
        `visual_key_value_delta` `(key_delta, value_delta)`, each of shape
        `(batch_size, seq_len, num_key_value_heads * head_dim)`, is added to the outputs of `k_proj` and `v_proj`,
        then the keys go through `k_norm` and the rotary embedding. Without a delta they are computed from the
        outputs of `k_proj` and `v_proj` alone.
        """
        if visual_key_value_delta is not None and not output_visual_key_value_states:
            raise ValueError("visual_key_value_delta is only read with output_visual_key_value_states=True")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)
        key_projection = self.k_proj(hidden_states)
        value_projection = self.v_proj(hidden_states)

        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb(self.q_norm(query_states.view(hidden_shape)).transpose(1, 2), cos, sin)
        key_states = self.normalize_key(key_projection, position_embeddings)
        value_states = value_projection.view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        if not output_visual_key_value_states:
            return attn_output, attn_weights

        visual_key_states, visual_value_states = key_projection, value_projection
        if visual_key_value_delta is not None:
            key_delta, value_delta = visual_key_value_delta
            visual_key_states = visual_key_states + key_delta.to(visual_key_states.dtype)
            visual_value_states = visual_value_states + value_delta.to(visual_value_states.dtype)
        visual_key_states = self.normalize_key(visual_key_states, position_embeddings)
        visual_value_states = visual_value_states.view(hidden_shape).transpose(1, 2)
        return attn_output, attn_weights, (visual_key_states, visual_value_states)


class IntegralMLP(nn.Module):
    def __init__(self, config: IntegralConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class IntegralRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Integral is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class IntegralDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: IntegralTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = IntegralGatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = IntegralAttention(config, layer_idx)
        self.mlp = IntegralMLP(config, config.intermediate_size)
        self.input_layernorm = IntegralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = IntegralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_visual_key_value_states: bool = False,
        visual_key_value_delta: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor | tuple:
        """
        Returns the hidden states, and with `output_visual_key_value_states=True` a flat tuple
        `(hidden_states, visual_key_states, visual_value_states)` with the visual key/value states returned by
        `IntegralAttention`, every tensor at the top level so that reentrant gradient checkpointing propagates
        gradients to all three. Only full-attention layers have key/value states.
        """
        if output_visual_key_value_states and self.block_type != "full_attention":
            raise ValueError(f"A {self.block_type} layer has no key/value states.")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif self.block_type == "full_attention":
            # Self Attention
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                output_visual_key_value_states=output_visual_key_value_states,
                visual_key_value_delta=visual_key_value_delta,
                **kwargs,
            )
            hidden_states = attn_outputs[0]

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if output_visual_key_value_states:
            visual_key_states, visual_value_states = attn_outputs[2]
            return hidden_states, visual_key_states, visual_value_states
        return hidden_states


class IntegralPreTrainedModel(PreTrainedModel):
    config: IntegralConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IntegralDecoderLayer", "IntegralVisionBlock", "IntegralTransformerBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _can_record_outputs = {
        "hidden_states": IntegralDecoderLayer,
        "attentions": IntegralAttention,
    }
    _is_stateful = True
    _can_compile_fullgraph = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, IntegralGatedDeltaNet):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        # We initialize with 0s to be 1 centered as the RMSNorm here does (1 + weight)
        elif isinstance(module, IntegralRMSNorm):
            init.zeros_(module.weight)


def _apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple:
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q.to(q_dtype), k.to(k_dtype)


def _normalize(x: Tensor) -> Tensor:
    """LayerNorm without affine over the last dim, in fp32, returned in the input dtype."""
    orig_dtype = x.dtype
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = torch.layer_norm(x.float(), x.shape[-1:])
    return x.to(orig_dtype)


def vision_position_ids(grid_thw: Tensor, spatial_merge_size: int) -> Tensor:
    """(h, w) index of every patch, in merge-block order. [total_patches, 2]"""
    device = grid_thw.device
    m = spatial_merge_size
    out = []
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                    indexing="ij")
        block = (h // m, m, w // m, m)
        hpos = hpos.reshape(block).transpose(1, 2).flatten()
        wpos = wpos.reshape(block).transpose(1, 2).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def vision_rowmajor_permutation(grid_hw: tuple, spatial_merge_size: int, device) -> Tensor:
    """Permutation from merge-block order to row-major order of one `(height, width)` patch grid:
    `perm[row_major_index]` is the merge-block-order index."""
    gh, gw = grid_hw
    m = spatial_merge_size
    i = torch.arange(gh * gw, device=device)
    bh = i // ((gw // m) * m * m)
    rem = i % ((gw // m) * m * m)
    bw = rem // (m * m)
    rem2 = rem % (m * m)
    row = bh * m + rem2 // m
    col = bw * m + rem2 % m
    perm = torch.empty(gh * gw, dtype=torch.long, device=device)
    perm[row * gw + col] = i
    return perm


def vision_bilinear(grid_thw: Tensor, side: int, spatial_merge_size: int) -> tuple:
    """Bilinear corners and weights into the side x side position table, in merge-block order.
    Returns indices [4, total_patches] (long) and weights [4, total_patches]."""
    device = grid_thw.device
    m = spatial_merge_size
    idx_parts = [[] for _ in range(4)]
    w_parts = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac, w_frac = h_grid - h_floor, w_grid - w_floor
        h_floor_off, h_ceil_off = h_floor * side, h_ceil * side
        corners = [
            (h_floor_off[:, None] + w_floor[None, :]).flatten(),
            (h_floor_off[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_off[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_off[:, None] + w_ceil[None, :]).flatten(),
        ]
        weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h, device=device).view(h // m, m)
        w_idx = torch.arange(w, device=device).view(w // m, m)
        reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :]).transpose(1, 2).flatten().repeat(t)
        for i in range(4):
            idx_parts[i].append(corners[i][reorder])
            w_parts[i].append(weights[i][reorder])
    return (torch.stack([torch.cat(p) for p in idx_parts]),
            torch.stack([torch.cat(p) for p in w_parts]))


class IntegralVisionPatchEmbed(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, temporal_patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        k = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(3, hidden_size, kernel_size=k, stride=k, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, 3, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(dtype=self.proj.weight.dtype)).view(-1, self.hidden_size)


class IntegralVisionRotaryEmbedding(nn.Module):
    """`inv_freq` is not saved in checkpoints; `_init_weights` rebuilds it with `compute_inv_freq`."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim, self.theta = dim, theta
        self.register_buffer("inv_freq", self.compute_inv_freq(), persistent=False)

    def compute_inv_freq(self) -> Tensor:
        return 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))

    def forward(self, position_ids: Tensor) -> Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class IntegralVisionAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scaling = (hidden_size // num_heads) ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor, cu_seqlens: Tensor, position_embeddings: tuple) -> Tensor:
        seq = x.shape[0]
        q, k, v = self.qkv(x).reshape(seq, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = _apply_rotary(q, k, *position_embeddings)
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))       # [1, heads, seq, head_dim]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs = [
            F.scaled_dot_product_attention(qi, ki, vi, attn_mask=None, dropout_p=0.0,
                                           scale=self.scaling, is_causal=False).transpose(1, 2).contiguous()
            for qi, ki, vi in zip(q.split(lengths, dim=2), k.split(lengths, dim=2), v.split(lengths, dim=2))
        ]
        return self.proj(torch.cat(outs, dim=1).reshape(seq, -1).contiguous())


class IntegralVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class IntegralVisionBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = IntegralVisionAttention(hidden_size, num_heads)
        self.mlp = IntegralVisionMLP(hidden_size, intermediate_size)

    def forward(self, x: Tensor, cu_seqlens: Tensor, position_embeddings: tuple) -> Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
        return x + self.mlp(self.norm2(x))


class IntegralVisionTower(nn.Module):
    """Image patches -> hidden states of the patch embeddings and every block, row-major and LayerNorm'd, and the
    last block's output. The Qwen3.5 vision tower without its patch merger; parameter names match Qwen3.5
    checkpoints."""

    def __init__(self, config: IntegralVisionTokenizerConfig):
        super().__init__()
        hidden_size = config.vision_hidden_size
        num_heads = config.vision_num_attention_heads
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = IntegralVisionPatchEmbed(hidden_size, config.patch_size, config.temporal_patch_size)
        self.pos_embed = nn.Embedding(config.vision_num_position_embeddings, hidden_size)
        self.num_grid_per_side = int(config.vision_num_position_embeddings ** 0.5)
        self.rotary_pos_emb = IntegralVisionRotaryEmbedding(hidden_size // num_heads // 2)
        self.blocks = nn.ModuleList([IntegralVisionBlock(hidden_size, config.vision_intermediate_size, num_heads)
                                     for _ in range(config.vision_num_hidden_layers)])

    def forward(self, pixel_values: Tensor, image_grid_thw: Tensor) -> tuple:
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(total_patches, 3 * temporal_patch_size * patch_size**2)`):
                Patches of all images, each image in merge-block order, in the layout of the Qwen2-VL image
                processor.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                Patch grid `(1, height, width)` of each image.

        Returns:
            `tuple(list[torch.FloatTensor], torch.FloatTensor)`:
            - `vision_num_hidden_layers + 1` tensors of shape `(total_patches, vision_hidden_size)`: the patch
              embeddings and the output of every block, row-major within each image and LayerNorm'd.
            - the last block's output, `(total_patches, vision_hidden_size)` in merge-block order and not normalized:
              the input of Qwen3.5's patch merger.
        """
        grids = image_grid_thw.tolist()
        if any(t != 1 for t, _, _ in grids):
            raise ValueError(f"only images (temporal grid 1) are supported, got {grids}")
        device = pixel_values.device
        lengths = [h * w for _, h, w in grids]
        perms, offset = [], 0
        for (_, h, w), n in zip(grids, lengths):
            perms.append(vision_rowmajor_permutation((h, w), self.spatial_merge_size, device) + offset)
            offset += n
        if offset != pixel_values.shape[0]:
            raise ValueError(f"image_grid_thw covers {offset} patches, pixel_values has {pixel_values.shape[0]}")
        perm = torch.cat(perms)
        cu_seqlens = torch.zeros(len(lengths) + 1, device=device, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(lengths, device=device, dtype=torch.int32).cumsum(0)

        hidden = self.patch_embed(pixel_values)
        idx, w = vision_bilinear(image_grid_thw, self.num_grid_per_side, self.spatial_merge_size)
        hidden = hidden + (self.pos_embed(idx) * w[:, :, None]).sum(0)
        rot = self.rotary_pos_emb(vision_position_ids(image_grid_thw, self.spatial_merge_size)).reshape(offset, -1)
        emb = torch.cat((rot, rot), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        hidden_states = [_normalize(hidden[perm])]
        for block in self.blocks:
            hidden = block(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
            hidden_states.append(_normalize(hidden[perm]))
        return hidden_states, hidden


@dataclass
class IntegralVisionTokenizerOutput(ModelOutput):
    r"""
    latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`):
        Mean of the VAE posterior, standardized with the VAE's `latent_mean` / `latent_std`.
    z_hat (`torch.FloatTensor` of shape `(total_patches, vae_hidden_size)`):
        VAE output, decoded from a posterior sample in training and from the mean in eval.
    kl_loss (`torch.FloatTensor`):
        KL divergence of the posterior from a standard normal.
    merger_output (`torch.FloatTensor` of shape `(total_merged_patches, merger_out_hidden_size)`):
        Merger output from `z_hat`, `total_patches / spatial_merge_size**2` rows.
    reconstructed_images (`list[torch.FloatTensor]`):
        Pixel decoder output from `z_hat`, one `(3, height, width)` tensor per image.
    vision_last_hidden_state (`torch.FloatTensor` of shape `(total_patches, vision_hidden_size)`):
        The vision tower's last block output in merge-block order, not normalized: the input of Qwen3.5's patch
        merger.
    """

    latents: Optional[Tensor] = None
    z_hat: Optional[Tensor] = None
    kl_loss: Optional[Tensor] = None
    merger_output: Optional[Tensor] = None
    reconstructed_images: Optional[list] = None
    vision_last_hidden_state: Optional[Tensor] = None


class IntegralLayerNorm(nn.LayerNorm):
    """LayerNorm that returns the input dtype."""

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_dtype)


class IntegralSwiGLU(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(in_size, 2 * hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, out_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class IntegralFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class IntegralSwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size)
        self.w2 = nn.Linear(hidden_size, intermediate_size)
        self.w3 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class IntegralSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: Tensor, rope: Optional[tuple] = None) -> Tensor:
        """
        Args:
            x (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`)
            rope (`tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                `(cos, sin)` from `rope_2d`, each `(1 or batch_size, seq_len, head_dim)`.
        """
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if rope is not None:
            q, k = apply_rotary_pos_emb(q, *rope), apply_rotary_pos_emb(k, *rope)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class IntegralTransformerBlock(nn.Module):
    """Pre-norm transformer block. Abstraction fusion uses the GELU MLP without qkv bias; the pixel
    decoder uses the SwiGLU MLP with qkv bias."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float,
                 qkv_bias: bool, use_swiglu: bool):
        super().__init__()
        self.norm1 = IntegralLayerNorm(hidden_size, eps=eps)
        self.attn = IntegralSelfAttention(hidden_size, num_heads, qkv_bias)
        self.norm2 = IntegralLayerNorm(hidden_size, eps=eps)
        self.mlp = (IntegralSwiGLUFeedForward(hidden_size, intermediate_size) if use_swiglu
                    else IntegralFeedForward(hidden_size, intermediate_size))

    def forward(self, x: Tensor, rope: Optional[tuple] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


def rope_2d_frequencies(rotary_dim: int, theta: float, device=None) -> Tensor:
    """The `rotary_dim // 4` frequencies of the 2D rotary embedding, `theta ** (i / (rotary_dim // 4))`, in fp32."""
    n = rotary_dim // 4
    return theta ** (torch.arange(n, device=device, dtype=torch.float32) / n)


def rope_2d(frequencies: Tensor, grids: list, num_patches: int) -> tuple:
    """
    `(cos, sin)` of the 2D rotary embedding of patch grids, in fp32 on the device of `frequencies`. Patch coordinates
    are normalized to [-1, 1]; of the `rotary_dim = 4 * len(frequencies)` angles, the first half is the row axis then
    the column axis, and the second half repeats it.

    Args:
        frequencies (`torch.FloatTensor` of shape `(rotary_dim // 4,)`):
            From `rope_2d_frequencies`.
        grids (`list[tuple[int, int]]`):
            Patch grid `(height, width)` of each target.
        num_patches (`int`):
            The padded length of every target.

    Returns:
        `tuple[torch.FloatTensor, torch.FloatTensor]`, each `(1, num_patches, rotary_dim)` when every grid is the same
        and has `num_patches` patches, otherwise `(len(grids), num_patches, rotary_dim)` and zero past each grid's
        patches.
    """
    def table(gh: int, gw: int) -> tuple:
        device = frequencies.device
        coords_h = torch.arange(0.5, gh, device=device, dtype=torch.float32) / gh
        coords_w = torch.arange(0.5, gw, device=device, dtype=torch.float32) / gw
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = (2 * math.pi * coords[:, :, None] / frequencies.float()[None, None, :]).flatten(1, 2).tile(2)
        return angles.cos(), angles.sin()

    if len(set(grids)) == 1 and grids[0][0] * grids[0][1] == num_patches:
        cos, sin = table(*grids[0])
        return cos[None], sin[None]
    rotary_dim = 4 * frequencies.shape[0]
    cos = torch.zeros(len(grids), num_patches, rotary_dim, device=frequencies.device, dtype=torch.float32)
    sin = torch.zeros(len(grids), num_patches, rotary_dim, device=frequencies.device, dtype=torch.float32)
    for b, (gh, gw) in enumerate(grids):
        cos[b, :gh * gw], sin[b, :gh * gw] = table(gh, gw)
    return cos, sin


class IntegralRoPE2D(nn.Module):
    """2D rotary embedding (`rope_2d`) over the whole head, frequencies in the persistent `freqs` buffer."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        assert head_dim % 4 == 0, head_dim
        self.register_buffer("freqs", rope_2d_frequencies(head_dim, theta), persistent=True)

    def forward(self, grids: list, num_patches: int) -> tuple:
        """`(cos, sin)` of `rope_2d`."""
        return rope_2d(self.freqs, grids, num_patches)


def abstraction_fusion_group_sizes(num_hidden_states: int, group_size: int) -> list:
    """Hidden states per abstraction group, shallow to deep: `ceil(num_hidden_states / group_size)`
    groups as even as possible, the shorter ones deepest. 28 hidden states at group size 4 make
    seven groups of four."""
    num_groups = math.ceil(num_hidden_states / group_size)
    base, remainder = divmod(num_hidden_states, num_groups)
    return [base + 1] * remainder + [base] * (num_groups - remainder)


class IntegralAbstractionAttention(nn.Module):
    """Self-attention across the group axis of `[B * N, num_groups, C]`. Batches larger than
    `MAX_BATCH` are split to stay under CUDA's grid size limit."""

    MAX_BATCH = 32768

    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, intermediate_size: int, eps: float):
        super().__init__()
        self.layers = nn.ModuleList([IntegralTransformerBlock(hidden_size, num_heads, intermediate_size, eps,
                                                              qkv_bias=False, use_swiglu=False)
                                     for _ in range(num_layers)])
        self.norm = IntegralLayerNorm(hidden_size, eps=eps, elementwise_affine=False)

    def forward(self, x: Tensor) -> Tensor:
        outputs = []
        for chunk in x.split(self.MAX_BATCH, dim=0):
            for layer in self.layers:
                chunk = layer(chunk)
            outputs.append(self.norm(chunk))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


class IntegralAbstractionFusion(nn.Module):
    """Abstraction fusion: vision tower hidden states -> one token per patch.

        hidden states -> per-layer standardization (layer_mean / layer_std)
          -> one projection per group of `abstraction_fusion_group_size` hidden states (projs)
          -> + level_embed -> self-attention across the groups (attention)
          -> linear pooler (pooler)

    `layer_mean` / `layer_std` are persistent, non-trainable buffers.
    """

    def __init__(self, config: IntegralVisionTokenizerConfig):
        super().__init__()
        vision_size, hidden_size = config.vision_hidden_size, config.abstraction_fusion_hidden_size
        self.num_hidden_states = config.num_vision_hidden_states
        self.group_sizes = abstraction_fusion_group_sizes(self.num_hidden_states, config.abstraction_fusion_group_size)
        num_groups = len(self.group_sizes)
        self.register_buffer("layer_mean", torch.zeros(self.num_hidden_states, vision_size), persistent=True)
        self.register_buffer("layer_std", torch.ones(self.num_hidden_states, vision_size), persistent=True)
        self.projs = nn.ModuleList([nn.Linear(size * vision_size, hidden_size, bias=False)
                                    for size in self.group_sizes])
        self.level_embed = nn.Parameter(torch.zeros(1, num_groups, hidden_size))
        self.attention = IntegralAbstractionAttention(hidden_size, config.abstraction_fusion_num_hidden_layers,
                                                      config.abstraction_fusion_num_attention_heads,
                                                      config.abstraction_fusion_intermediate_size,
                                                      config.layer_norm_eps)
        self.pooler = nn.Linear(num_groups * hidden_size, hidden_size, bias=False)

    @torch.no_grad()
    def set_layer_statistics(self, mean: Tensor, std: Tensor) -> None:
        """
        Sets the per-layer statistics of the vision tower hidden states.

        Args:
            mean, std (`torch.FloatTensor` of shape `(num_vision_hidden_states, vision_hidden_size)`)
        """
        assert tuple(mean.shape) == tuple(self.layer_mean.shape), (mean.shape, self.layer_mean.shape)
        assert tuple(std.shape) == tuple(self.layer_std.shape), (std.shape, self.layer_std.shape)
        self.layer_mean.copy_(mean)
        self.layer_std.copy_(std)

    def forward(self, hidden_states: list) -> Tensor:
        """
        Args:
            hidden_states (`list[torch.FloatTensor]`):
                `num_vision_hidden_states` tensors of shape `(num_patches, vision_hidden_size)`.

        Returns:
            `torch.FloatTensor` of shape `(num_patches, abstraction_fusion_hidden_size)`.
        """
        assert len(hidden_states) == self.num_hidden_states, \
            f"{len(hidden_states)} hidden states, config says {self.num_hidden_states}"
        N, C = hidden_states[-1].shape
        x = torch.stack(list(hidden_states), dim=-1)                                 # [N, C, L]
        layers = ((x.transpose(1, 2) - self.layer_mean) / self.layer_std).to(self.projs[0].weight.dtype)  # [N, L, C]
        groups, offset = [], 0
        for proj, size in zip(self.projs, self.group_sizes):
            groups.append(proj(layers[:, offset:offset + size, :].reshape(N, size * C)))
            offset += size
        groups = torch.stack(groups, dim=1) + self.level_embed                       # [N, G, C]
        groups = self.attention(groups)
        return self.pooler(groups.reshape(N, -1))


class IntegralLatentVAE(nn.Module):
    """[..., C] -> mu, logvar [..., latent_size] -> z -> [..., C]. z is a posterior sample in training and
    mu in eval.

    The latent exposed outside the VAE is mu standardized with `latent_mean` / `latent_std`, persistent
    non-trainable buffers: `(mu - latent_mean) / latent_std`. `decode` takes that latent.
    """

    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        self.encoder = nn.Linear(hidden_size, 2 * latent_size)
        self.decoder = nn.Linear(latent_size, hidden_size)
        self.register_buffer("latent_mean", torch.zeros(latent_size), persistent=True)
        self.register_buffer("latent_std", torch.ones(latent_size), persistent=True)

    @torch.no_grad()
    def set_latent_statistics(self, mean: Tensor, std: Tensor) -> None:
        """
        Sets the statistics of mu used for standardization.

        Args:
            mean, std (`torch.FloatTensor` of shape `(latent_size,)`)
        """
        if mean.shape != self.latent_mean.shape or std.shape != self.latent_std.shape:
            raise ValueError(f"statistics of shape {tuple(mean.shape)}, {tuple(std.shape)}; "
                             f"expected {tuple(self.latent_mean.shape)}")
        self.latent_mean.copy_(mean)
        self.latent_std.copy_(std)

    def forward(self, x: Tensor) -> tuple:
        """Returns `(z_hat, kl, latents)`, `latents` being the standardized mu."""
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if self.training else mu
        z_hat = self.decoder(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return z_hat, kl, (mu - self.latent_mean) / self.latent_std

    def mean(self, x: Tensor) -> Tensor:
        """`x` -> mu, without sampling, `z_hat` or the KL."""
        return self.encoder(x).chunk(2, dim=-1)[0]

    def standardize(self, mu: Tensor) -> Tensor:
        """mu -> `(mu - latent_mean) / latent_std`."""
        return (mu - self.latent_mean) / self.latent_std

    def encode(self, x: Tensor) -> Tensor:
        """`x` -> the standardized mu `(mu - latent_mean) / latent_std`, without sampling, `z_hat` or the KL."""
        return self.standardize(self.mean(x))

    def decode(self, latents: Tensor) -> Tensor:
        """Standardized latents -> `decoder(latents * latent_std + latent_mean)`."""
        return self.decoder(latents.float() * self.latent_std + self.latent_mean)


class IntegralBottleneck(nn.Module):
    """pre_proj -> post_proj + token_embed -> LayerNorm, the input of the VAE (`vae`)."""

    def __init__(self, config: IntegralVisionTokenizerConfig):
        super().__init__()
        vae_size = config.vae_hidden_size
        self.pre_proj = IntegralSwiGLU(config.abstraction_fusion_hidden_size,
                                       config.bottleneck_pre_intermediate_size, config.bottleneck_hidden_size)
        self.post_proj = IntegralSwiGLU(config.bottleneck_hidden_size, config.bottleneck_post_intermediate_size,
                                        vae_size)
        self.token_embed = nn.Parameter(torch.zeros(1, vae_size))
        self.norm = nn.LayerNorm(vae_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.vae = IntegralLatentVAE(vae_size, config.latent_size)

    def forward(self, x: Tensor) -> Tensor:
        """`(num_patches, abstraction_fusion_hidden_size)` -> the VAE's input `(num_patches, vae_hidden_size)`."""
        return self.norm(self.post_proj(self.pre_proj(x)) + self.token_embed)


def _merge_blocks(x: Tensor, grid_hw: tuple, s: int) -> Tensor:
    """[B, gh*gw, C] row-major -> [B, (gh/s)*(gw/s), s*s*C], each s x s block concatenated in
    row-major order within the block."""
    gh, gw = grid_hw
    B, N, C = x.shape
    assert N == gh * gw, f"patch count {N} != patch grid {gh}x{gw}"
    assert gh % s == 0 and gw % s == 0, f"patch grid {gh}x{gw} not divisible by {s}"
    x = x.reshape(B, gh // s, s, gw // s, s, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, (gh // s) * (gw // s), s * s * C)


class IntegralVisionPatchMerger(nn.Module):
    """`z_hat` -> language model input embeddings, one per s x s block of patches.

        in_proj (SwiGLU) + in_bias -> LayerNorm -> s x s merge -> linear_fc1 -> GELU -> linear_fc2
    """

    def __init__(self, config: IntegralVisionTokenizerConfig):
        super().__init__()
        hidden_size = config.merger_hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        merged_size = hidden_size * self.spatial_merge_size ** 2
        self.in_proj = IntegralSwiGLU(config.vae_hidden_size, config.merger_intermediate_size, hidden_size)
        self.in_bias = nn.Parameter(torch.zeros(hidden_size))
        self.norm = nn.LayerNorm(hidden_size, eps=config.merger_layer_norm_eps)
        self.linear_fc1 = nn.Linear(merged_size, merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(merged_size, config.merger_out_hidden_size)

    def forward(self, z_hat: Tensor, grid_hw: tuple) -> Tensor:
        """
        Args:
            z_hat (`torch.FloatTensor` of shape `(batch_size, num_patches, vae_hidden_size)`)
            grid_hw (`tuple[int, int]`):
                Patch grid `(height, width)` of the images.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_merged_patches, merger_out_hidden_size)`.
        """
        x = self.norm(self.in_proj(z_hat) + self.in_bias)
        x = _merge_blocks(x, grid_hw, self.spatial_merge_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x))).float()


class IntegralVisualGenerationAdapter(nn.Module):
    """Standardized latents -> one key/value delta per visual generation block, at the merged patch positions.

        latents [B, N, latent_size] -> s x s merge [B, N / s**2, s**2 * latent_size]
          -> up_projs[j] -> GELU -> kv_projs[j] -> key, value

    `kv_projs` are initialized to zero.
    """

    def __init__(self, config: IntegralVisionTokenizerConfig):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.num_key_value_heads = config.visual_generation_adapter_num_key_value_heads
        self.head_dim = config.visual_generation_adapter_head_dim
        merged_size = config.latent_size * self.spatial_merge_size ** 2
        intermediate_size = config.visual_generation_adapter_intermediate_size
        out_size = 2 * self.num_key_value_heads * self.head_dim
        num_layers = config.visual_generation_adapter_num_layers
        self.up_projs = nn.ModuleList([nn.Linear(merged_size, intermediate_size) for _ in range(num_layers)])
        self.act_fn = nn.GELU()
        self.kv_projs = nn.ModuleList([nn.Linear(intermediate_size, out_size) for _ in range(num_layers)])
        for kv_proj in self.kv_projs:
            kv_proj._zero_init = True        # read by `_init_weights`

    def forward(self, latents: Tensor, grid_hw: tuple) -> tuple:
        """
        Args:
            latents (`torch.FloatTensor` of shape `(batch_size, num_patches, latent_size)`):
                Standardized latents.
            grid_hw (`tuple[int, int]`):
                Patch grid `(height, width)` of the images.

        Returns:
            `tuple[torch.FloatTensor]` of length `visual_generation_adapter_num_layers`, one per visual generation
            block, each of shape `(batch_size, num_merged_patches, 2, visual_generation_adapter_num_key_value_heads,
            visual_generation_adapter_head_dim)`: the key (`[:, :, 0]`) and value (`[:, :, 1]`) delta.
        """
        x = _merge_blocks(latents, grid_hw, self.spatial_merge_size)
        B, n, _ = x.shape
        # one tensor per block, not stacked: reentrant gradient checkpointing backpropagates through each block's
        # deltas separately, so they must not share an autograd graph
        return tuple(kv_proj(self.act_fn(up_proj(x))).view(B, n, 2, self.num_key_value_heads, self.head_dim)
                     for up_proj, kv_proj in zip(self.up_projs, self.kv_projs))


class IntegralPixelDecoder(nn.Module):
    """`z_hat` -> pixels.

        in_proj (SwiGLU) -> [B, C, gh, gw] -> conv_in -> transformer blocks with 2D RoPE -> LayerNorm
          -> conv_out -> pixel shuffle -> de-normalization with `image_mean` / `image_std`
    """

    def __init__(self, config: IntegralVisionTokenizerConfig):
        super().__init__()
        hidden_size = config.pixel_decoder_hidden_size
        num_heads = config.pixel_decoder_num_attention_heads
        self.hidden_size = hidden_size
        self.image_mean = list(config.image_mean)
        self.image_std = list(config.image_std)
        self.in_proj = IntegralSwiGLU(config.vae_hidden_size, config.pixel_decoder_in_intermediate_size,
                                      config.pixel_decoder_in_hidden_size)
        self.conv_in = nn.Conv2d(config.pixel_decoder_in_hidden_size, hidden_size, kernel_size=1)
        self.rope = IntegralRoPE2D(hidden_size // num_heads, config.pixel_decoder_rope_theta)
        self.blocks = nn.ModuleList([IntegralTransformerBlock(hidden_size, num_heads,
                                                              config.pixel_decoder_intermediate_size,
                                                              config.layer_norm_eps, qkv_bias=True, use_swiglu=True)
                                     for _ in range(config.pixel_decoder_num_hidden_layers)])
        self.norm = IntegralLayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.conv_out = nn.Conv2d(hidden_size, 3 * config.patch_size ** 2, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(config.patch_size)

    def forward(self, z_hat: Tensor, grid_hw: tuple) -> Tensor:
        """
        Args:
            z_hat (`torch.FloatTensor` of shape `(batch_size, num_patches, vae_hidden_size)`)
            grid_hw (`tuple[int, int]`):
                Patch grid `(height, width)` of the images.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, 3, height, width)`, in [0, 1] and not clamped.
        """
        x = self.in_proj(z_hat)
        B, N, C = x.shape
        gh, gw = grid_hw
        if N != gh * gw:
            raise ValueError(f"patch count {N} != patch grid {gh}x{gw}")
        x = self.conv_in(x.transpose(1, 2).reshape(B, C, gh, gw)).flatten(2).transpose(1, 2)
        rope = tuple(t.to(x.dtype) for t in self.rope([grid_hw], N))
        for block in self.blocks:
            x = block(x, rope)
        x = self.norm(x).transpose(1, 2).reshape(B, self.hidden_size, gh, gw)
        x = self.pixel_shuffle(self.conv_out(x))
        std = torch.tensor(self.image_std, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        mean = torch.tensor(self.image_mean, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
        return x * std + mean


class IntegralVisionTokenizerPreTrainedModel(PreTrainedModel):
    config_class = IntegralVisionTokenizerConfig
    base_model_prefix = "vision_tokenizer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["IntegralTransformerBlock", "IntegralVisionBlock"]

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if getattr(module, "_zero_init", False):
                module.weight.zero_()
            else:
                module.weight.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.normal_(mean=0.0, std=std)
        elif isinstance(module, IntegralVisionRotaryEmbedding):
            # not saved in checkpoints
            module.inv_freq.copy_(module.compute_inv_freq().to(module.inv_freq.device))
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.zero_()
            if module.weight is not None:
                module.weight.fill_(1.0)


def _per_image(fn, x: Tensor, image_grid_thw: Tensor) -> list:
    """
    Splits `x` (`(total_patches, ...)`, images concatenated) into images by `image_grid_thw`, calls
    `fn(batch, (height, width))` once per distinct patch grid with that grid's images stacked, and returns
    one output per image in input order. `fn` returns a tensor or a tuple of tensors with the image batch
    as the first dimension.
    """
    grids = [(h, w) for _, h, w in image_grid_thw.tolist()]
    parts = torch.split(x, [h * w for h, w in grids])
    images_by_grid = {}
    for i, grid in enumerate(grids):
        images_by_grid.setdefault(grid, []).append(i)
    outputs = [None] * len(grids)
    for grid, indices in images_by_grid.items():
        y = fn(torch.stack([parts[i] for i in indices]), grid)
        for j, i in enumerate(indices):
            outputs[i] = tuple(t[j] for t in y) if isinstance(y, tuple) else y[j]
    return outputs


class IntegralVisionTokenizer(IntegralVisionTokenizerPreTrainedModel):
    """Image patches -> standardized latents -> language model input embeddings, key/value deltas and pixels.

        patches -> vision tower (`visual`, frozen) -> hidden states of the patch embeddings and every block
                -> abstraction fusion (`abstraction_fusion`)                      one token per patch
                -> bottleneck: pre_proj -> post_proj + token_embed -> LayerNorm -> VAE
                -> latents [total_patches, latent_size], the VAE mean standardized with `latent_mean` / `latent_std`

        latents -> vae.decode -> merger                     -> language model input embeddings
        latents -> visual_generation_adapter                -> key/value deltas for the visual generation blocks
        latents -> vae.decode -> pixel_decoder              -> pixels

    `encode`, `merge`, `adapt` and `decode` are the inference path; `encode_and_merge` gives the latents and the
    embeddings of the same images in one pass, the merger reading the VAE decoder of mu itself. `forward` is the training path: the VAE output
    `z_hat` (from a posterior sample in training mode) goes to the pixel decoder and the merger, and the KL is
    computed.

    Images are passed as the patches of all images concatenated, with `image_grid_thw` giving each image's
    patch grid, as produced by the Qwen2-VL image processor. s = `spatial_merge_size`.
    """

    def __init__(self, config: IntegralVisionTokenizerConfig):
        super().__init__(config)
        self.visual = IntegralVisionTower(config)
        self.abstraction_fusion = IntegralAbstractionFusion(config)
        self.bottleneck = IntegralBottleneck(config)
        self.merger = IntegralVisionPatchMerger(config)
        self.visual_generation_adapter = IntegralVisualGenerationAdapter(config)
        self.pixel_decoder = IntegralPixelDecoder(config)

        self.post_init()
        self.visual.requires_grad_(False)

    def encode(self, pixel_values: Tensor, image_grid_thw: Tensor) -> Tensor:
        """
        Encodes images into standardized latents. The vision tower runs under `no_grad`.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(total_patches, 3 * temporal_patch_size * patch_size**2)`):
                Patches of all images, normalized with `config.image_mean` and `config.image_std`, in the layout
                of the Qwen2-VL image processor.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                Patch grid `(1, height, width)` of each image.

        Returns:
            `torch.FloatTensor` of shape `(total_patches, latent_size)`: the VAE mean standardized with
            `latent_mean` / `latent_std`.
        """
        with torch.no_grad():
            hidden_states, _ = self.visual(pixel_values, image_grid_thw)
        return self.bottleneck.vae.encode(self.bottleneck(self.abstraction_fusion(hidden_states)))

    def encode_and_merge(self, pixel_values: Tensor, image_grid_thw: Tensor) -> tuple:
        """
        Encodes images into standardized latents and language model input embeddings in one pass: the embeddings
        are the merger of the VAE decoder of mu. The vision tower runs under `no_grad`.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(total_patches, 3 * temporal_patch_size * patch_size**2)`):
                Patches of all images, as in `encode`.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                Patch grid `(1, height, width)` of each image.

        Returns:
            `tuple(torch.FloatTensor)`: the standardized latents `(total_patches, latent_size)` and the language model
            input embeddings `(total_merged_patches, merger_out_hidden_size)`, images in input order.
        """
        with torch.no_grad():
            hidden_states, _ = self.visual(pixel_values, image_grid_thw)
        vae = self.bottleneck.vae
        mu = vae.mean(self.bottleneck(self.abstraction_fusion(hidden_states)))
        embeddings = torch.cat(_per_image(self.merger, vae.decoder(mu), image_grid_thw))
        return vae.standardize(mu), embeddings

    def merge(self, latents: Tensor, image_grid_thw: Tensor) -> Tensor:
        """
        Computes language model input embeddings from standardized latents.

        Args:
            latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`)
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`)

        Returns:
            `torch.FloatTensor` of shape `(total_merged_patches, merger_out_hidden_size)`, images in input order.
        """
        return torch.cat(_per_image(self.merger, self.bottleneck.vae.decode(latents), image_grid_thw))

    def adapt(self, latents: Tensor, image_grid_thw: Tensor) -> list:
        """
        Computes one key/value delta per visual generation block from standardized latents. The latents are
        detached first.

        Args:
            latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`)
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`)

        Returns:
            `list[tuple[torch.FloatTensor, torch.FloatTensor]]` of length `visual_generation_adapter_num_layers`,
            one `(key, value)` per visual generation block, each of shape
            `(total_merged_patches, visual_generation_adapter_num_key_value_heads, visual_generation_adapter_head_dim)`,
            images in input order.
        """
        per_image = _per_image(self.visual_generation_adapter, latents.float().detach(), image_grid_thw)
        return [(delta[:, 0], delta[:, 1]) for delta in (torch.cat(layer) for layer in zip(*per_image))]

    def decode(self, latents: Tensor, image_grid_thw: Tensor) -> list:
        """
        Decodes standardized latents into pixels.

        Args:
            latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`)
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`)

        Returns:
            `list[torch.FloatTensor]`, one `(3, height, width)` image per input image, in [0, 1] and not clamped.
        """
        return _per_image(self.pixel_decoder, self.bottleneck.vae.decode(latents), image_grid_thw)

    def forward(self, pixel_values: Tensor, image_grid_thw: Tensor, with_reconstruction: bool = True,
                with_merger: bool = True) -> IntegralVisionTokenizerOutput:
        """
        Encodes images, then runs the pixel decoder and the merger on `z_hat`, the VAE output decoded from a
        posterior sample in training mode and from the mean in eval mode. The vision tower runs under `no_grad`.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(total_patches, 3 * temporal_patch_size * patch_size**2)`):
                As in `encode`.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
                As in `encode`.
            with_reconstruction (`bool`, *optional*, defaults to `True`):
                Whether to return `reconstructed_images`.
            with_merger (`bool`, *optional*, defaults to `True`):
                Whether to return `merger_output`.

        Returns:
            [`IntegralVisionTokenizerOutput`].
        """
        with torch.no_grad():
            hidden_states, vision_last_hidden_state = self.visual(pixel_values, image_grid_thw)
        z_hat, kl_loss, latents = self.bottleneck.vae(self.bottleneck(self.abstraction_fusion(hidden_states)))
        output = IntegralVisionTokenizerOutput(latents=latents, z_hat=z_hat, kl_loss=kl_loss,
                                               vision_last_hidden_state=vision_last_hidden_state)
        if with_reconstruction:
            output.reconstructed_images = _per_image(self.pixel_decoder, z_hat, image_grid_thw)
        if with_merger:
            output.merger_output = torch.cat(_per_image(self.merger, z_hat, image_grid_thw))
        return output


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """Sinusoidal embedding `[cos, sin]` of `t` (any shape) -> `(..., dim)`, in fp32."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


def sincos_position_embedding_2d(dim: int, height: int, width: int) -> Tensor:
    """Fixed 2D sinusoidal position embedding of a `height x width` grid, row-major -> `(1, height * width, dim)`:
    the first half of the channels encodes the row, the second half the column."""
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32) / d))

    def table(n: int) -> Tensor:
        out = torch.arange(n, dtype=torch.float32)[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], dim=1)

    emb_h, emb_w = table(height), table(width)
    pe = torch.cat([emb_h[:, None, :].expand(height, width, 2 * d), emb_w[None, :, :].expand(height, width, 2 * d)],
                   dim=-1)
    return pe.reshape(1, height * width, dim)


@dataclass
class IntegralVisualKeyValues(ModelOutput):
    r"""
    The language model key/value states the visual generation blocks read, one entry per layer in
    `visual_generation_config.key_value_layer_indices`.

    key_states (`tuple[torch.FloatTensor]`):
        Each `(batch_size, num_key_value_heads, sequence_length, head_dim)`, after `k_norm` and the rotary embedding,
        with the visual generation adapter's deltas added at the image tokens before `k_norm`.
    value_states (`tuple[torch.FloatTensor]`):
        Each `(batch_size, num_key_value_heads, sequence_length, head_dim)`, with the deltas added at the image tokens.
    attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        `1` for the positions that may be attended to.
    """

    key_states: tuple | None = None
    value_states: tuple | None = None
    attention_mask: Tensor | None = None


class IntegralVisualAttentionLayout:
    """
    The targets' patches laid end to end, and what each target's patches attend to in the visual generation blocks:
    the language model positions of its row in `condition_mask`, and its own patches. Built once per forward, with the
    metadata of one attention implementation, and read by every block and the flow head.

    The visual generation model computes on the real patches only, `(1, total_patches, ...)` with target `i` at
    `segments[i]`. `pad` and `unpad` convert to and from `(num_targets, num_patches, ...)`; when every target has the
    same grid and `num_patches` patches, both are reshapes.

    Args:
        condition_mask (`torch.BoolTensor` of shape `(num_targets, sequence_length)`):
            The language model positions each target attends to.
        grids (`list[tuple[int, int]]`):
            Patch grid `(height, width)` of each target.
        batch_index (`torch.LongTensor` of shape `(num_targets,)`, *optional*):
            The language model row each target reads. Target `i` reads row `i` if not given.
        num_patches (`int`):
            `max_num_patches`, the padded length of every target.
        attn_implementation (`str`):
            A key of `VISUAL_GENERATION_ATTENTION_FUNCTIONS`. Its metadata is built; the other metadata attributes below
            are `None`.

    Attributes:
        latent_mask (`torch.BoolTensor` of shape `(num_targets, num_patches)`):
            The real patches of each target. `None` when every target has the same grid and `num_patches` patches.
        patch_index (`torch.LongTensor` of shape `(total_patches,)`):
            Position of each real patch in the flattened `(num_targets * num_patches)` layout. `None` with
            `latent_mask`.
        segments (`list[tuple[int, int]]`):
            `[start, end)` of each target's patches in the patch sequence.
        sdpa_mask (`torch.BoolTensor` of shape `(num_targets, 1, num_patches, sequence_length + num_patches)`):
            `True` where a patch may attend, for `"eager"` and `"sdpa"`.
        varlen (`dict`):
            For `"flash_attention_2"`, indices and sequence lengths of variable-length attention over the patches and
            the condition positions: `condition_rows`, `condition_positions` (the condition positions of every target,
            target by target), `key_order` (the order of `[condition positions; patches]` that puts each target's keys
            together), `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`. `cu_seqlens_q` and
            `max_seqlen_q` also delimit the targets in the flow head's self-attention.
        flex_block_masks (`tuple`):
            For `"flex_attention"`, `(condition block mask, patch block mask or None)`.
    """

    def __init__(self, condition_mask: Tensor, grids: list, batch_index: Optional[Tensor], num_patches: int,
                 attn_implementation: str):
        if attn_implementation not in VISUAL_GENERATION_ATTENTION_FUNCTIONS:
            raise ValueError(f"The visual generation blocks support {sorted(VISUAL_GENERATION_ATTENTION_FUNCTIONS)}, "
                             f"not {attn_implementation}")
        self.attn_implementation = attn_implementation
        self.condition_mask = condition_mask.to(torch.bool)
        self.grids = list(grids)
        self.batch_index = batch_index
        self.num_patches = num_patches
        self.sdpa_mask = self.varlen = self.flex_block_masks = None

        N, L = self.condition_mask.shape
        T = num_patches
        device = self.condition_mask.device
        lengths = [h * w for h, w in self.grids]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        self.segments = list(zip(offsets[:-1], offsets[1:]))
        self.latent_mask = self.patch_index = None
        if not (len(set(self.grids)) == 1 and lengths[0] == T):
            self.latent_mask = torch.arange(T, device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
            self.patch_index = torch.cat([torch.arange(i * T, i * T + n) for i, n in enumerate(lengths)]).to(device)
        patch_mask = (self.latent_mask if self.latent_mask is not None
                      else torch.ones(N, T, dtype=torch.bool, device=device))
        if attn_implementation in ("eager", "sdpa"):
            keep = torch.cat([self.condition_mask, patch_mask], dim=1)
            self.sdpa_mask = keep[:, None, None, :].expand(N, 1, T, L + T)
        elif attn_implementation == "flash_attention_2":
            condition = self.condition_mask.nonzero()
            row = batch_index if batch_index is not None else torch.arange(N, device=device)
            condition_lengths = self.condition_mask.sum(1)
            patch_lengths = torch.tensor(lengths, device=device)
            condition_offsets = [0] + condition_lengths.cumsum(0).tolist()
            total_condition = condition_offsets[-1]
            key_order = torch.cat([torch.cat([torch.arange(condition_offsets[i], condition_offsets[i + 1]),
                                              total_condition + torch.arange(*self.segments[i])])
                                   for i in range(N)]).to(device)
            zero = torch.zeros(1, dtype=torch.int32, device=device)
            self.varlen = dict(
                condition_rows=row[condition[:, 0]], condition_positions=condition[:, 1], key_order=key_order,
                cu_seqlens_q=torch.cat([zero, patch_lengths.cumsum(0).to(torch.int32)]),
                cu_seqlens_k=torch.cat([zero, (condition_lengths + patch_lengths).cumsum(0).to(torch.int32)]),
                max_seqlen_q=max(lengths), max_seqlen_k=int((condition_lengths + patch_lengths).max()))
        else:
            from torch.nn.attention.flex_attention import create_block_mask

            condition_mask, latent_mask = self.condition_mask, self.latent_mask
            condition = create_block_mask(lambda b, h, q, k: condition_mask[b, k], B=N, H=None,
                                          Q_LEN=T, KV_LEN=L, device=device)
            patches = None
            if latent_mask is not None:
                patches = create_block_mask(lambda b, h, q, k: latent_mask[b, k], B=N, H=None,
                                            Q_LEN=T, KV_LEN=T, device=device)
            self.flex_block_masks = (condition, patches)

    def rows(self, states: Tensor) -> Tensor:
        """`(batch_size, ...)` language model states -> `(num_targets, ...)`, the row each target reads."""
        if self.batch_index is None:
            return states
        if states.shape[0] == 1:
            return states.expand(self.condition_mask.shape[0], *states.shape[1:])
        return states.index_select(0, self.batch_index)

    def unpad(self, x: Tensor) -> Tensor:
        """`(num_targets, num_patches, ...)` -> `(1, total_patches, ...)`, the real patches target by target."""
        if self.latent_mask is None:
            return x.reshape(1, -1, *x.shape[2:])
        return x.reshape(-1, *x.shape[2:]).index_select(0, self.patch_index)[None]

    def pad(self, x: Tensor) -> Tensor:
        """`(1, total_patches, ...)` -> `(num_targets, num_patches, ...)`, zero past each target's patches."""
        N, T = len(self.grids), self.num_patches
        if self.latent_mask is None:
            return x.reshape(N, T, *x.shape[2:])
        return x.new_zeros(N * T, *x.shape[2:]).index_copy(0, self.patch_index, x[0]).reshape(N, T, *x.shape[2:])

    def per_target(self, fn, x: Tensor) -> Tensor:
        """
        Applies `fn(patches, index, grid)` to each target's patches of `x` `(1, total_patches, ...)` and returns the
        results in the same layout. `index` selects the target's row of a per-target tensor; with no `latent_mask` all
        targets go in one call, as `(num_targets, num_patches, ...)` with `index` selecting every row.
        """
        if self.latent_mask is None:
            return self.unpad(fn(self.pad(x), slice(None), self.grids[0]))
        parts = x.split([end - start for start, end in self.segments], dim=1)
        return torch.cat([fn(part, slice(i, i + 1), self.grids[i]) for i, part in enumerate(parts)], dim=1)


def _attention_dtype(query: Tensor) -> torch.dtype:
    """The dtype attention runs in: the autocast dtype under autocast, the query's otherwise."""
    if torch.is_autocast_enabled(query.device.type):
        return torch.get_autocast_dtype(query.device.type)
    return query.dtype


def _padded_heads(layout, query: Tensor, key: Tensor, value: Tensor) -> tuple:
    """Patch-sequence `(1, total_patches, heads, head_dim)` query/key/value -> `(num_targets, heads, num_patches,
    head_dim)`, query and key contiguous."""
    query, key, value = (layout.pad(t).transpose(1, 2) for t in (query, key, value))
    return query.contiguous(), key.contiguous(), value


def visual_generation_sdpa_attention(module, query, key, value, condition_key, condition_value, layout):
    """One `scaled_dot_product_attention` over `[condition; patches]` per target, the patches padded to
    `num_patches`, every key/value repeated to the query heads, the condition positions and padding masked out."""
    query, key, value = _padded_heads(layout, query, key, value)
    num_heads = query.shape[1]
    keys = torch.cat([repeat_kv(layout.rows(condition_key).to(query.dtype), num_heads // condition_key.shape[1]),
                      repeat_kv(key, num_heads // key.shape[1])], dim=2)
    values = torch.cat([repeat_kv(layout.rows(condition_value).to(query.dtype), num_heads // condition_value.shape[1]),
                        repeat_kv(value, num_heads // value.shape[1])], dim=2)
    attn = F.scaled_dot_product_attention(query, keys, values, attn_mask=layout.sdpa_mask, scale=module.scaling)
    return layout.unpad(attn.transpose(1, 2))


def visual_generation_flash_attention(module, query, key, value, condition_key, condition_value, layout):
    """flash-attn varlen over each target's condition positions and patches: the condition positions are gathered
    from the language model rows and the keys/values repeated to a common head count."""
    from flash_attn import flash_attn_varlen_func

    dtype = _attention_dtype(query)
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"flash_attention_2 runs in float16 or bfloat16, not {dtype}; use autocast or another "
                         f"attention implementation")
    meta = layout.varlen
    num_key_value_heads = math.lcm(condition_key.shape[1], key.shape[2])

    def packed(condition, patches):
        condition = condition[meta["condition_rows"], :, meta["condition_positions"]]
        both = torch.cat([condition.repeat_interleave(num_key_value_heads // condition.shape[1], dim=1).to(dtype),
                          patches.repeat_interleave(num_key_value_heads // patches.shape[1], dim=1).to(dtype)])
        return both.index_select(0, meta["key_order"])

    out = flash_attn_varlen_func(
        query[0].to(dtype), packed(condition_key, key[0]), packed(condition_value, value[0]), meta["cu_seqlens_q"],
        meta["cu_seqlens_k"], meta["max_seqlen_q"], meta["max_seqlen_k"], softmax_scale=module.scaling, causal=False,
        deterministic=torch.are_deterministic_algorithms_enabled())
    return out[None]


def _flex_attention_with_lse(module, query, key, value, block_mask):
    from torch.nn.attention.flex_attention import flex_attention
    from transformers.integrations.flex_attention import (_TORCH_FLEX_USE_AUX, compile_friendly_flex_attention,
                                                          get_flex_attention_lse_kwargs)

    kwargs = dict(block_mask=block_mask, enable_gqa=True, scale=module.scaling, **get_flex_attention_lse_kwargs(True))
    if query.is_cuda:
        out = compile_friendly_flex_attention(query, key, value, training=module.training, **kwargs)
    else:
        out = flex_attention(query, key, value, **kwargs)
    if _TORCH_FLEX_USE_AUX:
        out, aux = out
        return out, aux.lse
    return out


def visual_generation_flex_attention(module, query, key, value, condition_key, condition_value, layout):
    """flex attention in two parts, the condition positions (read in place from the language model rows) and the
    patches padded to `num_patches`, combined with their log-sum-exp into the one softmax over both."""
    query, key, value = _padded_heads(layout, query, key, value)
    dtype = _attention_dtype(query)
    condition_mask, patch_mask = layout.flex_block_masks
    query = query.to(dtype)
    out_condition, lse_condition = _flex_attention_with_lse(
        module, query, layout.rows(condition_key).to(dtype), layout.rows(condition_value).to(dtype), condition_mask)
    out_patches, lse_patches = _flex_attention_with_lse(module, query, key.to(dtype), value.to(dtype), patch_mask)
    lse = torch.logaddexp(lse_condition, lse_patches)
    out = (out_condition.float() * (lse_condition - lse).exp()[..., None]
           + out_patches.float() * (lse_patches - lse).exp()[..., None])
    return layout.unpad(out.to(dtype).transpose(1, 2))


VISUAL_GENERATION_ATTENTION_FUNCTIONS = {
    "eager": visual_generation_sdpa_attention,
    "sdpa": visual_generation_sdpa_attention,
    "flash_attention_2": visual_generation_flash_attention,
    "flex_attention": visual_generation_flex_attention,
}


class IntegralVisualGenerationBlock(GradientCheckpointingLayer):
    """One visual generation block. Its queries attend, in one softmax, to the key/value states of one language
    model layer and to its own keys/values; adaLN-single modulation from the timestep."""

    def __init__(self, config: IntegralVisualGenerationConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, config.intermediate_size), nn.GELU(),
                                 nn.Linear(config.intermediate_size, hidden_size))
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, hidden_states: Tensor, key_states: Tensor, value_states: Tensor, modulation: Tensor,
                layout: IntegralVisualAttentionLayout, rope: tuple) -> Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(1, total_patches, hidden_size)`):
                Every target's patches, target by target (`layout.segments`).
            key_states, value_states (`torch.FloatTensor` of shape `(batch_size, num_key_value_heads, sequence_length, head_dim)`):
                The language model layer's visual key/value states, one row per language model row.
            modulation (`torch.FloatTensor` of shape `(num_targets, 6, hidden_size)`):
                adaLN-single modulation from the timestep.
            layout ([`IntegralVisualAttentionLayout`]):
                The targets' patches, the language model row and positions each target attends to, and the attention
                implementation.
            rope (`tuple[torch.FloatTensor, torch.FloatTensor]`):
                `(cos, sin)` of the patches from `rope_2d`, each `(1, total_patches, rotary_dim)`, applied in the dtype
                of `hidden_states` to the first `rotary_dim` channels of each head.
        """
        attention_interface = VISUAL_GENERATION_ATTENTION_FUNCTIONS[layout.attn_implementation]
        P = hidden_states.shape[1]
        cos, sin = (r.to(hidden_states.dtype) for r in rope)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + modulation).chunk(6, dim=1)
        h = layout.per_target(lambda x, i, _: self.norm1(x) * (1 + scale_attn[i]) + shift_attn[i], hidden_states)
        q = self.q_proj(h).view(1, P, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(1, P, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(h).view(1, P, self.num_key_value_heads, self.head_dim)
        q, k = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2), apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
        attn = self.o_proj(attention_interface(self, q, k, v, key_states, value_states, layout).reshape(1, P, -1))
        hidden_states = hidden_states + layout.per_target(lambda x, i, _: gate_attn[i] * x, attn)
        h = layout.per_target(lambda x, i, _: self.norm2(x) * (1 + scale_mlp[i]) + shift_mlp[i], hidden_states)
        return hidden_states + layout.per_target(lambda x, i, _: gate_mlp[i] * x, self.mlp(h))


class IntegralFlowHeadBlock(nn.Module):
    """Self-attention block of the flow head with per-patch adaLN-Zero modulation from the condition."""

    def __init__(self, config: IntegralVisualGenerationConfig):
        super().__init__()
        dim = config.flow_head_hidden_size
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=config.layer_norm_eps)
        self.attn = IntegralSelfAttention(dim, config.flow_head_num_attention_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(nn.Linear(dim, config.flow_head_intermediate_size), nn.GELU(),
                                 nn.Linear(config.flow_head_intermediate_size, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.ada[1]._zero_init = True

    def forward(self, x: Tensor, condition: Tensor, rope: tuple, layout: IntegralVisualAttentionLayout) -> Tensor:
        """`x`, `condition` `(1, total_patches, flow_head_hidden_size)`; `rope` `(cos, sin)`, each `(1, total_patches,
        head_dim)`."""
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada(condition).chunk(6, dim=-1)
        x = x + gate_attn * flow_head_attention(self.attn, self.norm1(x) * (1 + scale_attn) + shift_attn, rope, layout)
        return x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)


def flow_head_attention(attention: IntegralSelfAttention, x: Tensor, rope: tuple,
                        layout: IntegralVisualAttentionLayout) -> Tensor:
    """
    The flow head's self-attention of each target's patches among themselves, with `attention`'s projections:
    flash-attn varlen over the patch sequence for `"flash_attention_2"`, otherwise `scaled_dot_product_attention` on
    the targets padded to `num_patches`, padding masked out.

    Args:
        attention ([`IntegralSelfAttention`])
        x (`torch.FloatTensor` of shape `(1, total_patches, hidden_size)`)
        rope (`tuple[torch.FloatTensor, torch.FloatTensor]`):
            `(cos, sin)`, each `(1, total_patches, head_dim)`.
        layout ([`IntegralVisualAttentionLayout`])
    """
    P, C = x.shape[1], x.shape[2]
    q, k, v = attention.qkv(x).reshape(1, P, 3, attention.num_heads, attention.head_dim).unbind(2)
    q, k = apply_rotary_pos_emb(q, *rope, unsqueeze_dim=2), apply_rotary_pos_emb(k, *rope, unsqueeze_dim=2)
    if layout.attn_implementation == "flash_attention_2":
        from flash_attn import flash_attn_varlen_func

        dtype = _attention_dtype(q)
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"flash_attention_2 runs in float16 or bfloat16, not {dtype}; use autocast or another "
                             f"attention implementation")
        meta = layout.varlen
        out = flash_attn_varlen_func(q[0].to(dtype), k[0].to(dtype), v[0].to(dtype), meta["cu_seqlens_q"],
                                     meta["cu_seqlens_q"], meta["max_seqlen_q"], meta["max_seqlen_q"], causal=False,
                                     deterministic=torch.are_deterministic_algorithms_enabled())[None]
    else:
        q, k, v = _padded_heads(layout, q, k, v)
        mask = None if layout.latent_mask is None else layout.latent_mask[:, None, None, :]
        out = layout.unpad(F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2))
    return attention.proj(out.reshape(1, P, C))


class IntegralFlowHeadFinalLayer(nn.Module):
    """LayerNorm with adaLN-Zero shift/scale from the condition, then a linear map to the latent size."""

    def __init__(self, config: IntegralVisualGenerationConfig):
        super().__init__()
        dim = config.flow_head_hidden_size
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=config.layer_norm_eps)
        self.linear = nn.Linear(dim, config.latent_size)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear._zero_init = True
        self.ada[1]._zero_init = True

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.ada(condition).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1 + scale) + shift)


class IntegralFlowHead(nn.Module):
    """Predicts the velocity of every patch from the noisy latents, the timestep and the visual generation blocks'
    output.

        noisy latents -> input_proj -> + 2D sincos position embedding
        condition: blocks' output -> condition_norm -> condition_proj
        per-patch modulation: silu(timestep_mlp(timestep) + condition)
          -> blocks (2D rotary self-attention, adaLN-Zero) -> final_layer -> velocity
    """

    def __init__(self, config: IntegralVisualGenerationConfig):
        super().__init__()
        dim = config.flow_head_hidden_size
        self.hidden_size = dim
        self.condition_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.condition_proj = nn.Linear(config.hidden_size, dim)
        self.input_proj = nn.Linear(config.latent_size, dim)
        self.timestep_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.rope = IntegralRoPE2D(dim // config.flow_head_num_attention_heads, config.flow_head_rope_theta)
        self.blocks = nn.ModuleList([IntegralFlowHeadBlock(config) for _ in range(config.flow_head_num_hidden_layers)])
        self.final_layer = IntegralFlowHeadFinalLayer(config)

    def forward(self, noisy_latents: Tensor, timesteps: Tensor, hidden_states: Tensor,
                layout: IntegralVisualAttentionLayout) -> Tensor:
        """
        Args:
            noisy_latents (`torch.FloatTensor` of shape `(1, total_patches, latent_size)`):
                Every target's patches, target by target (`layout.segments`).
            timesteps (`torch.FloatTensor` of shape `(num_targets,)`)
            hidden_states (`torch.FloatTensor` of shape `(1, total_patches, hidden_size)`):
                Output of the visual generation blocks.
            layout ([`IntegralVisualAttentionLayout`]):
                The targets' grids and patches, and the attention implementation.

        Returns:
            `torch.FloatTensor` of shape `(1, total_patches, latent_size)`.
        """
        condition = self.condition_proj(self.condition_norm(hidden_states.to(self.condition_proj.weight.dtype)))
        dt = self.input_proj.weight.dtype
        h = self.input_proj(noisy_latents.to(dt))
        num_targets = len(layout.grids)
        rope = tuple(layout.unpad(t.expand(num_targets, -1, -1)).to(h.dtype)
                     for t in self.rope(layout.grids, layout.num_patches))
        positions = {grid: sincos_position_embedding_2d(self.hidden_size, *grid).to(device=h.device, dtype=h.dtype)
                     for grid in set(layout.grids)}
        h = layout.per_target(lambda x, _, grid: x + positions[grid], h)
        t_emb = self.timestep_mlp(timestep_embedding(timesteps, self.hidden_size).to(dt)).unsqueeze(1)
        c = layout.per_target(lambda x, i, _: F.silu(t_emb[i] + x), condition.to(dt))
        for block in self.blocks:
            h = block(h, c, rope, layout)
        return self.final_layer(h, c)


class IntegralVisualGenerationPreTrainedModel(PreTrainedModel):
    config_class = IntegralVisualGenerationConfig
    base_model_prefix = "visual_generation_model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IntegralVisualGenerationBlock", "IntegralFlowHeadBlock"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                module.weight.zero_()
            else:
                module.weight.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, IntegralVisualGenerationBlock):
            module.scale_shift_table.normal_().div_(module.scale_shift_table.shape[-1] ** 0.5)


class IntegralVisualGenerationModel(IntegralVisualGenerationPreTrainedModel):
    """Noisy latents, timesteps and the language model's key/value states -> velocity.

        noisy latents -> input_proj -> blocks (one per layer in `key_value_layer_indices`) -> norm -> flow_head
        timestep -> sinusoidal embedding of `timestep * timestep_scale` -> timestep_mlp -> timestep_modulation
    """

    def __init__(self, config: IntegralVisualGenerationConfig):
        super().__init__(config)
        hidden_size = config.hidden_size
        self.input_proj = nn.Sequential(nn.Linear(config.latent_size, hidden_size), nn.GELU(),
                                        nn.Linear(hidden_size, hidden_size))
        self.timestep_mlp = nn.Sequential(nn.Linear(config.timestep_embedding_size, hidden_size), nn.SiLU(),
                                          nn.Linear(hidden_size, hidden_size))
        self.timestep_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.blocks = nn.ModuleList([IntegralVisualGenerationBlock(config)
                                     for _ in config.key_value_layer_indices])
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=config.layer_norm_eps)
        self.flow_head = IntegralFlowHead(config)
        self.post_init()

    def forward(self, noisy_latents: Tensor, timesteps: Tensor, grid_thw: Tensor, key_values: IntegralVisualKeyValues,
                batch_index: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            noisy_latents (`torch.FloatTensor` of shape `(num_targets, max_num_patches, latent_size)`):
                Noisy standardized latents, padded to the longest target.
            timesteps (`torch.FloatTensor` of shape `(num_targets,)`):
                Timesteps in [0, 1].
            grid_thw (`torch.LongTensor` of shape `(num_targets, 3)`):
                Patch grid `(1, height, width)` of each target.
            key_values ([`IntegralVisualKeyValues`]):
                The language model key/value states, `batch_size` rows.
            batch_index (`torch.LongTensor` of shape `(num_targets,)`, *optional*):
                The row of `key_values` each target reads. Target `i` reads row `i` if not given.
            attention_mask (`torch.Tensor` of shape `(num_targets, sequence_length)`, *optional*):
                The language model positions each target may attend to. The row's `key_values.attention_mask` if
                not given.

        Returns:
            `torch.FloatTensor` of shape `(num_targets, max_num_patches, latent_size)`: the predicted velocity, zero past
            each target's patches.

        Every module computes on the targets' real patches only, laid end to end (`IntegralVisualAttentionLayout`);
        the sdpa and flex attentions pad them back to `max_num_patches` for the attention call.
        """
        config = self.config
        grids = [(int(h), int(w)) for _, h, w in grid_thw.tolist()]
        B, N, _ = noisy_latents.shape
        if attention_mask is None:
            attention_mask = key_values.attention_mask
            if batch_index is not None:
                attention_mask = (attention_mask.expand(B, -1) if attention_mask.shape[0] == 1
                                  else attention_mask.index_select(0, batch_index))
        layout = IntegralVisualAttentionLayout(attention_mask, grids, batch_index, N,
                                               config._attn_implementation or "sdpa")

        latents = layout.unpad(noisy_latents)
        dt = self.input_proj[0].weight.dtype
        x = self.input_proj(latents.to(dt))
        t_emb = self.timestep_mlp(
            timestep_embedding(timesteps * config.timestep_scale, config.timestep_embedding_size).to(dt))
        modulation = self.timestep_modulation(t_emb).reshape(B, 6, -1)
        # frequencies computed on the input's device on every call; the flow head and the pixel decoder keep theirs
        # in a persistent buffer, and the two differ in the last bits between CPU and GPU
        frequencies = rope_2d_frequencies(int(config.head_dim * config.partial_rotary_factor), config.rope_theta,
                                          x.device)
        rope = tuple(layout.unpad(t.expand(B, -1, -1)) for t in rope_2d(frequencies, grids, N))
        for block, key_states, value_states in zip(self.blocks, key_values.key_states, key_values.value_states):
            x = block(x, key_states, value_states, modulation, layout, rope)
        return layout.pad(self.flow_head(latents, timesteps, self.norm(x).float(), layout))


@auto_docstring
@dataclass
class IntegralModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
        The attribute is deprecated and will be removed in v5.20, use `model.base_model.rope_deltas` instead.
    visual_velocity (`torch.FloatTensor` of shape `(num_targets, max_num_patches, latent_size)`, *optional*):
        Velocity predicted for `visual_noisy_latents`.
    visual_key_values ([`IntegralVisualKeyValues`], *optional*):
        The language model key/value states the visual generation blocks read.
    """

    rope_deltas: torch.LongTensor | None = None
    visual_velocity: torch.FloatTensor | None = None
    visual_key_values: IntegralVisualKeyValues | None = None


@auto_docstring
@dataclass
class IntegralTextModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    visual_key_value_states (`tuple[tuple[torch.FloatTensor, torch.FloatTensor]]`, *optional*):
        One `(visual_key_states, visual_value_states)` per layer in `visual_key_value_layer_indices`, in that order,
        each of shape `(batch_size, num_key_value_heads, sequence_length, head_dim)`: that layer's `k_proj` and
        `v_proj` outputs plus `visual_key_value_deltas`, the keys after `k_norm` and the rotary embedding. Computed
        separately from the key/value states the layer attends with and stores in `past_key_values`, and for this
        call's tokens only.
    """

    visual_key_value_states: tuple[tuple[torch.FloatTensor, torch.FloatTensor], ...] | None = None


class IntegralTextModel(IntegralPreTrainedModel):
    config: IntegralTextConfig
    # Qwen3.5 stores the Gated DeltaNet layers' `A_log` and gated norm weight in fp32; a bf16 / fp16 load keeps them
    # fp32 instead of rounding them (`_strict`: the plain `_keep_in_fp32_modules` acts on fp16 loads only).
    _keep_in_fp32_modules_strict = ["linear_attn.A_log", "linear_attn.norm.weight"]

    def __init__(self, config: IntegralTextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [IntegralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = IntegralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = IntegralTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        visual_key_value_layer_indices: list[int] | None = None,
        visual_key_value_deltas: tuple | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> IntegralTextModelOutputWithPast:
        r"""
        visual_key_value_layer_indices (`list[int]`, *optional*):
            Full-attention layers, in increasing order, whose visual key/value states are returned in
            `visual_key_value_states`.
        visual_key_value_deltas (`tuple[tuple[torch.Tensor, torch.Tensor]]`, *optional*):
            One `(key_delta, value_delta)` per layer in `visual_key_value_layer_indices`, each of shape
            `(batch_size, sequence_length, num_key_value_heads * head_dim)`, added to that layer's `k_proj` and `v_proj`
            outputs for the visual key/value states only.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        visual_key_value_layer_indices = list(visual_key_value_layer_indices or [])
        if visual_key_value_layer_indices != sorted(set(visual_key_value_layer_indices)):
            raise ValueError(f"visual_key_value_layer_indices must be increasing, got {visual_key_value_layer_indices}")
        if visual_key_value_deltas is not None and len(visual_key_value_deltas) != len(visual_key_value_layer_indices):
            raise ValueError(f"{len(visual_key_value_deltas)} visual_key_value_deltas for "
                             f"{len(visual_key_value_layer_indices)} visual_key_value_layer_indices")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # the hard coded `4` is for text, temporal, height and width.
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        visual_key_value_states = []
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            visual = i in visual_key_value_layer_indices
            visual_key_value_delta = None
            if visual and visual_key_value_deltas is not None:
                visual_key_value_delta = visual_key_value_deltas[visual_key_value_layer_indices.index(i)]
            outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_visual_key_value_states=visual,
                visual_key_value_delta=visual_key_value_delta,
                **kwargs,
            )
            if visual:
                hidden_states, visual_key_states, visual_value_states = outputs
                visual_key_value_states.append((visual_key_states, visual_value_states))
            else:
                hidden_states = outputs

        hidden_states = self.norm(hidden_states)

        return IntegralTextModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            visual_key_value_states=tuple(visual_key_value_states) if visual_key_value_layer_indices else None,
        )


@auto_docstring
class IntegralModel(IntegralPreTrainedModel):
    base_model_prefix = "model"
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    _no_split_modules = ["IntegralDecoderLayer", "IntegralVisionBlock", "IntegralTransformerBlock"]
    # The vision tokenizer and the visual generation model run in fp32 (the language model's dtype is the load's):
    # `from_pretrained(dtype=torch.bfloat16)` gives the language model and its head in bf16 and these in fp32.
    _keep_in_fp32_modules_strict = ["vision_tokenizer", "visual_generation_model"]

    def __init__(self, config):
        super().__init__(config)
        self.vision_tokenizer = IntegralVisionTokenizer._from_config(config.vision_tokenizer_config)
        self.language_model = IntegralTextModel._from_config(config.text_config)
        self.visual_generation_model = IntegralVisualGenerationModel._from_config(config.visual_generation_config)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        spatial_merge_size: int = 1,
        device: str | torch.device | None = None,
    ):
        """
        Compute 3D positional indices for the language model tokens of one image.

        The positions are generated from the image's patch grid, with the height and width divided by the spatial
        merge size, and offset by `start_position`.

        Args:
            start_position (`int`):
                Offset added to all computed positional indices.
            grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
                The patch grid `(1, height, width)` of the image.
            spatial_merge_size (`int`, *optional*, defaults to 1):
                Patches merged per side into one language model token.
            device (`str` or `torch.device`, *optional*):
                Device on which the resulting tensor is allocated. If `None`, uses the current default device.

        Returns:
            torch.LongTensor of shape (3, sequence_length):
                Positional indices for temporal, height, and width dimensions,
                flattened into sequence form and offset by `start_position`.
        """
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item(),
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        position_temporal = torch.arange(llm_grid_t, device=device)
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position

        T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
        vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
        vision_position_ids[0] += start_position
        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the 3D rotary position ids of text and image tokens.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
                Token type ids: text (0), image (1).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                Patch grid `(1, height, width)` of each image.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_tokenizer_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_grids = iter(image_grid_thw) if image_grid_thw is not None else None

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                # image == 1
                else:
                    grid_thw = next(image_grids)
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    @accepts_precomputed_kwargs(modality="video")
    @can_return_tuple
    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Video input is not supported by this model. Always raises `ValueError`.

        pixel_values_videos (`torch.FloatTensor`):
            Not supported.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Not supported.
        """
        raise ValueError("Integral does not support video input.")

    @accepts_precomputed_kwargs(modality="image")
    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_latents: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(total_patches, 3 * temporal_patch_size * patch_size**2)`, *optional*):
            Patches of all images, as produced by the Qwen2-VL image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Patch grid `(1, height, width)` of each image.
        image_latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`, *optional*):
            Standardized latents of the images, given instead of `pixel_values`.

        Returns:
            [`BaseModelOutputWithPooling`] with `last_hidden_state` the standardized latents
            `(total_patches, latent_size)` and `pooler_output` a tuple of language model input embeddings, one
            `(num_merged_patches, hidden_size)` tensor per image.
        """
        if (pixel_values is None) == (image_latents is None):
            raise ValueError("You must specify exactly one of pixel_values or image_latents")
        if image_latents is None:
            image_latents, image_embeds = self.vision_tokenizer.encode_and_merge(pixel_values, image_grid_thw)
        else:
            image_embeds = self.vision_tokenizer.merge(image_latents, image_grid_thw)
        merge_size = self.config.vision_tokenizer_config.spatial_merge_size
        split_sizes = (image_grid_thw.prod(-1) // merge_size**2).tolist()
        return BaseModelOutputWithPooling(last_hidden_state=image_latents,
                                          pooler_output=torch.split(image_embeds, split_sizes))

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains the image placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token
        count is equal to the length of the image features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )
        return special_image_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        has_multimodal = image_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Images were passed (via `image_grid_thw`) but `mm_token_type_ids` is missing. Please pass "
                "`mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be computed correctly. "
                "`mm_token_type_ids` is returned by the processor alongside `input_ids`."
            )
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        # Use pre-calculated rope-deltas to infer correct 3D position ids during incremental
        # generation (past_key_values_length > 0) or when only inputs_embeds is provided (no input_ids
        # to recompute from). Skip when input_ids is provided without past_key_values to avoid shape
        # mismatches from stale rope_deltas (e.g., training forward pass after generation).
        elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
            delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            # Can't build correct 3D positions. Let the model infer it
            position_ids = None
        return position_ids

    @auto_docstring
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        image_latents: torch.FloatTensor | None = None,
        visual_noisy_latents: torch.FloatTensor | None = None,
        visual_timesteps: torch.FloatTensor | None = None,
        visual_grid_thw: torch.LongTensor | None = None,
        visual_batch_index: torch.LongTensor | None = None,
        visual_attention_mask: torch.Tensor | None = None,
        visual_key_values: IntegralVisualKeyValues | None = None,
        return_visual_key_values: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | IntegralModelOutputWithPast:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Not supported.
        image_latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`, *optional*):
            Standardized latents of the images in the prompt, given instead of `pixel_values`.
        visual_noisy_latents (`torch.FloatTensor` of shape `(num_targets, max_num_patches, latent_size)`, *optional*):
            Noisy standardized latents of the images to generate. When given, `visual_velocity` is predicted.
        visual_timesteps (`torch.FloatTensor` of shape `(num_targets,)`, *optional*):
            Timesteps in [0, 1] of `visual_noisy_latents`.
        visual_grid_thw (`torch.LongTensor` of shape `(num_targets, 3)`, *optional*):
            Patch grid `(1, height, width)` of each target.
        visual_batch_index (`torch.LongTensor` of shape `(num_targets,)`, *optional*):
            The batch row each target reads. Target `i` reads row `i` if not given.
        visual_attention_mask (`torch.Tensor` of shape `(num_targets, sequence_length)`, *optional*):
            The language model positions each target may attend to. The row's attention mask if not given.
        visual_key_values ([`IntegralVisualKeyValues`], *optional*):
            Key/value states from an earlier call. When given, the language model does not run.
        return_visual_key_values (`bool`, *optional*, defaults to `False`):
            Whether to return the key/value states the visual generation blocks read.
        """
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise ValueError("Integral does not support video input.")
        if visual_key_values is not None:
            if input_ids is not None or inputs_embeds is not None or pixel_values is not None or image_latents is not None:
                raise ValueError("visual_key_values replaces the language model inputs; give one or the other")
            if visual_noisy_latents is None:
                raise ValueError("visual_key_values is only read together with visual_noisy_latents")
            velocity = self.visual_generation_model(visual_noisy_latents, visual_timesteps, visual_grid_thw,
                                                    visual_key_values, visual_batch_index, visual_attention_mask)
            return IntegralModelOutputWithPast(visual_velocity=velocity,
                                               visual_key_values=visual_key_values if return_visual_key_values else None)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        build_visual_key_values = visual_noisy_latents is not None or return_visual_key_values

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        if pixel_values is not None or image_latents is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values, image_grid_thw, image_latents=image_latents, return_dict=True, **kwargs
            )
            image_latents = image_outputs.last_hidden_state
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            image_mask = image_mask[..., 0]

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        visual_key_value_deltas = None
        if build_visual_key_values and image_mask is not None:
            # the visual generation adapter's key/value deltas at the image tokens, zero elsewhere
            text_config = self.config.text_config
            width = text_config.num_key_value_heads * text_config.head_dim
            base = inputs_embeds.new_zeros(*image_mask.shape, width)
            mask = image_mask[..., None]
            visual_key_value_deltas = tuple(
                (base.masked_scatter(mask, key_delta.reshape(-1, width).to(base.dtype)),
                 base.masked_scatter(mask, value_delta.reshape(-1, width).to(base.dtype)))
                for key_delta, value_delta in self.vision_tokenizer.adapt(image_latents, image_grid_thw))
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_key_value_layer_indices=(self.config.visual_generation_config.key_value_layer_indices
                                            if build_visual_key_values else None),
            visual_key_value_deltas=visual_key_value_deltas,
            **kwargs,
        )

        velocity = None
        if build_visual_key_values:
            if attention_mask is None:
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
            visual_key_values = IntegralVisualKeyValues(
                key_states=tuple(k for k, _ in outputs.visual_key_value_states),
                value_states=tuple(v for _, v in outputs.visual_key_value_states),
                attention_mask=attention_mask)
            if visual_noisy_latents is not None:
                velocity = self.visual_generation_model(visual_noisy_latents, visual_timesteps, visual_grid_thw,
                                                        visual_key_values, visual_batch_index, visual_attention_mask)

        return IntegralModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            visual_velocity=velocity,
            visual_key_values=visual_key_values if return_visual_key_values else None,
        )


@auto_docstring
@dataclass
class IntegralCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
        The attribute is deprecated and will be removed in v5.20, use `model.base_model.rope_deltas` instead.
    lm_loss (`torch.FloatTensor`, *optional*):
        Cross-entropy of `labels`.
    visual_loss (`torch.FloatTensor`, *optional*):
        Mean over targets of the mean squared error between `visual_velocity` and `visual_velocity_targets` over each
        target's real patches.
    visual_velocity (`torch.FloatTensor` of shape `(num_targets, max_num_patches, latent_size)`, *optional*):
        Velocity predicted for `visual_noisy_latents`.
    visual_key_values ([`IntegralVisualKeyValues`], *optional*):
        The language model key/value states the visual generation blocks read.
    """

    rope_deltas: torch.LongTensor | None = None
    lm_loss: torch.FloatTensor | None = None
    visual_loss: torch.FloatTensor | None = None
    visual_velocity: torch.FloatTensor | None = None
    visual_key_values: IntegralVisualKeyValues | None = None


@dataclass
class IntegralImageGenerationCache:
    r"""
    The prompt side of image generation, computed once by `prepare_image_generation_cache` and reusable across
    `generate_image` calls (other seeds, step counts, guidance scales or images per prompt).

    visual_key_values ([`IntegralVisualKeyValues`]):
        The key/value states of the prompt rows: the `num_prompts` conditional rows, then, when
        `has_unconditional`, their `num_prompts` unconditional rows in the same order.
    visual_grid_thw (`torch.LongTensor` of shape `(num_prompts, 3)`):
        Patch grid `(1, height, width)` of the image each prompt asks for.
    has_unconditional (`bool`):
        Whether the unconditional rows classifier-free guidance needs are present.
    image_free_key_values ([`IntegralVisualKeyValues`], *optional*):
        The key/value states of the image-free rows image guidance needs (`num_prompts` rows, the instructions with
        no input image), from their own language model call.
    conditional_key_values ([`IntegralVisualKeyValues`], *optional*):
        With image guidance: the conditional rows' key/value states from a language model call of those rows alone,
        which the steps outside `guidance_interval` read (the reference runs the conditional branch alone there).
    """

    visual_key_values: IntegralVisualKeyValues
    visual_grid_thw: Tensor
    has_unconditional: bool
    image_free_key_values: IntegralVisualKeyValues | None = None
    conditional_key_values: IntegralVisualKeyValues | None = None

    @property
    def num_prompts(self) -> int:
        return self.visual_grid_thw.shape[0]


@dataclass
class IntegralImageGenerationOutput(ModelOutput):
    r"""
    images (`list[PIL.Image.Image]` or `torch.FloatTensor` of shape `(num_images, 3, height, width)`):
        The generated images, `num_images_per_prompt` per prompt in prompt order. Tensors are in [0, 1].
    latents (`torch.FloatTensor` of shape `(num_images, num_patches, latent_size)`):
        The generated standardized latents.
    cache ([`IntegralImageGenerationCache`]):
        The prompt cache the images were generated from, for further `generate_image` calls.
    """

    images: Any = None
    latents: torch.FloatTensor | None = None
    cache: IntegralImageGenerationCache | None = None


class IntegralForConditionalGeneration(IntegralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.model = IntegralModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.post_init()

    @auto_docstring
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Video input is not supported by this model. Always raises `ValueError`.

        pixel_values_videos (`torch.FloatTensor`):
            Not supported.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Not supported.
        """
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        image_latents: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(total_patches, 3 * temporal_patch_size * patch_size**2)`, *optional*):
            Patches of all images, as produced by the Qwen2-VL image processor.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            Patch grid `(1, height, width)` of each image.
        image_latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`, *optional*):
            Standardized latents of the images, given instead of `pixel_values`.
        """
        return self.model.get_image_features(pixel_values, image_grid_thw, image_latents=image_latents, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_latents: torch.FloatTensor | None = None,
        visual_noisy_latents: torch.FloatTensor | None = None,
        visual_timesteps: torch.FloatTensor | None = None,
        visual_grid_thw: torch.LongTensor | None = None,
        visual_batch_index: torch.LongTensor | None = None,
        visual_attention_mask: torch.Tensor | None = None,
        visual_velocity_targets: torch.FloatTensor | None = None,
        visual_key_values: IntegralVisualKeyValues | None = None,
        return_visual_key_values: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | IntegralCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            Not supported.
        image_latents (`torch.FloatTensor` of shape `(total_patches, latent_size)`, *optional*):
            Standardized latents of the images in the prompt, given instead of `pixel_values`.
        visual_noisy_latents (`torch.FloatTensor` of shape `(num_targets, max_num_patches, latent_size)`, *optional*):
            Noisy standardized latents of the images to generate.
        visual_timesteps (`torch.FloatTensor` of shape `(num_targets,)`, *optional*):
            Timesteps in [0, 1] of `visual_noisy_latents`.
        visual_grid_thw (`torch.LongTensor` of shape `(num_targets, 3)`, *optional*):
            Patch grid `(1, height, width)` of each target.
        visual_batch_index (`torch.LongTensor` of shape `(num_targets,)`, *optional*):
            The batch row each target reads. Target `i` reads row `i` if not given.
        visual_attention_mask (`torch.Tensor` of shape `(num_targets, sequence_length)`, *optional*):
            The language model positions each target may attend to.
        visual_velocity_targets (`torch.FloatTensor` of shape `(num_targets, max_num_patches, latent_size)`, *optional*):
            Velocity targets `noise - latents`. When given, `visual_loss` is computed.
        visual_key_values ([`IntegralVisualKeyValues`], *optional*):
            Key/value states from an earlier call. When given, the language model does not run.
        return_visual_key_values (`bool`, *optional*, defaults to `False`):
            Whether to return the key/value states the visual generation blocks read.
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            image_latents=image_latents,
            visual_noisy_latents=visual_noisy_latents,
            visual_timesteps=visual_timesteps,
            visual_grid_thw=visual_grid_thw,
            visual_batch_index=visual_batch_index,
            visual_attention_mask=visual_attention_mask,
            visual_key_values=visual_key_values,
            return_visual_key_values=return_visual_key_values,
            **kwargs,
        )

        logits = lm_loss = None
        if outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            if labels is not None:
                lm_loss = self.loss_function(logits=logits, labels=labels,
                                             vocab_size=self.config.text_config.vocab_size)

        visual_loss = None
        if visual_velocity_targets is not None:
            velocity = outputs.visual_velocity
            lengths = (visual_grid_thw.prod(-1)).to(velocity.device)
            keep = torch.arange(velocity.shape[1], device=velocity.device)[None, :] < lengths[:, None]
            squared = ((velocity - visual_velocity_targets) * keep[..., None]).pow(2).sum(dim=(1, 2))
            visual_loss = (squared / (lengths * velocity.shape[-1]).to(squared.dtype)).mean()

        losses = [l for l in (lm_loss, visual_loss) if l is not None]
        return IntegralCausalLMOutputWithPast(
            loss=sum(losses) if losses else None,
            lm_loss=lm_loss,
            visual_loss=visual_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            visual_velocity=outputs.visual_velocity,
            visual_key_values=outputs.visual_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        image_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        # Overwritten -- requires 3D position ids

        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        # Early exit in case we are continuing generation from past kv
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.model.rope_deltas
            return position_ids

        # Otherwise compute 3d position ids for vision tokens and concat with text position ids
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if (
            is_input_ids
            and model_kwargs.get("mm_token_type_ids") is not None
            and model_kwargs.get("image_grid_thw") is not None
        ):
            model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        # Concatenate "text + vision" positions into [4, bs, seq-len]
        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)

        return position_ids

    def _get_image_nums(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get the number of images of each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size,)`)
        """
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        return torch.sum(vision_first_mask & image_mask, dim=1)

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: torch.LongTensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- pixel_values and image_grid_thw have no batch dimension:
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            image_nums = self._get_image_nums(input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None))

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if key == "position_ids" and dict_to_expand[key].ndim == 3:
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=1)
                elif (
                    dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @torch.no_grad()
    def prepare_image_generation_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        mm_token_type_ids: torch.IntTensor,
        visual_grid_thw: torch.LongTensor,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        autocast_dtype: torch.dtype | None = torch.bfloat16,
    ) -> IntegralImageGenerationCache:
        """
        Runs the language model once over image generation prompts and keeps the key/value states the visual
        generation blocks read. The inputs are moved to the model's device.

        Args:
            input_ids (`torch.LongTensor` of shape `(num_rows, sequence_length)`):
                The prompts, right-padded: `num_prompts` conditional rows, optionally followed by one unconditional
                row each in the same order, optionally followed by one image-free row each in the same order
                (`IntegralProcessor.prepare_image_generation`).
            attention_mask (`torch.Tensor` of shape `(num_rows, sequence_length)`):
                `1` for the prompt tokens, `0` for the padding.
            mm_token_type_ids (`torch.IntTensor` of shape `(num_rows, sequence_length)`):
                Token type ids: text (0), image (1).
            visual_grid_thw (`torch.LongTensor` of shape `(num_prompts, 3)`):
                Patch grid `(1, height, width)` of the image each prompt asks for.
            pixel_values (`torch.FloatTensor`, *optional*):
                The input images the rows read, in row order (`IntegralProcessor.prepare_image_generation`).
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                Patch grid of each input image.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The autocast dtype the model runs under; `None` runs without autocast.

        Returns:
            [`IntegralImageGenerationCache`]. The image-free rows are run in their own language model call, each group
            cut to its own longest row.
        """
        num_prompts, num_rows = visual_grid_thw.shape[0], input_ids.shape[0]
        if num_rows not in (num_prompts, 2 * num_prompts, 3 * num_prompts):
            raise ValueError(
                f"{num_rows} prompt rows for {num_prompts} images: expected the conditional rows, optionally "
                f"followed by as many unconditional rows, optionally followed by as many image-free rows")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw are given together")
        device = self.device
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        mm_token_type_ids = mm_token_type_ids.to(device)
        if pixel_values is not None:
            pixel_values, image_grid_thw = pixel_values.to(device), image_grid_thw.to(device)

        def prefill(rows, pixel_values, image_grid_thw):
            ids, mask, types = input_ids[rows], attention_mask[rows], mm_token_type_ids[rows]
            if num_rows == 3 * num_prompts:
                width = int(mask.sum(dim=-1).max())
                ids, mask, types = ids[:, :width], mask[:, :width], types[:, :width]
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                position_ids, _ = self.model.get_rope_index(ids, types, image_grid_thw=image_grid_thw,
                                                            attention_mask=mask)
                return self(
                    input_ids=ids,
                    attention_mask=mask,
                    position_ids=position_ids,
                    mm_token_type_ids=types,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    logits_to_keep=torch.zeros(0, dtype=torch.long, device=device),
                    return_visual_key_values=True,
                    use_cache=False,
                ).visual_key_values

        image_free_key_values = conditional_key_values = None
        if num_rows == 3 * num_prompts:
            if bool(mm_token_type_ids[2 * num_prompts:].any()):
                raise ValueError("the image-free rows hold image tokens")
            visual_key_values = prefill(slice(0, 2 * num_prompts), pixel_values, image_grid_thw)
            image_free_key_values = prefill(slice(2 * num_prompts, None), None, None)
            # the conditional rows alone, with their own images (the first half: the unconditional rows hold copies)
            images = image_grid_thw.shape[0] // 2
            patches = int(image_grid_thw[:images].prod(dim=-1).sum())
            conditional_key_values = prefill(slice(0, num_prompts), pixel_values[:patches], image_grid_thw[:images])
        else:
            visual_key_values = prefill(slice(None), pixel_values, image_grid_thw)
        return IntegralImageGenerationCache(
            visual_key_values=visual_key_values,
            visual_grid_thw=visual_grid_thw.to(self.model.visual_generation_model.device),
            has_unconditional=num_rows >= 2 * num_prompts,
            image_free_key_values=image_free_key_values,
            conditional_key_values=conditional_key_values)

    @torch.no_grad()
    def generate_image(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        visual_grid_thw: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        cache: IntegralImageGenerationCache | None = None,
        num_images_per_prompt: int = 1,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 50,
        solver: str = "dpm_solver++",
        time_shift: float | None = None,
        latents: torch.FloatTensor | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        autocast_dtype: torch.dtype | None = torch.bfloat16,
        image_guidance_scale: float | None = 1.5,
        guidance_interval: tuple[float, float] = (0.0, 1.0),
        guidance_renorm: str = "text_channel",
        guidance_renorm_min: float = 1.0,
    ) -> IntegralImageGenerationOutput:
        """
        Generates images by integrating the flow ODE from noise. The language model runs once per prompt
        (`prepare_image_generation_cache`); every step runs the visual generation model on its key/value states.

        With `image_guidance_scale` the guidance has three branches, as BAGEL's editing guidance: the conditional
        velocity `v_cond`, the unconditional (text-free) `v_text` and the image-free `v_img` (the cache's
        image-free rows, `IntegralProcessor.prepare_image_generation(image_guidance=True)`), combined in fp32 as

            v_text_ = v_text + guidance_scale * (v_cond - v_text)
            v       = v_img + image_guidance_scale * (v_text_ - v_img)      (when image_guidance_scale > 1, else v_text_)

        with the renorm `guidance_renorm`: `"text_channel"` scales `v_text_` per token by
        `clamp(|v_cond| / (|v_text_| + 1e-8), guidance_renorm_min, 1)` before the image guidance; `"global"` scales
        `v` by the same ratio of the norms over the whole batch after it; `"none"` does not renorm. Guidance applies
        for `guidance_interval[0] < t <= guidance_interval[1]`; elsewhere the step takes `v_cond`. The conditional
        and text-free rows run in one visual generation model call and the image-free rows in another.

        Args:
            input_ids (`torch.LongTensor` of shape `(num_rows, sequence_length)`, *optional*):
                The prompts, as in `prepare_image_generation_cache`. Given instead of `cache`.
            attention_mask (`torch.Tensor` of shape `(num_rows, sequence_length)`, *optional*):
                As in `prepare_image_generation_cache`.
            mm_token_type_ids (`torch.IntTensor` of shape `(num_rows, sequence_length)`, *optional*):
                As in `prepare_image_generation_cache`.
            visual_grid_thw (`torch.LongTensor` of shape `(num_prompts, 3)`, *optional*):
                Patch grid `(1, height, width)` of the image each prompt asks for; the same for every prompt.
            pixel_values (`torch.FloatTensor`, *optional*), image_grid_thw (`torch.LongTensor`, *optional*):
                The input images the prompts read (editing), as in `prepare_image_generation_cache`.
            cache ([`IntegralImageGenerationCache`], *optional*):
                A prompt cache from `prepare_image_generation_cache` or an earlier `generate_image` call. Given
                instead of the prompts.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Images per prompt; they read the same prompt rows of the cache.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Classifier-free guidance scale against the unconditional rows. 1.0 runs the conditional rows only.
            num_inference_steps (`int`, *optional*, defaults to 50):
                Solver steps, one model call each.
            solver (`str`, *optional*, defaults to `"dpm_solver++"`):
                `"dpm_solver++"` (second-order multistep DPM-Solver++: `sampling.dpm_solver_timesteps`,
                `sampling.dpm_solver_sample`; at least 2 steps; the final protocol's solver) or `"euler"`
                (`sampling.flow_timesteps`, `sampling.euler_step`).
            time_shift (`float`, *optional*):
                Shift of the timestep schedule. Defaults to `config.visual_generation_config.time_shift`.
            latents (`torch.FloatTensor` of shape `(num_images, num_patches, latent_size)`, *optional*):
                The initial noise, used in fp32. Drawn from `generator` when not given.
            generator (`torch.Generator`, *optional*):
                Generator for the initial noise; the noise is drawn on its device, then moved to the model's.
            output_type (`str`, *optional*, defaults to `"pil"`):
                `"pil"` (pixels `int(255 * x)` of the [0, 1] images), `"pt"` (images in [0, 1]) or `"latent"` (no
                decoding).
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The autocast dtype the model runs under; `None` runs without autocast.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                The image guidance scale, applied when the cache has image-free rows (editing:
                `IntegralProcessor.prepare_image_generation` adds them by default with input images); text-to-image
                has none and is unaffected. `None` is text-only classifier-free guidance.
            guidance_interval (`tuple[float, float]`, *optional*, defaults to `(0.0, 1.0)`):
                With image guidance: the timesteps `(low, high]` guidance applies at.
            guidance_renorm (`str`, *optional*, defaults to `"text_channel"`):
                With image guidance: `"text_channel"`, `"global"` or `"none"`.
            guidance_renorm_min (`float`, *optional*, defaults to 1.0):
                With image guidance: the lower bound of the renorm ratio; 1.0 leaves the velocity as it is.

        Returns:
            [`IntegralImageGenerationOutput`].
        """
        if (cache is None) == (input_ids is None) or (cache is not None and pixel_values is not None):
            raise ValueError("Give either the prompts (`input_ids`, ...) or a `cache`, not both and not neither")
        if cache is None:
            cache = self.prepare_image_generation_cache(input_ids, attention_mask, mm_token_type_ids,
                                                        visual_grid_thw, pixel_values=pixel_values,
                                                        image_grid_thw=image_grid_thw, autocast_dtype=autocast_dtype)
        guidance = guidance_scale != 1.0
        if guidance and not cache.has_unconditional:
            raise ValueError(f"guidance_scale {guidance_scale} needs the unconditional rows, which the cache does not "
                             f"have")
        if output_type not in ("pil", "pt", "latent"):
            raise ValueError(f"output_type must be 'pil', 'pt' or 'latent', got {output_type!r}")
        if solver not in ("euler", "dpm_solver++"):
            raise ValueError(f"solver must be 'euler' or 'dpm_solver++', got {solver!r}")
        if cache.image_free_key_values is None:
            image_guidance_scale = None
        if image_guidance_scale is not None:
            if not cache.has_unconditional:
                raise ValueError("image guidance needs the unconditional rows")
            if guidance_renorm not in ("text_channel", "global", "none"):
                raise ValueError(f"guidance_renorm must be 'text_channel', 'global' or 'none', got {guidance_renorm!r}")

        num_prompts = cache.num_prompts
        device = self.model.visual_generation_model.device
        grid_thw = cache.visual_grid_thw.to(device).repeat_interleave(num_images_per_prompt, dim=0)
        if not bool((grid_thw == grid_thw[:1]).all()):
            raise ValueError(f"all images of a call need the same size, got grids {cache.visual_grid_thw.tolist()}")
        num_images, num_patches = grid_thw.shape[0], int(grid_thw[0].prod())
        latent_size = self.config.visual_generation_config.latent_size
        shape = (num_images, num_patches, latent_size)
        if latents is None:
            noise_device = generator.device if generator is not None else device
            latents = torch.randn(shape, generator=generator, device=noise_device, dtype=torch.float32)
        elif latents.shape != shape:
            raise ValueError(f"latents of shape {tuple(latents.shape)}, expected {shape}")
        latents = latents.to(device=device, dtype=torch.float32)

        # Image i reads its prompt's conditional row and, under guidance, that prompt's unconditional row.
        prompt_rows = torch.arange(num_prompts, device=device).repeat_interleave(num_images_per_prompt)
        image_grid = grid_thw
        rows = prompt_rows
        if guidance or image_guidance_scale is not None:
            rows = torch.cat([rows, rows + num_prompts])
            grid_thw = torch.cat([grid_thw, grid_thw])
        num_rows = cache.visual_key_values.key_states[0].shape[0]
        batch_index = None if torch.equal(rows, torch.arange(num_rows, device=device)) else rows
        free_index = (None if torch.equal(prompt_rows, torch.arange(num_prompts, device=device))
                      else prompt_rows)

        def velocity_with_image_guidance(latents, t):
            low, high = guidance_interval
            now = float(t.reshape(-1)[0])
            if not (low < now <= high):
                return self(
                    visual_key_values=cache.conditional_key_values,
                    visual_noisy_latents=latents,
                    visual_timesteps=t.expand(num_images),
                    visual_grid_thw=image_grid,
                    visual_batch_index=free_index,
                ).visual_velocity.float()
            noisy = torch.cat([latents, latents])
            velocity = self(
                visual_key_values=cache.visual_key_values,
                visual_noisy_latents=noisy,
                visual_timesteps=t.expand(noisy.shape[0]),
                visual_grid_thw=grid_thw,
                visual_batch_index=batch_index,
            ).visual_velocity
            v_cond, v_text = velocity[:num_images].float(), velocity[num_images:].float()
            v_img = self(
                visual_key_values=cache.image_free_key_values,
                visual_noisy_latents=latents,
                visual_timesteps=t.expand(num_images),
                visual_grid_thw=image_grid,
                visual_batch_index=free_index,
            ).visual_velocity.float()
            v_text_ = v_text + guidance_scale * (v_cond - v_text)
            if guidance_renorm == "text_channel":
                scale = (torch.norm(v_cond, dim=-1, keepdim=True)
                         / (torch.norm(v_text_, dim=-1, keepdim=True) + 1e-8)).clamp(min=guidance_renorm_min, max=1.0)
                v_text_ = v_text_ * scale
                return (v_img + image_guidance_scale * (v_text_ - v_img)
                        if image_guidance_scale > 1.0 else v_text_)
            v = v_img + image_guidance_scale * (v_text_ - v_img) if image_guidance_scale > 1.0 else v_text_
            if guidance_renorm == "global":
                v = v * (torch.norm(v_cond) / (torch.norm(v) + 1e-8)).clamp(min=guidance_renorm_min, max=1.0)
            return v

        def velocity_at(latents, t):
            if image_guidance_scale is not None:
                return velocity_with_image_guidance(latents, t)
            noisy = torch.cat([latents, latents]) if guidance else latents
            velocity = self(
                visual_key_values=cache.visual_key_values,
                visual_noisy_latents=noisy,
                visual_timesteps=t.expand(noisy.shape[0]),
                visual_grid_thw=grid_thw,
                visual_batch_index=batch_index,
            ).visual_velocity
            if guidance:
                velocity = classifier_free_guidance(velocity[:num_images], velocity[num_images:], guidance_scale)
            return velocity

        if time_shift is None:
            time_shift = self.config.visual_generation_config.time_shift
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            if solver == "euler":
                timesteps = flow_timesteps(num_inference_steps, time_shift, device=device, dtype=latents.dtype)
                for t, t_next in zip(timesteps[:-1], timesteps[1:]):
                    latents = euler_step(latents, velocity_at(latents, t), t, t_next)
            else:
                timesteps = dpm_solver_timesteps(num_inference_steps, time_shift, device=device)
                latents = dpm_solver_sample(velocity_at, latents, timesteps)

            images = None
            if output_type != "latent":
                images = self.model.vision_tokenizer.decode(latents.reshape(-1, latent_size), grid_thw[:num_images])
                images = torch.stack(images).clamp(0, 1)
        if output_type == "pil":
            from PIL import Image

            pixels = images.mul(255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            images = [Image.fromarray(p) for p in pixels]
        return IntegralImageGenerationOutput(images=images, latents=latents, cache=cache)


__all__ = [
    "IntegralVisionTokenizer",
    "IntegralVisionTokenizerOutput",
    "IntegralVisionTokenizerPreTrainedModel",
    "IntegralVisualGenerationModel",
    "IntegralVisualGenerationPreTrainedModel",
    "IntegralVisualKeyValues",
    "IntegralTextModel",
    "IntegralModel",
    "IntegralForConditionalGeneration",
    "IntegralImageGenerationCache",
    "IntegralImageGenerationOutput",
    "IntegralPreTrainedModel",
]
