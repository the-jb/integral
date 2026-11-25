"""What the model needs from a packed row: the sample boundaries, positions, and each target's condition.

- Boundaries: samples must not see each other. The full-attention layers take `cu_seq_lens_q/k` (flash-attention
  only), the linear-attention layers `cu_seqlens` (the gated delta rule) and `seq_idx` (the causal conv);
  `assert_packing_supported` checks that every one of those kernels is in use.
- Positions: every sample's mrope positions start again, built with the model's `get_rope_index` per sample.
- Conditions: target g's visual generation model reads the language model's key/values over
  `[gen_start[g], gen_point[g])`, where `gen_point` is the target's first slot. One language model pass serves every
  target of the row.
"""

import torch

from .segments import PackedBatch


def assert_packing_supported(text_config=None):
    """Raise unless every kernel that reads a packed row's boundaries is in use.

    - the gated delta rule must take `cu_seqlens` (flash-linear-attention; the pure-torch fallback does not)
    - the causal conv must take `seq_idx` (causal_conv1d; the `nn.Conv1d` fallback does not)
    - flash-attention must be installed and `text_config` (the language model's config, if given) built with it;
      sdpa ignores `cu_seq_lens_q/k`
    """
    import inspect

    from integral import modeling_integral as m

    fn = m.chunk_gated_delta_rule
    assert fn is not None and "cu_seqlens" in inspect.signature(fn).parameters, (
        f"the gated delta rule in use ({getattr(fn, '__module__', None)}) takes no `cu_seqlens`: "
        f"this is the pure-torch fallback, and under it a packed row carries recurrent state from "
        f"one sample into the next, silently. Install flash-linear-attention or stop packing.")
    conv = m.causal_conv1d_fn
    assert conv is not None and "seq_idx" in inspect.signature(conv).parameters, (
        "causal_conv1d is not installed, so the model uses `F.silu(self.conv1d(x))`, which has "
        "no `seq_idx`: the depthwise conv then reaches across every packed boundary. Install "
        "causal_conv1d or stop packing.")

    from transformers.utils import is_flash_attn_2_available
    assert is_flash_attn_2_available(), (
        "flash-attn is not installed. The full-attention layers take their sample boundaries from "
        "`cu_seq_lens_q`/`cu_seq_lens_k`, and sdpa ignores both -- a packed row would be one "
        "sequence there. Install flash-attn, or stop packing and give each sample its own row.")
    if text_config is not None:
        impl = getattr(text_config, "_attn_implementation", None)
        assert impl is not None and impl.startswith("flash_attention_2"), (
            f"the language model was built with attn_implementation={impl!r}. flash-attn is "
            f"installed but unused, so `cu_seq_lens_q` reaches the mixer and is dropped, and every "
            f"sample in the row attends across the boundaries in front of it.")


def packing_kwargs(batch: PackedBatch) -> dict:
    """The boundary arguments for the model's forward (passed down to every layer as `**kwargs`).

    A padded tail is closed as its own sequence.
    """
    L = batch.input_ids.shape[1]
    dev = batch.input_ids.device
    cu = batch.cu_seqlens.to(dev)
    if int(cu[-1]) < L:
        cu = torch.cat([cu, torch.tensor([L], dtype=cu.dtype, device=dev)])
    cu = cu.to(torch.int32)
    lens = (cu[1:] - cu[:-1]).tolist()
    seq_idx = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.int32, device=dev),
        torch.tensor(lens, device=dev))[None, :]
    return {
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": int(max(lens)),
        "max_length_k": int(max(lens)),
        "seq_idx": seq_idx,
    }


def sample_position_ids(batch: PackedBatch, rope_index) -> torch.Tensor:
    """-> `[3, 1, L]` mrope positions for a packed row: `rope_index` (the model's `get_rope_index`) on each sample's
    slice. The padded tail keeps zeros."""
    assert batch.input_ids.shape[0] == 1, \
        f"packed positions want one row, got {batch.input_ids.shape[0]}"
    ids = batch.input_ids
    L = ids.shape[1]
    out = torch.zeros(3, 1, L, dtype=torch.long, device=ids.device)

    span = 0
    for s in range(len(batch.sample_lens)):
        lo, hi = int(batch.cu_seqlens[s]), int(batch.cu_seqlens[s + 1])
        n_span = batch.sample_spans[s]
        grids = batch.slot_grid[span:span + n_span]
        span += n_span
        thw = torch.stack([torch.ones(len(grids), dtype=torch.long, device=ids.device),
                           grids[:, 0], grids[:, 1]], dim=-1) if len(grids) else \
            torch.zeros(0, 3, dtype=torch.long, device=ids.device)
        pos, _ = rope_index(
            ids[:, lo:hi],
            batch.image_slots[:, lo:hi].to(torch.int32),
            image_grid_thw=thw,
            attention_mask=batch.attn[:, lo:hi],
        )
        out[:, :, lo:hi] = pos
    return out


def target_condition(batch: PackedBatch) -> torch.Tensor:
    """-> `[N_tgt, L]` bool: target g attends to `[gen_start[g], gen_point[g])`, its own sample up to its first
    slot."""
    B, L = batch.input_ids.shape
    assert B == 1, f"target_condition is for a packed row, got {B} rows"
    pos = torch.arange(L, device=batch.gen_point.device)
    return ((pos[None, :] >= batch.gen_start[:, None])
            & (pos[None, :] < batch.gen_point[:, None]))


def image_offsets(grids: list) -> list:
    """`[(gh, gw), ...]` -> the start of each image's latents in the flat `[total, dim]` buffer, plus the total."""
    off, n = [0], 0
    for gh, gw in grids:
        n += gh * gw
        off.append(n)
    return off


def slot_latents(batch: PackedBatch, latents: torch.Tensor, grids: list) -> list:
    """The latents `[n_lat_i, dim]` of each image span, in the order the spans appear in the row (`slot_image`)."""
    off = image_offsets(grids)
    return [latents[off[i]:off[i + 1]] for i in batch.slot_image.tolist()]
