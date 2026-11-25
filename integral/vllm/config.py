"""The Integral config as vLLM's Qwen3.5 code reads it.

vLLM's Qwen3.5 implementation (`vllm/model_executor/models/qwen3_5.py`, `qwen3_vl.py`) reads the config as vLLM's own
`Qwen3_5Config` (`vllm/transformers_utils/configs/qwen3_5.py`): `text_config` as its `Qwen3_5TextConfig` and the vision
geometry from `vision_config` (`patch_size`, `spatial_merge_size`, `temporal_patch_size`, `out_hidden_size`,
`deepstack_visual_indexes`). `IntegralVLLMConfig` is read from the same `config.json` as `IntegralConfig`: the text
config is the same values as Qwen3.5-9B's (the converter copies them), the vision geometry comes from
`vision_tokenizer_config`, and `vision_tokenizer_config` itself is kept for the vision tokenizer the vLLM model builds.
Only `integral.vllm` uses this class.
"""

from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config

from ..configuration_integral import IntegralVisionTokenizerConfig


def _as_dict(config):
    if config is None or isinstance(config, dict):
        return config
    return config.to_dict()


class IntegralVLLMConfig(Qwen3_5Config):
    model_type = "integral"

    def __init__(self, text_config=None, vision_tokenizer_config=None, visual_generation_config=None, **kwargs):
        # vLLM's Qwen3_5Config builds its sub-configs from dicts only
        text_config = _as_dict(text_config)
        if text_config is not None:
            text_config = {k: v for k, v in text_config.items() if k != "model_type"}
        if isinstance(vision_tokenizer_config, dict):
            vision_tokenizer_config = IntegralVisionTokenizerConfig(
                **{k: v for k, v in vision_tokenizer_config.items() if k != "model_type"})
        elif vision_tokenizer_config is None:
            vision_tokenizer_config = IntegralVisionTokenizerConfig()
        vt = vision_tokenizer_config
        vision_config = dict(
            depth=vt.vision_num_hidden_layers, hidden_size=vt.vision_hidden_size,
            intermediate_size=vt.vision_intermediate_size, num_heads=vt.vision_num_attention_heads,
            patch_size=vt.patch_size, spatial_merge_size=vt.spatial_merge_size,
            temporal_patch_size=vt.temporal_patch_size, out_hidden_size=vt.merger_out_hidden_size,
            num_position_embeddings=vt.vision_num_position_embeddings)
        kwargs.pop("vision_config", None)
        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)
        self.vision_config.deepstack_visual_indexes = []   # as Qwen3.5-9B's config: the merger output only
        self.vision_tokenizer_config = vision_tokenizer_config
        # the generation side is not loaded by the vLLM model; kept so the config round-trips
        self.visual_generation_config = visual_generation_config
