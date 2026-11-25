"""Integral in vLLM (image understanding and text generation; image generation is transformers'
`IntegralForConditionalGeneration.generate_image`).

Call `register()` at the top level of the script that builds the vLLM engine (`vllm.LLM`), before building it. vLLM
forks the engine process, which inherits the registration, or spawns it when CUDA is already initialized; a spawned
process re-runs the script's top level, `register()` included. It registers
  - `IntegralVLLMConfig` for `model_type` "integral" with `AutoConfig` (the config vLLM's Qwen3.5 code reads)
  - `IntegralForConditionalGeneration` (the `architectures` entry of a converted directory's config.json) with
    vLLM's model registry, loaded lazily from `integral.vllm.model`
"""


def register():
    import integral  # noqa: F401  (registers the transformers classes, IntegralProcessor among them)
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .config import IntegralVLLMConfig

    AutoConfig.register("integral", IntegralVLLMConfig, exist_ok=True)
    if "IntegralForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("IntegralForConditionalGeneration",
                                     "integral.vllm.model:IntegralForConditionalGeneration")
