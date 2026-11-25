"""Integral. Importing the package registers its classes with transformers' Auto classes, so a directory whose
config has `model_type` `integral` loads through `AutoConfig`, `AutoModelForImageTextToText` and `AutoProcessor`
with this package's code and no `trust_remote_code` prompt.

`register` only adds to the Auto mappings: saving writes no `auto_map` and copies no code files (that is
`register_for_auto_class`, which nothing here calls).
"""

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from .configuration_integral import IntegralConfig
from .modeling_integral import IntegralForConditionalGeneration
from .processing_integral import IntegralProcessor

AutoConfig.register("integral", IntegralConfig)
AutoModelForImageTextToText.register(IntegralConfig, IntegralForConditionalGeneration)
AutoProcessor.register(IntegralConfig, IntegralProcessor)
