<div align="center">

# Integral

**Let Vision-Language Models Generate Images through Their Representations**

[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white)](https://thejb.ai/integral/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.31237-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2611.31237)
[![Model](https://img.shields.io/badge/Model-Integral--9B-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/the-jb/Integral-9B)

</div>

Integral is a unified multimodal model that extends pretrained vision-language models with image generation through
shared visual representations. Built upon Qwen3.5-9B, Integral supports image understanding, text-to-image
generation, and image editing within a single framework.

Integral introduces two complementary components: Abstraction Fusion, which constructs a shared visual representation
space for understanding and generation, and Block-wise MoT, which enables efficient visual generation while leveraging
pretrained multimodal knowledge.

## Setup

```bash
git clone https://github.com/the-jb/integral.git
cd integral
pip install -r requirements.txt
```

Run from the repository root (or put it on `PYTHONPATH`).

## Load model

```python
import torch
from integral import IntegralForConditionalGeneration, IntegralProcessor

repo = "the-jb/Integral-9B"
model = IntegralForConditionalGeneration.from_pretrained(
    repo, dtype=torch.bfloat16,
    attn_implementation={"text_config": "flash_attention_2"},
).to("cuda").eval()
processor = IntegralProcessor.from_pretrained(repo)
```

Do not call `model.to(torch.bfloat16)` afterwards: parts of the model stay in fp32.

## Text-to-image

```python
inputs = processor.prepare_image_generation(["a photo of a red apple on a wooden table"], height=512, width=512)
out = model.generate_image(**inputs, generator=torch.Generator().manual_seed(0))
out.images[0].save("apple.png")
```

Height and width are multiples of 32.

| Argument | Default | Description |
|---|---|---|
| `solver` | `"dpm_solver++"` | Sampler: `"dpm_solver++"` or `"euler"` |
| `num_inference_steps` | 50 | Number of sampling steps |
| `guidance_scale` | 4.0 | Text guidance scale |
| `time_shift` | 4.0 | Timestep schedule shift |
| `num_images_per_prompt` | 1 | Number of images per prompt |
| `latents` / `generator` | `None` | Initial noise, or the generator it is drawn from |
| `output_type` | `"pil"` | `"pil"`, `"pt"` or `"latent"` |

## Image editing

```python
from PIL import Image

source = Image.open("input.jpg").convert("RGB").resize((512, 512), Image.BICUBIC)
inputs = processor.prepare_image_generation(["Change the color of the car to red."], height=512, width=512,
                                            images=[source])
edited = model.generate_image(**inputs, generator=torch.Generator().manual_seed(0))
edited.images[0].save("edited.png")
```

Input image sides are multiples of 32.

| Argument | Default | Description |
|---|---|---|
| `image_guidance_scale` | 1.5 | Image guidance scale; `None` for text guidance only |
| `guidance_interval` | `(0.0, 1.0)` | Timestep range where guidance is applied |
| `guidance_renorm` | `"text_channel"` | Guidance renormalization: `"text_channel"`, `"global"` or `"none"` |
| `guidance_renorm_min` | 1.0 | Lower bound of the renormalization ratio |

## Image understanding

```python
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = processor(text=[text], images=[source], return_tensors="pt").to("cuda")
answer = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(processor.tokenizer.decode(answer[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## vLLM

From the repository root:

```python
from integral.vllm import register
register()   # at the script's top level, before creating the LLM

from vllm import LLM, SamplingParams
llm = LLM(model="the-jb/Integral-9B")
out = llm.generate({"prompt": text, "multi_modal_data": {"image": [source]}},
                   SamplingParams(temperature=0, max_tokens=256))
```

## Citation

```bibtex
@article{kim2026integral,
  title   = {Let Vision-Language Models Generate Images through
             Their Representations},
  author  = {Kim, JB and Jeon, Byungwoo and Lee, Kyungmin and Kang, Hyeonmo and
             Park, Cheonbok and Joo, Shinyoung and Shin, Jinwoo and Kim, Jeonghoon},
  journal = {arXiv preprint arXiv:2611.31237},
  year    = {2026}
}
```
