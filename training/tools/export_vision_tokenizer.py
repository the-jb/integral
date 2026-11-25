"""A stage-`main` checkpoint of train_vision_tokenizer.py -> an IntegralVisionTokenizer directory.

The checkpoint holds the TrainingModel's namespace: the tokenizer under `model.`, plus the frozen distillation target
(`teacher_merger.`) and LPIPS (`lpips.`), which are not part of the tokenizer and are left out. The geometry comes
from the run's own `training_config.yaml`. The vision tower's weights are taken from the checkpoint when it has them
and from the run's `vision_model_path` otherwise (they are frozen, so the two are the same tensors). Every key of the
tokenizer must load; anything missing or left over stops the export.

    python tools/export_vision_tokenizer.py --checkpoint <run>/checkpoint-N --out <new tokenizer dir>
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from integral.configuration_integral import IntegralVisionTokenizerConfig  # noqa: E402
from integral.modeling_integral import IntegralVisionTokenizer  # noqa: E402
from train_vision_tokenizer import load_vision_tower  # noqa: E402

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "pytorch_model_fsdp.bin")
NOT_THE_TOKENIZER = ("teacher_merger.", "lpips.")


def load_checkpoint(path: Path) -> dict:
    for name in WEIGHT_FILES:
        f = path / name
        if f.exists():
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file
                return load_file(str(f))
            return torch.load(f, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"none of {WEIGHT_FILES} in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a checkpoint-N directory of a stage-main run")
    ap.add_argument("--config", default=None,
                    help="the run's config (default: training_config.yaml next to the checkpoint)")
    ap.add_argument("--out", required=True, help="output directory; must not exist")
    args = ap.parse_args()
    assert not os.path.exists(args.out), f"{args.out} exists; refusing to write into it"

    ckpt = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else ckpt.parent / "training_config.yaml"
    with open(config_path) as f:
        run = yaml.safe_load(f)

    state = load_checkpoint(ckpt)
    left_over = sorted(k for k in state if not k.startswith(("model.",) + NOT_THE_TOKENIZER))
    if left_over:
        raise KeyError(f"{len(left_over)} checkpoint keys belong to nothing known, e.g. {left_over[:5]}")
    tokenizer_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}

    model = IntegralVisionTokenizer(IntegralVisionTokenizerConfig(**run["model_config"]))
    if not any(k.startswith("visual.") for k in tokenizer_state):
        load_vision_tower(model, run["vision_model_path"])
        print(f"[export] vision tower from {run['vision_model_path']}")
    missing, unexpected = model.load_state_dict(tokenizer_state, strict=False)
    missing = [k for k in missing if not k.startswith("visual.") or any(s.startswith("visual.")
                                                                       for s in tokenizer_state)]
    if missing or unexpected:
        raise RuntimeError(f"missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
    model.save_pretrained(args.out, max_shard_size="50GB")     # one model.safetensors
    print(f"[export] {len(tokenizer_state)} tensors from {ckpt} -> {args.out}")


if __name__ == "__main__":
    main()
