#!/bin/bash
# Launch vision tokenizer training (train_vision_tokenizer.py) on 1 node x 8 GPUs.
#
#   STAGE=main CONFIG=configs/vision_tokenizer.yaml [RESUME=<run>/checkpoint-N] bash scripts/train_vision_tokenizer.sh
#   STAGE=gan  CONFIG=configs/vision_tokenizer_gan.yaml TOKENIZER=<tokenizer dir> [RESUME=<gan run dir>] \
#       bash scripts/train_vision_tokenizer.sh
#
# Paths (CONFIG, TOKENIZER, RESUME) are relative to the training directory, which the script runs in.
#
# Stage main resuming under cosine_with_min_lr: the scheduler is built for max_steps from a fresh start, so the
# config's num_cycles has to be the one that reaches min_lr at max_steps from the resume step; it is checked here.
set -o pipefail
cd "$(dirname "$0")/.."

STAGE="${STAGE:?set STAGE to main or gan}"
CONFIG="${CONFIG:?set CONFIG to a vision tokenizer config}"

# the repository root, for the `integral` package
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:${PYTHONPATH}}"
CFG_PROJECT=$(sed -n 's/^project:[[:space:]]*//p' "${CONFIG}" | head -1 | tr -d '"'"'"' \r')
export WANDB_PROJECT="${WANDB_PROJECT:-${CFG_PROJECT}}"
mkdir -p logs
RUN_NAME="$(basename "${CONFIG}" .yaml)-$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${RUN_NAME}.log"

case "${STAGE}" in
main)
    if [[ -n "${RESUME}" ]]; then
        OUTPUT_DIR="$(dirname "${RESUME}")"
        python - "${CONFIG}" "${RESUME}" <<'PY' || exit 1
import json, sys
import yaml
cfg = yaml.safe_load(open(sys.argv[1]))
step = json.load(open(f"{sys.argv[2]}/trainer_state.json"))["global_step"]
if cfg.get("lr_scheduler_type") == "cosine_with_min_lr" and cfg.get("reset_state"):
    total, warmup = cfg["max_steps"], cfg.get("warmup_steps", 0)
    need = 1 / (2 * ((total - step - warmup) / (total - warmup)))
    have = cfg["lr_scheduler_kwargs"]["num_cycles"]
    print(f"resume global_step {step}: num_cycles {have} (needed {need:.6f})")
    if abs(have - need) > 1e-5:
        sys.exit(f"num_cycles {have} does not reach min_lr at step {total} from {step}; set it to {need:.6f}")
PY
        echo "resuming from: ${RESUME}"
    else
        OUTPUT_DIR="ckpts/${RUN_NAME}"
    fi
    {
    accelerate launch --config_file configs/accelerate_vision_tokenizer.yaml train_vision_tokenizer.py main \
        --config_path "${CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "${RUN_NAME}" \
        --report_to wandb \
        --dataloader_num_workers 4 \
        --dataloader_prefetch_factor 4 \
        --bf16 true \
        ${RESUME:+--resume_from_checkpoint ${RESUME}} "${@}"
    } 2>&1 | tee -a "${LOG_FILE}"
    ;;
gan)
    [[ -n "${TOKENIZER}" ]] || { echo "set TOKENIZER to the vision tokenizer directory to tune" >&2; exit 1; }
    # The discriminator's DINO weights (`disc_ckpt` of the config, or of the arguments): a Hugging Face repo id is
    # downloaded here, once, before the 8 processes start, and they are given the local file.
    DISC_CKPT=$(python - "${CONFIG}" "${@}" <<'PY'
import sys

import yaml

from losses.dinodisc import fetch_dino_weights

path = yaml.safe_load(open(sys.argv[1])).get("disc_ckpt")
args = sys.argv[2:]
for i, a in enumerate(args):
    if a == "--disc_ckpt" and i + 1 < len(args):
        path = args[i + 1]
    elif a.startswith("--disc_ckpt="):
        path = a.split("=", 1)[1]
if not path:
    sys.exit("disc_ckpt is not set")
print(fetch_dino_weights(path))
PY
) || exit 1
    echo "discriminator weights: ${DISC_CKPT}"
    {
    torchrun --nproc_per_node=8 train_vision_tokenizer.py gan \
        --config_path "${CONFIG}" \
        --hf_pretrained "${TOKENIZER}" \
        ${RESUME:+--resume ${RESUME}} "${@}" \
        --disc_ckpt "${DISC_CKPT}"
    } 2>&1 | tee -a "${LOG_FILE}"
    ;;
*)
    echo "STAGE '${STAGE}' is neither main nor gan" >&2; exit 1 ;;
esac
