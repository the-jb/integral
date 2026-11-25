#!/bin/bash
# Launch visual generation training (train_visual_generation.py) on 1, 2 or 8 nodes.
#
#   NODES=8 CONFIG=configs/visual_generation_frozen.yaml MODEL_PATH=<Integral directory> bash scripts/train_visual_generation.sh
#   RESUME=<run>/checkpoint-N ... bash scripts/train_visual_generation.sh     # resume into the same run directory
#
# Paths (CONFIG, MODEL_PATH, RESUME) are relative to the training directory, which the script runs in.
#
# Multi-node runs are launched from inside a Slurm allocation (one task per node, srun).
#
# FSDP_IGNORED_MODULES follows the config's train_language_model. Only frozen modules may be listed: an ignored
# module is neither sharded nor gradient-reduced.
#   false  the vision tokenizer's vision tower, abstraction fusion, bottleneck, merger and pixel decoder, the
#          language model and its head (the visual generation model and the visual generation adapter train)
#   true   the vision tokenizer's vision tower, abstraction fusion, bottleneck and pixel decoder (the language model,
#          its head and the merger train too)
set -o pipefail
cd "$(dirname "$0")/.."

NODES="${NODES:-1}"
case "${NODES}" in
    1) DEFAULT_ACC=configs/accelerate_visual_generation_1node.yaml ;;
    2) DEFAULT_ACC=configs/accelerate_visual_generation_hsdp2.yaml ;;
    8) DEFAULT_ACC=configs/accelerate_visual_generation_hsdp8.yaml ;;
    *) echo "NODES=${NODES} has no accelerate config; copy the hsdp8 one with dp_replicate_size=${NODES}" >&2
       exit 1 ;;
esac
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACC}}"
CONFIG="${CONFIG:?set CONFIG to a visual generation config}"
[[ -n "${MODEL_PATH}" ]] || { echo "set MODEL_PATH to the Integral directory to start from" >&2; exit 1; }

# the repository root, for the `integral` package
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:${PYTHONPATH}}"

CFG_PROJECT=$(sed -n 's/^project:[[:space:]]*//p' "${CONFIG}" | head -1 | tr -d '"'"'"' \r')
export WANDB_PROJECT="${WANDB_PROJECT:-${CFG_PROJECT}}"

CFG_TRAIN_LM=$(sed -n 's/^train_language_model:[[:space:]]*//p' "${CONFIG}" | head -1 | tr -d '"'"'"' \r')
TOKENIZER_FROZEN='^model\.vision_tokenizer\.(visual|abstraction_fusion|bottleneck|pixel_decoder)($|\.)'
case "${CFG_TRAIN_LM}" in
    false) export FSDP_IGNORED_MODULES="${TOKENIZER_FROZEN}|^model\.vision_tokenizer\.merger($|\.)|^model\.language_model($|\.)|^lm_head($|\.)" ;;
    true)  export FSDP_IGNORED_MODULES="${TOKENIZER_FROZEN}" ;;
    *) echo "train_language_model '${CFG_TRAIN_LM}' in ${CONFIG} is neither true nor false" >&2; exit 1 ;;
esac
echo "train_language_model=${CFG_TRAIN_LM} (from ${CONFIG})"
echo "FSDP_IGNORED_MODULES=${FSDP_IGNORED_MODULES}"

if [[ -n "${RESUME}" ]]; then
    OUTPUT_DIR="$(dirname "${RESUME}")"
    # job_name is prefixed back onto output_dir by the arguments' __post_init__
    JOB_NAME=$(sed -n 's/^job_name:[[:space:]]*//p' "${CONFIG}" | head -1 | tr -d '"'"'"' \r')
    OUTPUT_DIR="$(dirname "${OUTPUT_DIR}")/$(basename "${OUTPUT_DIR}" | sed "s/^${JOB_NAME}-//")"
    echo "resuming from: ${RESUME}"
else
    # job_name is prefixed onto it by the arguments' __post_init__
    OUTPUT_DIR="ckpts/$(date +%Y%m%d_%H%M%S)"
fi
RUN_NAME="$(basename "${OUTPUT_DIR}")"
mkdir -p logs
LOG_FILE="logs/$(basename "${CONFIG}" .yaml)-${RUN_NAME}.log"

TRAIN_ARGS=(train_visual_generation.py
    --config_path "${CONFIG}"
    --model_path "${MODEL_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --run_name "${RUN_NAME}"
    --report_to wandb
    --dataloader_num_workers 4
    --dataloader_prefetch_factor 4
    --bf16 true
    ${RESUME:+--resume_from_checkpoint ${RESUME}} "${@}")

{
echo "nodes=${NODES}  accelerate_config=${ACCELERATE_CONFIG}  config=${CONFIG}"
if [[ "${NODES}" -gt 1 ]]; then
    HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)
    HEAD_IP=$(scontrol show node "${HEAD_NODE}" | tr ' ' '\n' | grep '^NodeAddr=' | cut -d= -f2)
    echo "multi-node: ${NODES} nodes (${SLURM_JOB_NODELIST}), head ${HEAD_NODE} = ${HEAD_IP}"
    # the arguments quoted for the inner shell
    printf -v TRAIN_ARGS_QUOTED '%q ' "${TRAIN_ARGS[@]}"
    srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
        accelerate launch --config_file ${ACCELERATE_CONFIG} \
            --num_machines ${NODES} \
            --num_processes $((NODES * 8)) \
            --machine_rank \$SLURM_NODEID \
            --main_process_ip ${HEAD_IP} \
            --main_process_port ${MASTER_PORT:-29500} \
            ${TRAIN_ARGS_QUOTED}"
else
    accelerate launch --config_file "${ACCELERATE_CONFIG}" "${TRAIN_ARGS[@]}"
fi
} 2>&1 | tee -a "${LOG_FILE}"
