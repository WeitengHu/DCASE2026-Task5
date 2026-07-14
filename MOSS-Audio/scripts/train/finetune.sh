#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MOSS_AUDIO_UPSTREAM="${REPO_ROOT}/third_party/MOSS-Audio"
MOSS_FINETUNE_SCRIPT="${MOSS_AUDIO_UPSTREAM}/finetune/finetune.py"

if [[ ! -f "${MOSS_FINETUNE_SCRIPT}" ]]; then
    echo "Missing MOSS-Audio submodule at ${MOSS_AUDIO_UPSTREAM}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

export PYTHONPATH="${MOSS_AUDIO_UPSTREAM}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

accelerate launch --num_processes 4 "${MOSS_FINETUNE_SCRIPT}" \
    --model_dir path/to/moss-audio/model \
    --data_path path/to/train_data \
    --eval_data_path path/to/eval_data \
    --eval_strategy steps \
    --save_strategy steps \
    --output_dir path/to/output \
    --attn_implementation sdpa \
    --max_len 2048 \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_on_audio_encoder true \
    --bf16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_steps 10 \
    --lr_scheduler_type cosine \
    --warmup_steps 50 \
    --save_total_limit 5 \
    --max_grad_norm 1.0 \
    --report_to wandb
