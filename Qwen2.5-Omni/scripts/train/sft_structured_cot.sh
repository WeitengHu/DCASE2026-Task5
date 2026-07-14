#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MASTER_PORT=29501
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENABLE_AUDIO_OUTPUT=false
export USE_AUDIO_IN_VIDEO=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

swift sft \
  --model "path/to/Qwen2.5-Omni-7B" \
  --model_type qwen2_5_omni \
  --dataset "path/to/qwen_structured_cot_sft.jsonl" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_vit false \
  --freeze_aligner false \
  --freeze_llm false \
  --max_length 4096 \
  --truncation_strategy delete \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --lazy_tokenize true \
  --split_dataset_ratio 0.01 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --save_steps 10 \
  --eval_steps 10 \
  --logging_steps 1 \
  --save_total_limit 2 \
  --output_dir "path/to/output" \
  --check_model false \
  --warmup_steps 50 \
  --report_to wandb tensorboard
