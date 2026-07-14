#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Training starts from the answer-only SFT adapter.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MASTER_PORT=29502
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENABLE_AUDIO_OUTPUT=false
export USE_AUDIO_IN_VIDEO=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

swift rlhf \
  --rlhf_type grpo \
  --model "path/to/Qwen2.5-Omni-7B" \
  --model_type qwen2_5_omni \
  --adapters "path/to/answer_only_sft_checkpoint" \
  --ref_adapters "path/to/answer_only_sft_checkpoint" \
  --dataset "path/to/qwen_cot_grpo_data.jsonl" \
  --external_plugins "Qwen2.5-Omni/src/reward/cot_reward_plugin.py" \
  --reward_funcs external_audio_choice_accuracy external_cot_format \
  --reward_weights 2.0 0.5 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --generation_batch_size 32 \
  --num_generations 8 \
  --beta 0.001 \
  --max_completion_length 512 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 50 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
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
  --split_dataset_ratio 0 \
  --dataloader_num_workers 2 \
  --dataset_num_proc 2 \
  --save_steps 10 \
  --eval_steps 10 \
  --logging_steps 1 \
  --save_total_limit 2 \
  --output_dir "path/to/output" \
  --check_model false \
  --report_to wandb tensorboard \
  --log_completions true \
  --warmup_ratio 0.01
