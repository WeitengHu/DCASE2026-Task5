#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export MASTER_PORT=29503
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ENABLE_AUDIO_OUTPUT=false
export USE_AUDIO_IN_VIDEO=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export STRUCTURED_COT_REWARD_EMBEDDING_BACKEND=qwen
export STRUCTURED_COT_REWARD_BGE_PATH="path/to/bge-m3"
export STRUCTURED_COT_REWARD_QWEN_PATH="path/to/Qwen3-Embedding-0.6B"
export STRUCTURED_COT_REWARD_TOKENIZER_PATH="path/to/Qwen2.5-Omni-7B"
export STRUCTURED_COT_REWARD_DEVICE=auto
export STRUCTURED_COT_REWARD_BATCH_SIZE=1
export STRUCTURED_COT_REWARD_MAX_LENGTH=1024
export STRUCTURED_COT_REWARD_LEN_MIN_TOKENS=100
export STRUCTURED_COT_REWARD_LEN_MAX_TOKENS=500

swift rlhf \
  --rlhf_type grpo \
  --model "path/to/Qwen2.5-Omni-7B" \
  --model_type qwen2_5_omni \
  --adapters "path/to/sft_structured_cot_checkpoint" \
  --ref_adapters "path/to/sft_structured_cot_checkpoint" \
  --dataset "path/to/qwen_structured_cot_gdpo.jsonl" \
  --external_plugins "Qwen2.5-Omni/src/reward/structured_cot_reward_plugin.py" \
  --reward_funcs structured_cot_accuracy structured_cot_format structured_cot_qtype structured_cot_sim_question_analysis structured_cot_sim_audio_evidence structured_cot_sim_reasoning structured_cot_length \
  --reward_weights 2.0 0.5 0.5 0.25 0.5 0.5 0.25 \
  --scale_rewards gdpo \
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
  --save_steps 20 \
  --eval_steps 20 \
  --logging_steps 1 \
  --save_total_limit 5 \
  --output_dir "path/to/output" \
  --check_model false \
  --report_to wandb tensorboard \
  --log_completions true \
  --warmup_ratio 0.01
