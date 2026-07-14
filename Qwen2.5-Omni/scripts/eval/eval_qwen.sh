python Qwen2.5-Omni/src/eval/qwen_cot/main.py \
    --model_path "path/to/Qwen2.5-Omni-7B" \
    --embedding_model_path "path/to/embedding_model" \
    --output_dir "path/to/output/qwen_cot" \
    --lora_path "path/to/lora_checkpoint" \
    --dataset_path "path/to/dataset.jsonl" \
    --dataset_root "path/to/dataset_root"

python Qwen2.5-Omni/src/eval/qwen_structured_cot/main.py \
    --model_path "path/to/Qwen2.5-Omni-7B" \
    --embedding_model_path "path/to/embedding_model" \
    --output_dir "path/to/output/qwen_structured_cot" \
    --lora_path "path/to/lora_checkpoint" \
    --dataset_path "path/to/dataset.jsonl" \
    --dataset_root "path/to/dataset_root"
