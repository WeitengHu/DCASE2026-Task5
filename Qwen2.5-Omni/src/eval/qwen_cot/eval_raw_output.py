import csv
import time
from pathlib import Path
from typing import Dict, List, Optional


def build_instruction(question: dict, choices: bool) -> str:
    choice_lines = "\n".join(str(choice) for choice in choices)
    prompt = (
        "Answer the following multiple-choice question about the audio.\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choice_lines}\n\n"
    )
    prompt += "Please think and reason step by step about the audio through the reasoning in <think> </think> before the final answer."
    prompt += "Output the final answer in <answer> </answer>"
    return prompt

def get_first_model_device(model):
    return next(model.parameters()).device


def get_model_float_dtype(model):
    return next(param.dtype for param in model.parameters() if param.is_floating_point())


def load_model_and_processor(
    model_path: str,
    lora_path: Optional[str],
    use_flash_attn: bool,
    max_memory: Dict,
):
    import torch
    from peft import PeftModel
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"

    try:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
    except Exception as exc:
        print(f"[Warn] load with {attn_impl} failed: {exc}")
        print("[Info] fallback to sdpa ...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

    if lora_path is not None:
        print(f"Loading LoRA weights from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

    if hasattr(model, "disable_talker"):
        try:
            model.disable_talker()
        except Exception:
            pass

    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


def save_raw_output_csv(output_csv_path: str, rows: List[Dict]):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "raw_output"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"question": row["question"], "raw_output": row.get("raw_output") or ""})


def run_batch_raw(
    model,
    processor,
    batch_items: List[Dict],
    max_new_tokens: int,
    use_audio_in_video: bool,
) -> List[str]:
    import torch
    from qwen_omni_utils import process_mm_info

    conversations = []
    for item in batch_items:
        instruction = build_instruction(item["question_text"], item["multi_choice"])
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": item["abs_audio_path"]},
                        {"type": "text", "text": instruction},
                    ],
                },
            ]
        )

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )

    input_device = get_first_model_device(model)
    model_dtype = get_model_float_dtype(model)
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(input_device)
            if torch.is_floating_point(value):
                value = value.to(model_dtype)
            inputs[key] = value

    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.generation_config.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generate_ids = sequences[:, prompt_len:]
    decoded_list = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [str(output).strip() for output in decoded_list]


def eval_raw_output(
    model,
    processor,
    dataset: List[Dict],
    raw_output_csv_path: str,
    batch_size: int,
    max_new_tokens: int,
    use_audio_in_video: bool,
    print_every: int,
) -> Dict:
    raw_output_csv_path = Path(raw_output_csv_path)
    raw_outputs = [""] * len(dataset)
    valid_items = []
    missing_audio = 0

    for row_index, item in enumerate(dataset):
        if not Path(item["abs_audio_path"]).exists():
            missing_audio += 1
            continue
        valid_items.append((row_index, item))

    evaluated = 0
    started_at = time.time()
    for start in range(0, len(valid_items), batch_size):
        batch_pairs = valid_items[start : start + batch_size]
        batch_items = [item for _, item in batch_pairs]
        batch_outputs = run_batch_raw(
            model,
            processor,
            batch_items,
            max_new_tokens,
            use_audio_in_video,
        )
        for (row_index, _), raw_output in zip(batch_pairs, batch_outputs):
            raw_outputs[row_index] = raw_output
        evaluated += len(batch_items)
        if print_every and (evaluated % print_every == 0 or evaluated == len(valid_items)):
            print(f"raw outputs generated: {evaluated}/{len(valid_items)}")

    rows = [
        {"question": item["id"], "raw_output": raw_outputs[row_index]}
        for row_index, item in enumerate(dataset)
    ]
    save_raw_output_csv(raw_output_csv_path, rows)
    return {
        "raw_output_csv": str(raw_output_csv_path),
        "total_samples": len(dataset),
        "valid_evaluated_samples": evaluated,
        "missing_audio": missing_audio,
        "elapsed_seconds": time.time() - started_at,
    }
