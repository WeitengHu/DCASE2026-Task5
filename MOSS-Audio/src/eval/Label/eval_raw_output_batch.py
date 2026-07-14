import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def build_instruction(question: str, choices: List[str]) -> Tuple[str, List[str]]:
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    choice_lines = "\n".join([f"{lab}. {choice}" for lab, choice in zip(labels, choices)])
    instruction = (
        "Answer the following multiple-choice question about the audio.\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{choice_lines}\n\n"
        f"Please think and reason step by step about the audio before you respond. After your reasoning, output only one uppercase letter from {', '.join(labels)}."
    )
    return instruction, labels


def build_chat_prompt(system_prompt: str, instruction: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_first_model_device(model):
    return next(model.parameters()).device


def get_model_float_dtype(model):
    return next(param.dtype for param in model.parameters() if param.is_floating_point())


def load_model_and_processor(
    model_path: str,
    lora_path: Optional[str],
    enable_time_marker: bool,
):
    import torch
    from peft import PeftModel

    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        device_map = "cuda:0"
    elif torch.backends.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"

    model = MossAudioModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype="auto",
        device_map=device_map,
    )

    if lora_path:
        print(f"Loading LoRA weights from {lora_path} ...")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)

    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        enable_time_marker=enable_time_marker,
    )
    return model, processor


def save_raw_output_csv(output_csv_path: str, rows: List[Dict]):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "raw_output", "choices", "instruction"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question": row["question"],
                    "raw_output": row.get("raw_output") or "",
                    "choices": row.get("choices") or "",
                    "instruction": row.get("instruction") or "",
                }
            )


def get_pad_token_id(processor) -> int:
    tokenizer = getattr(processor, "tokenizer", None) or getattr(processor, "_base_tokenizer", None)
    for attr_name in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            return int(token_id)
    return 0


def prepare_single_inputs(processor, item: Dict, system_prompt: str):
    from src.audio_io import load_audio

    instruction, _ = build_instruction(item["question_text"], item["multi_choice"])
    # prompt = build_chat_prompt(system_prompt, instruction)
    raw_audio = load_audio(item["abs_audio_path"], sample_rate=processor.config.mel_sr)
    # inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs = processor(text=instruction, audios=[raw_audio], return_tensors="pt")
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    return inputs


def pad_batch_inputs(processor, item_inputs: List[Dict]) -> Tuple[Dict, int]:
    import torch

    pad_token_id = get_pad_token_id(processor)
    input_id_rows = [inputs["input_ids"].squeeze(0) for inputs in item_inputs]
    attention_rows = [inputs["attention_mask"].squeeze(0) for inputs in item_inputs]
    audio_mask_rows = [inputs["audio_input_mask"].squeeze(0) for inputs in item_inputs]
    max_text_len = max(row.shape[0] for row in input_id_rows)

    padded_input_ids = []
    padded_attention_masks = []
    padded_audio_input_masks = []
    for input_ids, attention_mask, audio_input_mask in zip(
        input_id_rows,
        attention_rows,
        audio_mask_rows,
    ):
        pad_len = max_text_len - input_ids.shape[0]
        padded_input = torch.full(
            (max_text_len,),
            pad_token_id,
            dtype=input_ids.dtype,
        )
        padded_attention = torch.zeros(
            (max_text_len,),
            dtype=attention_mask.dtype,
        )
        padded_audio_mask = torch.zeros(
            (max_text_len,),
            dtype=audio_input_mask.dtype,
        )
        padded_input[pad_len:] = input_ids
        padded_attention[pad_len:] = attention_mask
        padded_audio_mask[pad_len:] = audio_input_mask
        padded_input_ids.append(padded_input)
        padded_attention_masks.append(padded_attention)
        padded_audio_input_masks.append(padded_audio_mask)

    batch = {
        "input_ids": torch.stack(padded_input_ids, dim=0),
        "attention_mask": torch.stack(padded_attention_masks, dim=0),
        "audio_input_mask": torch.stack(padded_audio_input_masks, dim=0),
    }

    audio_rows = [inputs.get("audio_data") for inputs in item_inputs]
    if all(audio_data is not None for audio_data in audio_rows):
        mel_dim = audio_rows[0].shape[1]
        max_audio_len = max(audio_data.shape[-1] for audio_data in audio_rows)
        audio_batch = torch.zeros(
            (len(audio_rows), mel_dim, max_audio_len),
            dtype=audio_rows[0].dtype,
        )
        for row_index, audio_data in enumerate(audio_rows):
            audio_data = audio_data.squeeze(0)
            audio_batch[row_index, :, : audio_data.shape[-1]] = audio_data
        batch["audio_data"] = audio_batch
        batch["audio_data_seqlens"] = torch.cat(
            [inputs["audio_data_seqlens"].reshape(-1) for inputs in item_inputs],
            dim=0,
        )

    return batch, max_text_len


def move_batch_to_model(batch: Dict, model) -> Dict:
    import torch

    input_device = get_first_model_device(model)
    model_dtype = get_model_float_dtype(model)
    for key, value in list(batch.items()):
        if torch.is_tensor(value):
            value = value.to(input_device)
            if torch.is_floating_point(value):
                value = value.to(model_dtype)
            batch[key] = value
    return batch


def run_batch_raw(
    model,
    processor,
    batch_items: List[Dict],
    system_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> List[str]:
    import torch

    item_inputs = [prepare_single_inputs(processor, item, system_prompt) for item in batch_items]
    batch, prompt_len = pad_batch_inputs(processor, item_inputs)
    batch = move_batch_to_model(batch, model)
    pad_token_id = get_pad_token_id(processor)

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=1,
        use_cache=True,
        pad_token_id=pad_token_id,
    )
    if do_sample:
        generate_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        sequences = model.generate(**batch, **generate_kwargs)

    decoded_list = processor.batch_decode(
        sequences[:, prompt_len:],
        skip_special_tokens=True,
    )
    return [str(output).strip() for output in decoded_list]


def eval_raw_output(
    model,
    processor,
    dataset: List[Dict],
    raw_output_csv_path: str,
    system_prompt: str,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
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
            system_prompt,
            max_new_tokens,
            do_sample,
            temperature,
            top_p,
            top_k,
        )
        for (row_index, _), raw_output in zip(batch_pairs, batch_outputs):
            raw_outputs[row_index] = raw_output
        evaluated += len(batch_items)
        if print_every and (evaluated % print_every == 0 or evaluated == len(valid_items)):
            print(f"raw outputs generated: {evaluated}/{len(valid_items)}")

    rows = [
        {
            "question": item["id"],
            "choices": json.dumps(item["multi_choice"], ensure_ascii=False),
            "instruction": build_instruction(item["question_text"], item["multi_choice"])[0],
            "raw_output": raw_outputs[row_index],
        }
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
