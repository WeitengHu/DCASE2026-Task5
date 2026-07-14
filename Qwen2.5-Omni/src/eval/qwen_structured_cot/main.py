import argparse
import hashlib
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple


from eval_raw_output import eval_raw_output, load_model_and_processor
from metrics import metrics
from postprocess import postprocess
from postprocess import vote
from summary import summary



SYSTEM_PROMPT = """You are an audio understanding model that answers multiple choice questions based on audio content. Your answer MUST follow this format:

<think>
<question_analysis>[Identify what evidence the question asks for and clarify what evidence you need to find in the audio]</question_analysis>
<question_type>[The question type that must be exactly one of: sound, speech, music, temporal]</question_type>
<audio_evidence>[Describe concrete audio evidence heard in the audio and relevant to the question]</audio_evidence>
<reasoning>[Think and reason step by step to answer the question based on the audio evidence]</reasoning>
</think>
<answer>[Your final answer choice]</answer>

Rules:
  - The content of <question_type> must be exactly one of: sound, speech, music, temporal.
  - The content of <answer> must exactly match one provided choice.
  - Do not output any extra text before or after the schema.
  - Do not invent audio evidence that is not supported by the audio.
  - The content of <question_analysis> and <reasoning> should be no longer than 100 words and no less than 5 words each."""

DEFAULT_MODEL_PATH = "path/to/Qwen2.5-Omni-7B"
DEFAULT_DATASET_ROOT = Path("path/to/dataset_root")
DEFAULT_DATASET_PATH = DEFAULT_DATASET_ROOT / "dev.jsonl"
RAW_OUTPUT_FILENAME = "raw_output.csv"
OUTPUT_FILENAME = "output.csv"
METRICS_FILENAME = "metrics.json"
VOTE_METRICS_FILENAME = "vote_metrics.json"
SUMMARY_FILENAME = "summary.json"
NUM_RUNS = 5
BASE_SEED = 42
USE_FLASH_ATTN = True
USE_AUDIO_IN_VIDEO = False
BATCH_SIZE = 8
MAX_NEW_TOKENS = 4096
PRINT_EVERY = 20
MAX_MEMORY = {0: "24GiB", "cpu": "120GiB"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embedding_model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    return parser.parse_args()


def load_dataset(jsonl_path: Path, dataset_root: Path) -> List[Dict]:
    dataset = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = Path(item["audio_path"])
            abs_audio_path = audio_path if audio_path.is_absolute() else dataset_root / audio_path
            dataset.append(
                {
                    "id": item["id"],
                    "audio_path": item["audio_path"],
                    "abs_audio_path": str(abs_audio_path),
                    "question_text": item["question_text"],
                    "multi_choice": item["multi_choice"] if isinstance(item.get("multi_choice"), list) else [],
                    "answer": item["answer"],
                }
            )
    return dataset


def stable_seed_from_text(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def build_run_dataset(dataset: List[Dict], run_index: int) -> List[Dict]:
    run_dataset = []
    for item in dataset:
        run_item = deepcopy(item)
        choices = list(run_item["multi_choice"])
        if run_index == 0:
            permutation = list(range(len(choices)))
        else:
            local_seed = stable_seed_from_text(f"{BASE_SEED}::{run_index}::{run_item['id']}")
            rng = random.Random(local_seed)
            permutation = list(range(len(choices)))
            rng.shuffle(permutation)
        run_item["multi_choice"] = [choices[index] for index in permutation]
        run_dataset.append(run_item)
    return run_dataset


def get_run_paths(output_dir: Path, run_index: int) -> Tuple[Path, Path, Path, Path]:
    suffix = "original" if run_index == 0 else "shuffle"
    run_output_dir = output_dir / f"run_{run_index:02d}_{suffix}"
    return (
        run_output_dir,
        run_output_dir / RAW_OUTPUT_FILENAME,
        run_output_dir / OUTPUT_FILENAME,
        run_output_dir / METRICS_FILENAME,
    )


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    output_dir = Path(args.output_dir)
    dataset_path = Path(args.dataset_path)
    dataset_root = Path(args.dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MODEL_PATH   : {args.model_path}")
    print(f"EMBEDDING_MODEL_PATH: {args.embedding_model_path}")
    print(f"LORA_PATH    : {args.lora_path}")
    print(f"DATASET_PATH : {dataset_path}")
    print(f"DATASET_ROOT : {dataset_root}")
    print(f"OUTPUT_DIR   : {output_dir}")
    print(f"NUM_RUNS     : {NUM_RUNS}")
    print(f"BASE_SEED    : {BASE_SEED}")
    print(f"SYSTEM_PROMPT: {SYSTEM_PROMPT}")

    started_at = time.time()
    dataset = load_dataset(dataset_path, dataset_root)
    print(f"Total samples in file: {len(dataset)}")

    model, processor = load_model_and_processor(
        args.model_path,
        args.lora_path,
        use_flash_attn=USE_FLASH_ATTN,
        max_memory=MAX_MEMORY,
    )

    metrics_json_paths = []
    for run_index in range(NUM_RUNS):
        mode = "original order" if run_index == 0 else "shuffled choices"
        run_output_dir, raw_output_csv_path, output_csv_path, metrics_json_path = get_run_paths(
            output_dir,
            run_index,
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
        run_dataset = build_run_dataset(dataset, run_index)

        print("\n" + "=" * 70)
        print(f"Starting run {run_index + 1}/{NUM_RUNS}: {mode} -> {run_output_dir}")
        print("=" * 70)

        eval_info = eval_raw_output(
            model=model,
            processor=processor,
            dataset=run_dataset,
            raw_output_csv_path=raw_output_csv_path,
            system_prompt=SYSTEM_PROMPT,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            print_every=PRINT_EVERY,
        )
        postprocess_info = postprocess(
            raw_output_csv_path=raw_output_csv_path,
            output_csv_path=output_csv_path,
            dataset_path=dataset_path,
            embedding_model_path=args.embedding_model_path,
        )
        metrics_info = metrics(
            output_csv_path=output_csv_path,
            dataset_path=dataset_path,
            run_index=run_index,
            system_prompt=SYSTEM_PROMPT,
            model_path=args.model_path,
            lora_path=args.lora_path,
            metrics_json_path=metrics_json_path,
        )
        metrics_json_paths.append(metrics_json_path)
        print(f"raw_output_csv: {eval_info['raw_output_csv']}")
        print(f"output_csv    : {postprocess_info['output_csv']}")
        print(f"metrics_json  : {metrics_json_path}")
        print(f"accuracy      : {metrics_info['accuracy']:.6f}")
        print(f"none_count    : {metrics_info['none_count']}")

    elapsed_seconds_total = time.time() - started_at
    summary_json_path = output_dir / SUMMARY_FILENAME
    summary_info = summary(
        metrics_json_paths=metrics_json_paths,
        summary_json_path=summary_json_path,
        elapsed_seconds_total=elapsed_seconds_total,
    )
    print(f"summary_json  : {summary_json_path}")
    print(f"mean_accuracy : {summary_info['mean_top1_accuracy_all_runs']:.6f}")

    vote_output_csv_path = output_dir / OUTPUT_FILENAME
    vote_info = vote(
        eval_output_dir=str(output_dir),
        output_csv_path=str(vote_output_csv_path),
    )
    print(f"vote_output_csv: {vote_info['output_csv']}")

    vote_metrics_json_path = output_dir / VOTE_METRICS_FILENAME
    vote_metrics_info = metrics(
        output_csv_path=vote_output_csv_path,
        dataset_path=dataset_path,
        run_index="vote",
        system_prompt=SYSTEM_PROMPT,
        model_path=args.model_path,
        lora_path=args.lora_path,
        metrics_json_path=vote_metrics_json_path,
    )
    print(f"vote_metrics_json: {vote_metrics_json_path}")
    print(f"vote_accuracy    : {vote_metrics_info['accuracy']:.6f}")
    print(f"\nFinished in {elapsed_seconds_total:.2f} seconds")


if __name__ == "__main__":
    main()
