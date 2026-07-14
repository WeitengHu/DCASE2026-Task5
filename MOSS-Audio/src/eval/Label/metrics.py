import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def load_output_rows(output_csv_path: str) -> List[Dict[str, str]]:
    output_csv_path = Path(output_csv_path)
    rows = []
    with open(output_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "question": row["question"],
                    "answer": row.get("answer") or "None",
                }
            )
    return rows


def load_gold_answers(dataset_path: str) -> Dict[str, str]:
    dataset_path = Path(dataset_path)
    gold_answers = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            gold_answers[item["id"]] = item["answer"]
    return gold_answers


def save_metrics_json(metrics_json_path: str, metrics_data: Dict):
    metrics_json_path = Path(metrics_json_path)
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=2)


def metrics(
    output_csv_path: str,
    dataset_path: str,
    run_index: int | str,
    system_prompt: str,
    model_path: str,
    lora_path: Optional[str],
    metrics_json_path: str,
) -> Dict:
    output_rows = load_output_rows(output_csv_path)
    gold_answers = load_gold_answers(dataset_path)

    correct_samples = 0
    none_count = 0
    missing_gold_count = 0
    for row in output_rows:
        question_id = row["question"]
        answer = row["answer"]
        if answer == "None":
            none_count += 1
        if question_id not in gold_answers:
            missing_gold_count += 1
            continue
        if answer == gold_answers[question_id]:
            correct_samples += 1

    total_samples = len(output_rows)
    metrics_data = {
        "run_index": run_index,
        "correct_samples": correct_samples,
        "accuracy": correct_samples / total_samples if total_samples else 0.0,
        "none_count": none_count,
        "total_samples": total_samples,
        "missing_prediction_count": missing_gold_count,
        "system_prompt": system_prompt,
        "model_path": model_path,
        "lora_path": lora_path,
    }
    save_metrics_json(metrics_json_path, metrics_data)
    return metrics_data
