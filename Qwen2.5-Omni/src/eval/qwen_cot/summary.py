import json
import math
from pathlib import Path
from typing import Dict, List


def load_metrics(metrics_json_path: str) -> Dict:
    metrics_json_path = Path(metrics_json_path)
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean_and_std(values: List[float]):
    if not values:
        return 0.0, 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return mean_value, math.sqrt(variance)


def save_summary_json(summary_json_path: str, summary_data: Dict):
    summary_json_path = Path(summary_json_path)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)


def summary(
    metrics_json_paths: List[str],
    summary_json_path: str,
    elapsed_seconds_total: float,
) -> Dict:
    per_run_metrics = [load_metrics(path) for path in metrics_json_paths]
    per_run_metrics = sorted(per_run_metrics, key=lambda item: item["run_index"])
    run_top1_accuracies = [item["accuracy"] for item in per_run_metrics]
    shuffle_only_top1_accuracies = [
        item["accuracy"] for item in per_run_metrics if item["run_index"] != 0
    ]
    mean_all, std_all = mean_and_std(run_top1_accuracies)
    mean_shuffle, std_shuffle = mean_and_std(shuffle_only_top1_accuracies)

    first_metrics = per_run_metrics[0] if per_run_metrics else {}
    summary_data = {
        "system_prompt": first_metrics.get("system_prompt"),
        "model_path": first_metrics.get("model_path"),
        "lora_path": first_metrics.get("lora_path"),
        "run_top1_accuracies": run_top1_accuracies,
        "original_order_top1_accuracy": run_top1_accuracies[0] if run_top1_accuracies else None,
        "shuffle_only_top1_accuracies": shuffle_only_top1_accuracies,
        "mean_top1_accuracy_all_runs": mean_all,
        "std_top1_accuracy_all_runs": std_all,
        "mean_top1_accuracy_shuffle_only": mean_shuffle,
        "std_top1_accuracy_shuffle_only": std_shuffle,
        "elapsed_seconds_total": elapsed_seconds_total,
    }
    save_summary_json(summary_json_path, summary_data)
    return summary_data
