import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_EMBEDDING_QUERY_INSTRUCTION = "Given a model response to an audio multiple-choice question, retrieve the answer choice that is semantically equivalent to the response."
OUTPUT_FILENAME = "output.csv"


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'^[\"\']+|[\"\']+$', "", text)
    return text.strip()


def clean_candidate_text(text: str) -> str:
    text = str(text).strip()
    prefixes = [
        r"^final answer\s*[:：]\s*",
        r"^answer\s*[:：]\s*",
        r"^the answer is\s*",
        r"^selected choice\s*[:：]\s*",
        r"^selected option\s*[:：]\s*",
        r"^choice\s*[:：]\s*",
        r"^option\s*[:：]\s*",
    ]
    for prefix in prefixes:
        text = re.sub(prefix, "", text, flags=re.IGNORECASE)
    return text.strip().strip('"\'“”‘’').strip()


def build_embedding_query(text: str) -> str:
    return f"Instruct: {DEFAULT_EMBEDDING_QUERY_INSTRUCTION}\nQuery:{text}"


def extract_answer_field(raw_output: str) -> str:
    if not raw_output:
        return ""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", str(raw_output), flags=re.DOTALL | re.IGNORECASE)
    # if not match:
    #     return ""
    # return clean_candidate_text(match.group(1))
    if match:
        return clean_candidate_text(match.group(1))
    match = re.search(r"</think>\s*(.*)$", str(raw_output), flags=re.DOTALL | re.IGNORECASE)
    if match:
        return clean_candidate_text(match.group(1))
    return ""

def exact_or_contain_match(answer_text: str, choices: List[str]) -> Tuple[Optional[str], Optional[int], str]:
    if not answer_text:
        return None, None, "empty_answer"

    answer_norm = normalize_text(answer_text)
    choice_norms = [normalize_text(choice) for choice in choices]
    for index, choice_norm in enumerate(choice_norms):
        if answer_norm == choice_norm:
            return choices[index], index, "exact_match"

    for index, choice_norm in enumerate(choice_norms):
        if choice_norm in answer_norm:
            return choices[index], index, f"contain_match::{answer_text}"

    return None, None, f"no_rule_match::{answer_text}"


def load_raw_outputs(raw_output_csv_path: str) -> List[Dict[str, str]]:
    raw_output_csv_path = Path(raw_output_csv_path)
    raw_outputs = []
    with open(raw_output_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_outputs.append(
                {
                    "question": row["question"],
                    "raw_output": row.get("raw_output") or "",
                }
            )
    return raw_outputs


def load_dataset_choices(dataset_path: str) -> Dict[str, List[str]]:
    dataset_path = Path(dataset_path)
    dataset_choices = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            dataset_choices[item["id"]] = (
                item["multi_choice"] if isinstance(item.get("multi_choice"), list) else []
            )
    return dataset_choices


def save_output_csv(output_csv_path: str, rows: List[Dict]):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"question": row["question"], "answer": row["answer"]})


def load_output_answers(output_csv_path: str) -> List[Dict[str, str]]:
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


def get_run_output_csv_paths(eval_output_dir: str) -> List[Path]:
    eval_output_dir = Path(eval_output_dir)
    output_csv_paths = []
    for run_index in range(5):
        suffix = "original" if run_index == 0 else "shuffle"
        output_csv_path = eval_output_dir / f"run_{run_index:02d}_{suffix}" / OUTPUT_FILENAME
        if not output_csv_path.exists():
            raise FileNotFoundError(f"Missing run output file: {output_csv_path}")
        output_csv_paths.append(output_csv_path)
    return output_csv_paths


class QwenEmbeddingMatcher:
    def __init__(self, embedding_model_path: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        # self.device = torch.device(
        #     os.environ.get(
        #         "POSTPROCESS_EMBEDDING_DEVICE",
        #         "cuda" if torch.cuda.is_available() else "cpu",
        #     )
        # )
        self.device = torch.device("cpu")
        self.batch_size = int(os.environ.get("POSTPROCESS_EMBEDDING_BATCH_SIZE", "8"))
        self.max_length = int(os.environ.get("POSTPROCESS_EMBEDDING_MAX_LENGTH", "8192"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(embedding_model_path), padding_side="left")
        self.model = AutoModel.from_pretrained(str(embedding_model_path)).to(self.device)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states, attention_mask):
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def encode(self, texts: List[str]):
        embeddings = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                batch = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                pooled = self.last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.float().detach().cpu())
        return self.torch.cat(embeddings, dim=0)

    def best_choice_from_embedding(
        self,
        query_embedding,
        choices: List[str],
        choice_embeddings: Dict[str, object],
    ) -> Tuple[str, int, float]:
        choice_matrix = self.torch.stack([choice_embeddings[choice] for choice in choices], dim=0)
        scores = choice_matrix @ query_embedding
        best_index = int(self.torch.argmax(scores).item())
        return choices[best_index], best_index, float(scores[best_index].item())


def postprocess(
    raw_output_csv_path: str,
    output_csv_path: str,
    dataset_path: str,
    embedding_model_path: str,
) -> Dict:
    output_csv_path = Path(output_csv_path)
    raw_outputs = load_raw_outputs(raw_output_csv_path)
    dataset_choices = load_dataset_choices(dataset_path)

    rows = []
    pending_embedding = []

    for row_index, item in enumerate(raw_outputs):
        question_id = item["question"]
        raw_output = item["raw_output"]
        choices = dataset_choices.get(question_id, [])
        answer_text = extract_answer_field(raw_output)

        if not choices:
            rows.append({"question": question_id, "answer": "None"})
            continue

        if answer_text:
            matched_choice, _, _ = exact_or_contain_match(answer_text, choices)
            if matched_choice is not None:
                rows.append({"question": question_id, "answer": matched_choice})
                continue
            embedding_text = answer_text
        else:
            embedding_text = raw_output

        if not str(embedding_text).strip():
            rows.append({"question": question_id, "answer": "None"})
            continue

        rows.append({"question": question_id, "answer": None})
        pending_embedding.append(
            {
                "row_index": row_index,
                "embedding_text": str(embedding_text),
                "choices": choices,
            }
        )

    if pending_embedding:
        matcher = QwenEmbeddingMatcher(embedding_model_path)
        unique_choices = sorted({choice for item in pending_embedding for choice in item["choices"]})
        choice_vectors = matcher.encode(unique_choices)
        choice_embeddings = {
            choice: choice_vectors[index]
            for index, choice in enumerate(unique_choices)
        }
        query_vectors = matcher.encode([build_embedding_query(item["embedding_text"]) for item in pending_embedding])
        for pending_index, item in enumerate(pending_embedding):
            matched_choice, _, _ = matcher.best_choice_from_embedding(
                query_vectors[pending_index],
                item["choices"],
                choice_embeddings,
            )
            rows[item["row_index"]]["answer"] = matched_choice

    save_output_csv(output_csv_path, rows)
    print(f"embedding_model_call_count: {len(pending_embedding)}")
    return {
        "output_csv": str(output_csv_path),
        "total_samples": len(raw_outputs),
        "embedding_model_call_count": len(pending_embedding),
    }


def vote(eval_output_dir: str, output_csv_path: str) -> Dict:
    eval_output_dir = Path(eval_output_dir)
    output_csv_path = Path(output_csv_path)
    output_csv_paths = get_run_output_csv_paths(str(eval_output_dir))
    per_run_rows = [load_output_answers(str(path)) for path in output_csv_paths]

    question_order = [row["question"] for row in per_run_rows[0]]
    per_run_answer_maps = [
        {row["question"]: row["answer"] for row in run_rows}
        for run_rows in per_run_rows
    ]

    voted_rows = []
    missing_vote_count = 0
    for question_id in question_order:
        answers = [
            answer_map[question_id]
            for answer_map in per_run_answer_maps
            if question_id in answer_map
        ]
        if not answers:
            voted_rows.append({"question": question_id, "answer": "None"})
            missing_vote_count += 1
            continue
        valid_answers = [answer for answer in answers if answer and answer != "None"]
        vote_pool = valid_answers if valid_answers else answers
        # vote_counts = Counter(answers)
        vote_counts = Counter(vote_pool)
        max_count = max(vote_counts.values())
        tied_answers = {answer for answer, count in vote_counts.items() if count == max_count}
        voted_answer = next(answer for answer in answers if answer in tied_answers)
        voted_rows.append({"question": question_id, "answer": voted_answer})

    save_output_csv(str(output_csv_path), voted_rows)
    return {
        "output_csv": str(output_csv_path),
        "total_samples": len(voted_rows),
        "run_output_csv_paths": [str(path) for path in output_csv_paths],
        "missing_vote_count": missing_vote_count,
    }
