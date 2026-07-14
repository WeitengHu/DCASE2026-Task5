#!/usr/bin/env python3
"""Structured-CoT reward plugin for AudioMCQ GRPO with ms-swift.

Registered reward names:
- structured_cot_accuracy
- structured_cot_format
- structured_cot_qtype
- structured_cot_sim_question_analysis
- structured_cot_sim_audio_evidence
- structured_cot_sim_reasoning
- structured_cot_length
- structured_cot_composite
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from swift.rewards import ORM, orms


ALLOWED_QUESTION_TYPES = {"sound", "speech", "music", "temporal"}
FIELDS = ("question_analysis", "audio_evidence", "reasoning")
DEFAULT_BGE_PATH = "path/to/bge-m3"
DEFAULT_QWEN_PATH = "path/to/Qwen3-Embedding-0.6B"
DEFAULT_TOKENIZER_PATH = "path/to/Qwen2.5-Omni-7B"

QWEN_FIELD_TASKS = {
    "question_analysis": (
        "Given a generated question analysis for an audio multiple-choice question, "
        "retrieve the reference question analysis with the same question intent and "
        "required audio evidence."
    ),
    "audio_evidence": (
        "Given generated audio evidence for an audio multiple-choice answer, retrieve "
        "the reference audio evidence that describes the same concrete sounds, speech, "
        "music, or temporal events."
    ),
    "reasoning": (
        "Given generated reasoning for an audio multiple-choice answer, retrieve the "
        "reference reasoning that makes the same inference from audio evidence to the "
        "final answer."
    ),
}


@dataclass(frozen=True)
class ParsedStructuredCot:
    raw: str
    question_analysis: str
    question_type: str
    audio_evidence: str
    reasoning: str
    answer: str
    strict_format: bool

    def field(self, name: str) -> str:
        return getattr(self, name)


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _as_text(value.get("content", ""))
    if isinstance(value, (list, tuple)) and value:
        return _as_text(value[-1])
    return str(value)


def _as_list(value, n: int, default=None) -> List:
    if value is None:
        return [default for _ in range(n)]
    if isinstance(value, list):
        if len(value) == n:
            return value
        if len(value) == 1:
            return value * n
    return [value for _ in range(n)]


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


_TOKENIZER = None
_TOKENIZER_PATH = None


def _resolve_tokenizer_path(reward_args=None) -> str:
    tokenizer_path = os.environ.get("STRUCTURED_COT_REWARD_TOKENIZER_PATH")
    if tokenizer_path:
        return tokenizer_path

    for attr in ("model", "model_path", "model_name_or_path"):
        value = getattr(reward_args, attr, None)
        if value:
            return str(value)
    return DEFAULT_TOKENIZER_PATH


def _get_reward_tokenizer(reward_args=None):
    global _TOKENIZER, _TOKENIZER_PATH
    tokenizer_path = _resolve_tokenizer_path(reward_args)
    if _TOKENIZER is None or _TOKENIZER_PATH != tokenizer_path:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        _TOKENIZER_PATH = tokenizer_path
    return _TOKENIZER


def _token_count(text: str, reward_args=None) -> int:
    tokenizer = _get_reward_tokenizer(reward_args)
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def parse_structured_cot(text) -> ParsedStructuredCot:
    raw = _as_text(text)
    strict_pattern = re.compile(
        r"^\s*<think>\s*"
        r"<question_analysis>(?P<question_analysis>.+?)</question_analysis>\s*"
        r"<question_type>(?P<question_type>.+?)</question_type>\s*"
        r"<audio_evidence>(?P<audio_evidence>.+?)</audio_evidence>\s*"
        r"<reasoning>(?P<reasoning>.+?)</reasoning>\s*"
        r"</think>\s*"
        r"<answer>(?P<answer>.+?)</answer>\s*$",
        flags=re.DOTALL,
    )
    match = strict_pattern.match(raw)
    if match:
        groups = {key: value.strip() for key, value in match.groupdict().items()}
        return ParsedStructuredCot(raw=raw, strict_format=True, **groups)
    return ParsedStructuredCot(
        raw=raw,
        question_analysis=_extract_tag(raw, "question_analysis"),
        question_type=_extract_tag(raw, "question_type"),
        audio_evidence=_extract_tag(raw, "audio_evidence"),
        reasoning=_extract_tag(raw, "reasoning"),
        answer=_extract_tag(raw, "answer"),
        strict_format=False,
    )


def _reference_answer(solution_text, answer_value=None) -> str:
    answer = _as_text(answer_value).strip()
    if answer:
        return answer
    return parse_structured_cot(solution_text).answer


def _reference_question_type(solution_text, question_type_value=None) -> str:
    question_type = _as_text(question_type_value).strip().lower()
    if question_type:
        return question_type
    return parse_structured_cot(solution_text).question_type.strip().lower()


def accuracy_reward(completions, solution=None, answer=None, **kwargs) -> List[float]:
    del kwargs
    n = len(completions)
    solutions = _as_list(solution, n, "")
    answers = _as_list(answer, n, None)
    rewards = []
    for completion, sol, ans in zip(completions, solutions, answers):
        pred = parse_structured_cot(completion).answer
        ref = _reference_answer(sol, ans)
        rewards.append(1.0 if pred and _normalize_text(pred) == _normalize_text(ref) else 0.0)
    return rewards


def format_reward(completions, solution=None, **kwargs) -> List[float]:
    del solution, kwargs
    return [1.0 if parse_structured_cot(completion).strict_format else 0.0 for completion in completions]


def qtype_reward(completions, solution=None, question_type=None, **kwargs) -> List[float]:
    del kwargs
    n = len(completions)
    solutions = _as_list(solution, n, "")
    question_types = _as_list(question_type, n, None)
    rewards = []
    for completion, sol, ref_qtype_value in zip(completions, solutions, question_types):
        pred_qtype = parse_structured_cot(completion).question_type.strip().lower()
        ref_qtype = _reference_question_type(sol, ref_qtype_value)
        rewards.append(
            1.0
            if pred_qtype in ALLOWED_QUESTION_TYPES and pred_qtype == ref_qtype
            else 0.0
        )
    return rewards


def length_reward(completions, solution=None, answer=None, reward_args=None, **kwargs) -> List[float]:
    fmt = format_reward(completions, solution=solution, **kwargs)
    acc = accuracy_reward(completions, solution=solution, answer=answer, **kwargs)
    min_tokens = int(os.environ.get("STRUCTURED_COT_REWARD_LEN_MIN_TOKENS",100))
    max_tokens = int(os.environ.get("STRUCTURED_COT_REWARD_LEN_MAX_TOKENS",500))
    rewards = []
    for completion, fmt_value, acc_value in zip(completions, fmt, acc):
        if acc_value != 1.0 or fmt_value != 1.0:
            rewards.append(0.0)
            continue
        count = _token_count(_as_text(completion), reward_args=reward_args)
        rewards.append(
            1.0
            if min_tokens <= count <= max_tokens
            else 0.0
        )
    return rewards


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _resolve_reward_device(device_setting: str | None = None) -> str:
    device = (device_setting or os.environ.get("STRUCTURED_COT_REWARD_DEVICE", "cpu")).strip()
    if device.lower() != "auto":
        return device

    local_rank = os.environ.get("LOCAL_RANK", "0").strip()
    try:
        local_rank_int = int(local_rank)
    except ValueError as exc:
        raise ValueError(f"LOCAL_RANK must be an integer when STRUCTURED_COT_REWARD_DEVICE=auto, got: {local_rank!r}") from exc
    if local_rank_int < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative when STRUCTURED_COT_REWARD_DEVICE=auto, got: {local_rank!r}")
    print(f"Resolved reward device to cuda:{local_rank_int} based on LOCAL_RANK={local_rank}")
    return f"cuda:{local_rank_int}"


class _EmbeddingScorer:
    def __init__(self) -> None:
        self.backend = os.environ.get("STRUCTURED_COT_REWARD_EMBEDDING_BACKEND", "bge").strip().lower()
        self.device = _resolve_reward_device()
        self.batch_size = int(os.environ.get("STRUCTURED_COT_REWARD_BATCH_SIZE", "1"))
        self.max_length = int(os.environ.get("STRUCTURED_COT_REWARD_MAX_LENGTH", "1024"))
        self._reference_cache: Dict[tuple, np.ndarray] = {}
        self._model = None
        self._tokenizer = None
        self._torch = None
        print(f"Embedding scorer {self.backend} initialized with batch size {self.batch_size} and max length {self.max_length}")
    def _load_bge(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            model_path = os.environ.get("STRUCTURED_COT_REWARD_BGE_PATH", DEFAULT_BGE_PATH)
            use_fp16 = os.environ.get("STRUCTURED_COT_REWARD_BGE_USE_FP16", "false").lower()
            self._model = BGEM3FlagModel(
                model_path,
                use_fp16=use_fp16 in {"1", "true", "yes", "y"},
                devices=self.device,
            )
        return self._model

    def _load_qwen(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_path = os.environ.get("STRUCTURED_COT_REWARD_QWEN_PATH", DEFAULT_QWEN_PATH)
            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
            self._model = AutoModel.from_pretrained(model_path).to(torch.device(self.device))
            self._model.eval()
        return self._model

    @staticmethod
    def _qwen_instruct(field: str, text: str) -> str:
        return f"Instruct: {QWEN_FIELD_TASKS[field]}\nQuery:{text}"

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
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

    def _encode_bge(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_bge()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
        )["dense_vecs"]
        return _normalize_embeddings(np.asarray(embeddings, dtype=np.float32))

    def _encode_qwen(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_qwen()
        embeddings = []
        with self._torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = list(texts[start : start + self.batch_size])
                batch = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(model.device) for key, value in batch.items()}
                outputs = model(**batch)
                pooled = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
                pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.float().detach().cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def _encode(self, texts: Sequence[str], field: str, side: str) -> np.ndarray:
        if self.backend == "none":
            return np.zeros((len(texts), 1), dtype=np.float32)
        if self.backend == "bge":
            return self._encode_bge(texts)
        if self.backend == "qwen":
            qwen_texts = [
                self._qwen_instruct(field, text) if side == "generated" else text
                for text in texts
            ]
            return self._encode_qwen(qwen_texts)
        raise ValueError(
            "STRUCTURED_COT_REWARD_EMBEDDING_BACKEND must be one of: bge, qwen, none"
        )

    def _encode_references(self, texts: Sequence[str], field: str) -> np.ndarray:
        missing_texts = []
        missing_keys = []
        for text in texts:
            key = (self.backend, field, text)
            if key not in self._reference_cache:
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            encoded = self._encode(missing_texts, field=field, side="reference")
            for key, vector in zip(missing_keys, encoded):
                self._reference_cache[key] = vector
        return np.stack([self._reference_cache[(self.backend, field, text)] for text in texts])

    def similarities(self, field: str, generated_texts: Sequence[str], reference_texts: Sequence[str]) -> List[float]:
        if not generated_texts:
            return []
        generated = self._encode(generated_texts, field=field, side="generated")
        references = self._encode_references(reference_texts, field=field)
        scores = np.sum(generated * references, axis=1)
        return [float(np.clip(score, 0.0, 1.0)) for score in scores]


_EMBEDDING_SCORER: _EmbeddingScorer | None = None


def _get_embedding_scorer() -> _EmbeddingScorer:
    global _EMBEDDING_SCORER
    if _EMBEDDING_SCORER is None:
        _EMBEDDING_SCORER = _EmbeddingScorer()
    return _EMBEDDING_SCORER


def _similarity_reward(field: str, gate: Sequence[float], completions, solution=None) -> List[float]:
    n = len(completions)
    solutions = _as_list(solution, n, "")
    rewards = [0.0 for _ in range(n)]
    active_indices = []
    generated_texts = []
    reference_texts = []
    for index, (completion, sol, gate_value) in enumerate(zip(completions, solutions, gate)):
        if gate_value != 1.0:
            continue
        pred = parse_structured_cot(completion)
        ref = parse_structured_cot(sol)
        generated = pred.field(field).strip()
        reference = ref.field(field).strip()
        if not generated or not reference:
            rewards[index] = 0.0
            continue
        active_indices.append(index)
        generated_texts.append(generated)
        reference_texts.append(reference)
    scores = _get_embedding_scorer().similarities(field, generated_texts, reference_texts)
    for index, score in zip(active_indices, scores):
        rewards[index] = score
    return rewards


def sim_question_analysis_reward(completions, solution=None, question_type=None, **kwargs) -> List[float]:
    gate = qtype_reward(completions, solution=solution, question_type=question_type, **kwargs)
    return _similarity_reward("question_analysis", gate, completions, solution=solution)


def sim_audio_evidence_reward(completions, solution=None, answer=None, **kwargs) -> List[float]:
    gate = accuracy_reward(completions, solution=solution, answer=answer, **kwargs)
    return _similarity_reward("audio_evidence", gate, completions, solution=solution)


def sim_reasoning_reward(completions, solution=None, answer=None, **kwargs) -> List[float]:
    gate = accuracy_reward(completions, solution=solution, answer=answer, **kwargs)
    return _similarity_reward("reasoning", gate, completions, solution=solution)


def _composite_weights() -> List[float]:
    text = os.environ.get("STRUCTURED_COT_REWARD_WEIGHTS", "2.0,0.5,0.5,0.25,0.5,0.5,0.25")
    return [float(part.strip()) for part in text.replace(" ", ",").split(",") if part.strip()]


def composite_reward(completions, solution=None, answer=None, question_type=None, reward_args=None, **kwargs) -> List[float]:
    components = [
        accuracy_reward(completions, solution=solution, answer=answer, **kwargs),
        format_reward(completions, solution=solution, **kwargs),
        qtype_reward(completions, solution=solution, question_type=question_type, **kwargs),
        sim_question_analysis_reward(completions, solution=solution, question_type=question_type, **kwargs),
        sim_audio_evidence_reward(completions, solution=solution, answer=answer, **kwargs),
        sim_reasoning_reward(completions, solution=solution, answer=answer, **kwargs),
        length_reward(completions, solution=solution, answer=answer, reward_args=reward_args, **kwargs),
    ]
    weights = _composite_weights()
    if len(weights) != len(components):
        raise ValueError("STRUCTURED_COT_REWARD_WEIGHTS must contain 7 values")
    rewards = []
    for values in zip(*components):
        rewards.append(float(sum(weight * value for weight, value in zip(weights, values))))
    return rewards


class StructuredCotAccuracyORM(ORM):
    def __call__(self, completions, solution=None, answer=None, **kwargs):
        return accuracy_reward(completions, solution=solution, answer=answer, **kwargs)


class StructuredCotFormatORM(ORM):
    def __call__(self, completions, solution=None, **kwargs):
        return format_reward(completions, solution=solution, **kwargs)


class StructuredCotQTypeORM(ORM):
    def __call__(self, completions, solution=None, question_type=None, **kwargs):
        return qtype_reward(completions, solution=solution, question_type=question_type, **kwargs)


class StructuredCotQuestionAnalysisSimORM(ORM):
    def __call__(self, completions, solution=None, question_type=None, **kwargs):
        return sim_question_analysis_reward(
            completions,
            solution=solution,
            question_type=question_type,
            **kwargs,
        )


class StructuredCotAudioEvidenceSimORM(ORM):
    def __call__(self, completions, solution=None, answer=None, **kwargs):
        return sim_audio_evidence_reward(completions, solution=solution, answer=answer, **kwargs)


class StructuredCotReasoningSimORM(ORM):
    def __call__(self, completions, solution=None, answer=None, **kwargs):
        return sim_reasoning_reward(completions, solution=solution, answer=answer, **kwargs)


class StructuredCotLengthORM(ORM):
    def __call__(self, completions, solution=None, answer=None, **kwargs):
        return length_reward(completions, solution=solution, answer=answer, reward_args=self.args, **kwargs)


class StructuredCotCompositeORM(ORM):
    def __call__(self, completions, solution=None, answer=None, question_type=None, **kwargs):
        return composite_reward(
            completions,
            solution=solution,
            answer=answer,
            question_type=question_type,
            reward_args=self.args,
            **kwargs,
        )


orms["structured_cot_accuracy"] = StructuredCotAccuracyORM
orms["structured_cot_format"] = StructuredCotFormatORM
orms["structured_cot_qtype"] = StructuredCotQTypeORM
orms["structured_cot_sim_question_analysis"] = StructuredCotQuestionAnalysisSimORM
orms["structured_cot_sim_audio_evidence"] = StructuredCotAudioEvidenceSimORM
orms["structured_cot_sim_reasoning"] = StructuredCotReasoningSimORM
orms["structured_cot_length"] = StructuredCotLengthORM
orms["structured_cot_composite"] = StructuredCotCompositeORM

print("Structured CoT reward functions registered successfully.")
