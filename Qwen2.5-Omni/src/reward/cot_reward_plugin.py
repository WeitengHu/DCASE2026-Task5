#!/usr/bin/env python3
"""Reward functions for AudioMCQ GRPO with ms-swift."""

import re
from typing import List

from swift.rewards import ORM, orms

def print_comparison_compact(completions, solution):
    print("\n📊 Completion vs Solution Comparison:")
    print("-" * 60)
    
    max_len = max(len(completions), len(solution))
    
    for i in range(max_len):
        comp = completions[i] if i < len(completions) else "N/A"
        sol = solution[i] if i < len(solution) else "N/A"
        
        print(f"{i:2d}. '{comp}' vs '{sol}'")
    
    print("-" * 60)

def _extract_answer(text: str) -> str:
    text = "" if text is None else str(text)
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    answer = match.group(1) if match else text
    return re.sub(r"\s+", " ", answer).strip().casefold()


def audio_choice_accuracy_reward(completions, solution, **kwargs) -> List[float]:
    # print_comparison_compact(completions, solution)
    rewards = []
    for content, sol in zip(completions, solution):
        try:
            rewards.append(1.0 if _extract_answer(content) == _extract_answer(sol) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def external_format_reward(completions, solution=None, **kwargs) -> List[float]:
    pattern = re.compile(r"^\s*<answer>.+?</answer>\s*$", re.DOTALL | re.IGNORECASE)
    return [1.0 if pattern.match("" if content is None else str(content)) else 0.0 for content in completions]


def external_cot_format_reward(completions, solution=None, **kwargs) -> List[float]:
    pattern = re.compile(
        r"^\s*<think>.*?</think>\s*<answer>.+?</answer>\s*$",
        re.DOTALL | re.IGNORECASE,
    )
    return [1.0 if pattern.match("" if content is None else str(content)) else 0.0 for content in completions]


class AudioChoiceAccuracyORM(ORM):
    def __call__(self, completions, solution, **kwargs):
        return audio_choice_accuracy_reward(completions, solution, **kwargs)


class ExternalFormatORM(ORM):
    def __call__(self, completions, solution=None, **kwargs):
        return external_format_reward(completions, solution, **kwargs)


class ExternalCotFormatORM(ORM):
    def __call__(self, completions, solution=None, **kwargs):
        return external_cot_format_reward(completions, solution, **kwargs)


orms["external_audio_choice_accuracy"] = AudioChoiceAccuracyORM
orms["external_format"] = ExternalFormatORM
orms["external_cot_format"] = ExternalCotFormatORM
print("Audio choice accuracy reward function registered successfully!")
print("External format reward function registered successfully!")
print("External CoT format reward function registered successfully!")