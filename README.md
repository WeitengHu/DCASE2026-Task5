# Reasoning-Oriented Post-Training and Inference-Time LoRA Rescaling for Audio-Dependent Question Answering

[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-DCASE2026--Task5--Training--Jsonl-yellow.svg)](https://huggingface.co/datasets/huweiteng/DCASE2026-Task5-Training-Jsonl)

This repository contains the task-specific training, reward, inference, post-processing, and evaluation code used to study Audio-Dependent Question Answering (ADQA) for [DCASE 2026 Challenge Task 5](https://dcase.community/challenge2026/task-audio-dependent-question-answering). The experiments cover two Qwen2.5-Omni post-training pipelines and MOSS-Audio-8B-Thinking systems.

Clone this repository together with its pinned MOSS-Audio dependency:

```bash
git clone --recurse-submodules git@github.com:WeitengHu/DCASE2026-Task5.git
cd DCASE2026-Task5
```

Both the main repository and the MOSS-Audio submodule use SSH URLs. Configure a GitHub SSH key before cloning.

If the repository was cloned without `--recurse-submodules`, initialize the dependency before using the MOSS pipeline:

```bash
git submodule update --init --recursive
```

## DCASE 2026 Task 5

DCASE 2026 Task 5 evaluates whether a Large Audio-Language Model actually uses the input audio rather than solving a question from textual shortcuts or prior knowledge. Each ADQA sample contains an audio recording, a multiple-choice question, and a set of candidate answers. The primary metric is Top-1 Accuracy.

The official resources are:

- [Task description and rules](https://dcase.community/challenge2026/task-audio-dependent-question-answering)
- [AudioMCQ-StrongAC-GeminiCoT training set](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT)
- [DCASE2026-Task5-DevSet](https://huggingface.co/datasets/Harland/DCASE2026-Task5-DevSet)
- [ADQA-Bench evaluation set](https://huggingface.co/datasets/Harland/ADQA-Bench)

The pipeline-ready training JSONL files produced for this project are released separately at [huweiteng/DCASE2026-Task5-Training-Jsonl](https://huggingface.co/datasets/huweiteng/DCASE2026-Task5-Training-Jsonl). This is a project-specific derived dataset and is not an official DCASE 2026 Task 5 resource.

The official training set contains 19,480 strongly audio-dependent questions with Gemini-generated Chain-of-Thought annotations. This repository uses those annotations to construct answer-only, free-form CoT, and structured-CoT supervision.

The pipeline-ready training annotations are distributed separately on Hugging Face. They do not include the audio files. See [Data Preparation](#data-preparation) before running any training script, and replace every `path/to/...` placeholder in the launchers.

## Qwen-CoT

Qwen-CoT uses two training stages:

1. **Answer-only SFT** adapts Qwen2.5-Omni-7B to the ADQA instruction and final-answer format.
2. **GRPO** starts from the answer-only SFT adapter and uses a CoT prompt. Its reward is the weighted sum of answer accuracy and `<think>...</think><answer>...</answer>` format compliance.

The released GRPO configuration uses reward weights `2.0` and `0.5` for accuracy and format, respectively.

## Qwen-Structured-CoT

The structured pipeline represents a response as:

```text
<think>
<question_analysis>...</question_analysis>
<question_type>...</question_type>
<audio_evidence>...</audio_evidence>
<reasoning>...</reasoning>
</think>
<answer>...</answer>
```

It also uses two stages:

1. **Structured-CoT SFT** supervises the complete structured response.
2. **GDPO** applies separately normalized rewards for answer accuracy, format, question type, semantic similarity of the three reasoning fields, and valid response length.

The field-similarity rewards use Qwen3-Embedding-0.6B. Similarity and length rewards are gated by answer correctness, while question-analysis similarity is gated by the question-type reward.

## MOSS-Audio-8B-Thinking

MOSS-Audio is evaluated in two prompt configurations:

- **MOSS-Thinking-Full** preserves natural-language reasoning and asks for the complete answer text.
- **MOSS-Thinking-Label** constrains the final answer to an option label to reduce answer-mapping ambiguity.

The repository supports both native zero-shot inference and task-specific LoRA SFT using the model's native `<think>...</think>` response format.

## Results Reported in the Paper

### Post-training on the development set

| System | Prompt-matched baseline | SFT | RL | Final change |
| --- | ---: | ---: | ---: | ---: |
| Qwen-CoT | 55.38 | 56.50 | 58.93 | +3.55 |
| Qwen-Structured-CoT | 54.39 | 57.93 | 58.93 | +4.54 |
| MOSS-Thinking-Full | 66.02 | 58.18 | - | -7.84 |
| MOSS-Thinking-Label | 67.70 | 60.36 | - | -7.34 |

### Best observed inference-time LoRA scaling

The training configuration uses LoRA rank `r = 8` and `lora_alpha = 32`, giving a scaling factor of `gamma = alpha / r = 4`.

| System | Adapter disabled | Training scale (`gamma=4`) | Best nonzero `gamma` | Best accuracy |
| --- | ---: | ---: | ---: | ---: |
| Qwen-CoT | 55.38 | 58.93 | 2.0 | **61.05** |
| Qwen-Structured-CoT | 54.39 | **58.93** | 4.0 | **58.93** |
| MOSS-Thinking-Full | 66.02 | 58.18 | 0.5 | **67.02** |
| MOSS-Thinking-Label | **67.70** | 60.36 | 1.0 | 67.52 |

The official Task 5 leaderboard lists our best system with **57.03% evaluation accuracy**, ranked third overall. Development-set results above are reported for analysis and should not be interpreted as evaluation-set scores.

## Repository Structure

```text
.
|-- MOSS-Audio/
|   |-- scripts/
|   |   |-- train/finetune.sh
|   |   `-- eval/eval_moss.sh
|   `-- src/
|       `-- eval/
|           |-- Full/
|           `-- Label/
|-- Qwen2.5-Omni/
|   |-- scripts/
|   |   |-- train/
|   |   |   |-- sft_answer_only.sh
|   |   |   |-- grpo_cot.sh
|   |   |   |-- sft_structured_cot.sh
|   |   |   `-- gdpo_structured_cot.sh
|   |   `-- eval/eval_qwen.sh
|   `-- src/
|       |-- reward/
|       |   |-- cot_reward_plugin.py
|       |   `-- structured_cot_reward_plugin.py
|       `-- eval/
|           |-- qwen_answer_only/
|           |-- qwen_cot/
|           `-- qwen_structured_cot/
|-- third_party/
|   `-- MOSS-Audio/              # pinned OpenMOSS/MOSS-Audio submodule
|-- .gitmodules
|-- LICENSE
`-- README.md
```

`MOSS-Audio/` contains only this project's Task 5 launchers and evaluators. The official model implementation and fine-tuning entry point are pinned under `third_party/MOSS-Audio/`; do not copy or move these files into one another.

## Environment

### Separate Conda environments

> [!IMPORTANT]
> Use two independent Conda environments for the Qwen2.5-Omni and MOSS-Audio pipelines. Their model stacks and pinned dependencies may conflict, so do not install both pipelines into the same environment.

Create both environments with Python 3.12:

```bash
conda create -n moss-audio python=3.12
conda create -n qwen python=3.12
```

Activate `qwen` only when running the Qwen training or evaluation pipeline, and activate `moss-audio` only when running the MOSS training or evaluation pipeline.

### Environment setup

**For qwen**

```bash
conda activate qwen
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchcodec==0.10.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
pip install uv
pip install uv
pip install 'ms-swift'
pip install 'ms-swift[megatron]'
pip install 'ms-swift[eval]' -U
pip install qwen-omni-utils -U
pip install --no-cache-dir --no-build-isolation flash-attn
uv pip install vllm==0.19.1 --torch-backend=auto --no-cache-dir
```

**For moss-audio**

```bash
conda activate moss-audio
conda install -c conda-forge "ffmpeg=7" -y
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e "./third_party/MOSS-Audio[torch-runtime]"
pip install librosa peft
```

The official [OpenMOSS/MOSS-Audio](https://github.com/OpenMOSS/MOSS-Audio) repository is included as an SSH Git submodule at `third_party/MOSS-Audio/`. This keeps the upstream implementation separate from the Task 5 code while pinning the exact source revision used by the launchers.

If you cannot initialize the submodule, you can git clone from the official repository and put it into the third_party dictionary.

### Model download
Download or otherwise make available:

- [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- Optionally, [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) for the alternative structured reward backend
- [MOSS-Audio-8B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Thinking)

## Data Preparation

> [!IMPORTANT]
> The released JSONL files contain annotations only and do **not** include the audio. Download the original audio separately from [Harland/AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT), then replace every JSONL audio path with the corresponding absolute path on your machine before training.

1. Download the pipeline-ready annotations from [huweiteng/DCASE2026-Task5-Training-Jsonl](https://huggingface.co/datasets/huweiteng/DCASE2026-Task5-Training-Jsonl).
2. Download the original audio from [Harland/AudioMCQ-StrongAC-GeminiCoT](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT).
3. Replace the relative audio prefix in every JSONL file with the absolute path to the downloaded audio directory.

For example, change:

```text
AudioMCQ-StrongAC-GeminiCoT/AudioCaps/9HUZjSJnUAA_30.wav
```

to a real local path such as:

```text
/absolute/path/to/AudioMCQ-StrongAC-GeminiCoT/AudioCaps/9HUZjSJnUAA_30.wav
```

For Qwen files, update the paths in the `audios` list. For MOSS-Audio files, update the `content` field of each `conversation` entry whose `message_type` is `audio`. Only change the audio paths; keep all other fields unchanged. Verify that every resulting absolute path points to an existing audio file.

The released files are organized as follows:

| Pipeline | Released JSONL file(s) | Representation |
| --- | --- | --- |
| Qwen answer-only SFT | `qwen/train/qwen_cot_answer_only_sft.jsonl` | ms-swift multimodal conversations with an answer-only assistant target |
| Qwen-CoT GRPO | `qwen/train/qwen_cot_grpo.jsonl` | ms-swift GRPO examples with the multimodal prompt and reference `solution` |
| Qwen structured-CoT SFT | `qwen/train/qwen_structured_cot_sft.jsonl` | ms-swift multimodal conversations with structured-CoT assistant targets |
| Qwen structured-CoT GDPO | `qwen/train/qwen_structured_cot_gdpo.jsonl` | ms-swift GRPO examples with the structured reference fields used by the rewards |
| MOSS-Audio Full SFT | `moss-audio/train/moss_full_sft_train.jsonl` and `moss-audio/eval/moss_full_sft_eval.jsonl` | MOSS-Audio native `conversation` JSONL with full reasoning targets |
| MOSS-Audio Label SFT | `moss-audio/train/moss_label_sft_train.jsonl` and `moss-audio/eval/moss_label_sft_eval.jsonl` | MOSS-Audio native `conversation` JSONL with label targets |

The official development and evaluation datasets remain separate downloads.

### Evaluation JSONL

All evaluation configurations expect one JSON object per line with this schema:

```json
{
  "id": "dev_0001",
  "audio_path": "audio/example.wav",
  "question_text": "Which sound occurs after the speech?",
  "multi_choice": ["Rain", "A door closing", "Music", "Applause"],
  "answer": "A door closing"
}
```

`audio_path` may be absolute. If it is relative, the evaluator resolves it as `dataset_root / audio_path`.

### MOSS-Audio SFT JSONL

MOSS training expects a `conversation` list. A minimal sample is:

```json
{"conversation": [
  {"role": "user", "message_type": "audio", "content": "path/to/audio.wav"},
  {"role": "user", "message_type": "text", "content": "Question and choices"},
  {"role": "assistant", "message_type": "text", "content": "<think>\nReasoning\n</think>\n\nFinal answer"}
]}
```

See the pinned upstream [MOSS-Audio fine-tuning guide](https://github.com/OpenMOSS/MOSS-Audio/blob/5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883/finetune/FINETUNE.md) for the complete format and arguments.

## Configure the Paths

Before running anything, locate all placeholders.

At minimum, configure:

1. Base model directories.
2. Training and evaluation JSONL files.
3. Dataset roots containing the audio files.
4. SFT adapter checkpoints used to initialize GRPO or GDPO.
5. Qwen3-Embedding-0.6B paths used by structured rewards and post-processing.
6. Output directories.

## Training

Run every command from the repository root.

### Qwen-CoT: answer-only SFT followed by GRPO

The answer-only SFT script uses the full dataset path placeholder and the paper's one-epoch setting, learning rate, batch size, gradient accumulation, LoRA rank, and LoRA alpha:

```bash
bash Qwen2.5-Omni/scripts/train/sft_answer_only.sh
```

After SFT, set both `--adapters` and `--ref_adapters` in `grpo_cot.sh` to the answer-only SFT checkpoint. GRPO starts as a new training run initialized from the SFT adapter.

```bash
bash Qwen2.5-Omni/scripts/train/grpo_cot.sh
```

### Qwen-Structured-CoT: structured SFT followed by GDPO

```bash
bash Qwen2.5-Omni/scripts/train/sft_structured_cot.sh
```

Then set the following paths in `gdpo_structured_cot.sh`:

- `--adapters` and `--ref_adapters`: structured-CoT SFT checkpoint
- `--dataset`: structured-CoT GRPO JSONL
- `STRUCTURED_COT_REWARD_QWEN_PATH`: Qwen3-Embedding-0.6B
- `STRUCTURED_COT_REWARD_BGE_PATH`: BGE-M3, if using the BGE backend

Run GDPO:

```bash
bash Qwen2.5-Omni/scripts/train/gdpo_structured_cot.sh
```

The plugin registers seven field-wise rewards, and `--scale_rewards gdpo` applies reward-decoupled normalization before weighted aggregation.

### MOSS-Audio LoRA SFT

MOSS-Audio uses a single baseline SFT stage. `--model_dir` points directly to the MOSS-Audio-8B-Thinking base model.

Update the model, train, validation, and output paths in `MOSS-Audio/scripts/train/finetune.sh`, then run:

```bash
bash MOSS-Audio/scripts/train/finetune.sh
```

The launcher can be called from any working directory: it resolves the repository root, checks `third_party/MOSS-Audio/`, sets `PYTHONPATH`, and then invokes the pinned upstream trainer.

## Evaluation

The convenience scripts contain placeholders for all supported evaluation configurations:

```bash
bash Qwen2.5-Omni/scripts/eval/eval_qwen.sh
bash MOSS-Audio/scripts/eval/eval_moss.sh
```

Edit each command separately so that its LoRA checkpoint and output directory correspond to the selected pipeline. You can also call an evaluator directly.

The Qwen evaluators use greedy decoding (`do_sample=False`). Both MOSS evaluators use sampling with `temperature=1.0`, `top_p=1.0`, and `top_k=50`.

The MOSS convenience launcher performs the submodule check and `PYTHONPATH` setup automatically. If you invoke either MOSS Python evaluator directly, first expose the pinned upstream package in the current shell:

```bash
export PYTHONPATH="$PWD/third_party/MOSS-Audio:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### Qwen-CoT

```bash
python Qwen2.5-Omni/src/eval/qwen_cot/main.py \
  --model_path "path/to/Qwen2.5-Omni-7B" \
  --embedding_model_path "path/to/Qwen3-Embedding-0.6B" \
  --lora_path "path/to/qwen_cot_lora" \
  --dataset_path "path/to/dev.jsonl" \
  --dataset_root "path/to/dev_dataset_root" \
  --output_dir "path/to/output/qwen_cot"
```

### Qwen-Structured-CoT

```bash
python Qwen2.5-Omni/src/eval/qwen_structured_cot/main.py \
  --model_path "path/to/Qwen2.5-Omni-7B" \
  --embedding_model_path "path/to/Qwen3-Embedding-0.6B" \
  --lora_path "path/to/qwen_structured_cot_lora" \
  --dataset_path "path/to/dev.jsonl" \
  --dataset_root "path/to/dev_dataset_root" \
  --output_dir "path/to/output/qwen_structured_cot"
```

### MOSS-Thinking-Full

```bash
python MOSS-Audio/src/eval/Full/main.py \
  --model_path "path/to/MOSS-Audio-8B-Thinking" \
  --embedding_model_path "path/to/Qwen3-Embedding-0.6B" \
  --lora_path "path/to/moss_lora" \
  --dataset_path "path/to/dev.jsonl" \
  --dataset_root "path/to/dev_dataset_root" \
  --output_dir "path/to/output/moss_full"
```

### MOSS-Thinking-Label

```bash
python MOSS-Audio/src/eval/Label/main.py \
  --model_path "path/to/MOSS-Audio-8B-Thinking" \
  --embedding_model_path "path/to/Qwen3-Embedding-0.6B" \
  --lora_path "path/to/moss_lora" \
  --dataset_path "path/to/dev.jsonl" \
  --dataset_root "path/to/dev_dataset_root" \
  --output_dir "path/to/output/moss_label"
```

To evaluate a prompt-matched base model, omit `--lora_path`.

## Five-Run Voting and Post-Processing

Each evaluator performs five choice-order runs:

1. One run with the original choice order.
2. Four runs with independently shuffled choices.

The choice permutations are reproducible from `BASE_SEED=42`. Qwen generation is greedy, while MOSS generation remains stochastic because sampling is enabled.

For every run, the code:

1. Saves the raw model response.
2. Extracts the final answer.
3. Maps the answer with pipeline-specific rules: exact or containment matching for response-text pipelines, and option-label extraction for MOSS-Thinking-Label.
4. Uses Qwen3-Embedding-0.6B as a fallback when the rule-based mapping fails.
5. Computes Top-1 Accuracy when ground-truth answers are available.

The final answer is selected by majority vote over the five runs. A typical output directory is:

```text
output_dir/
|-- run_00_original/
|   |-- raw_output.csv
|   |-- output.csv
|   `-- metrics.json
|-- run_01_shuffle/
|-- run_02_shuffle/
|-- run_03_shuffle/
|-- run_04_shuffle/
|-- output.csv
|-- summary.json
`-- vote_metrics.json
```

The root-level `output.csv` is the majority-voted prediction file. For an unlabeled evaluation set, use the generated predictions without relying on the metric files.

## Inference-Time LoRA Rescaling

For standard LoRA, the effective scaling factor is:

```text
gamma = lora_alpha / r
```

The trained adapters use `r = 8` and `lora_alpha = 32`, so the default scale is `gamma = 4`. To reproduce a scaling sweep without additional training:

1. Copy the adapter checkpoint to a new directory.
2. Keep the adapter weights unchanged.
3. Change `lora_alpha` in the copied `adapter_config.json`.
4. Evaluate the copied adapter with the same prompt and post-processing pipeline.

Do not overwrite the original checkpoint. Keep all decoding, prompts, dataset order, and post-processing settings fixed while comparing scales.


## License

The code in this repository is released under the [MIT License](LICENSE). Model weights and datasets are distributed under their respective licenses and terms of use.
