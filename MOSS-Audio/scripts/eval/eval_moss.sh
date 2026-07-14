#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MOSS_AUDIO_UPSTREAM="${REPO_ROOT}/third_party/MOSS-Audio"

if [[ ! -f "${MOSS_AUDIO_UPSTREAM}/src/modeling_moss_audio.py" ]]; then
    echo "Missing MOSS-Audio submodule at ${MOSS_AUDIO_UPSTREAM}" >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

export PYTHONPATH="${MOSS_AUDIO_UPSTREAM}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python "${REPO_ROOT}/MOSS-Audio/src/eval/Full/main.py" \
    --model_path "path/to/moss-audio/model" \
    --embedding_model_path "path/to/embedding_model" \
    --output_dir "path/to/output/moss_full" \
    --lora_path "path/to/lora" \
    --dataset_path "path/to/dataset_jsonl" \
    --dataset_root "path/to/dataset_root"

python "${REPO_ROOT}/MOSS-Audio/src/eval/Label/main.py" \
    --model_path "path/to/moss-audio/model" \
    --embedding_model_path "path/to/embedding_model" \
    --output_dir "path/to/output/moss_label" \
    --lora_path "path/to/lora" \
    --dataset_path "path/to/dataset_jsonl" \
    --dataset_root "path/to/dataset_root"
