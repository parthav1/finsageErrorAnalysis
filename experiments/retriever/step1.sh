#!/bin/bash

# =================================================================
# Hypothesis experiments
# =================================================================
# 
# Get hypothetical answers from the rewritten questions
#
# See step1.py for input and output format

EVAL_DIR="/root/autodl-tmp/RAG_Agent_vllm_cjj/eval"
LOTUS_ENV_NAME="lotusenv"

activate_conda_env() {
    local env_name=$1
    echo "Activating conda environment: $env_name"
    source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
    conda deactivate
    conda activate $env_name || { echo "Failed to activate conda environment: $env_name"; exit 1; }
}

run_step1() {
    local version_name=$1
    local input=$2
    local model_name=$3
    local api_key=$4
    local base_url=$5
    local output_dir=$6

    output="${output_dir}/result_1.json"

    mkdir -p "$(dirname $output)"
    
    activate_conda_env "$LOTUS_ENV_NAME"

    echo "Running evaluation for ${version_name}"

    cmd="python step1.py \
    --input "${input}" \
    --output "${output}" \
    --model_name "${model_name}" \
    --api_key "${api_key}" \
    --base_url "${base_url}""

    echo $cmd

    $cmd

    echo "Results saved in: ${output}"
    return 0
}

echo "=== Running step1 ==="

# Run with eval_all_retrieval.sh
version_name=${1}
rewritten_file=${2}
model_name=${3}
api_key=${4}
base_url=${5}
output_dir=${6}

# Setting 1: use qwen 7b model
# version_name=${1:-"eval_hyde_qwen7b_75"}
# rewritten_file=${2:-"${EVAL_DIR}/answer/75_testingset_75updated_mul.json"}
# model_name=${3:-"hyde"}
# api_key=${4:-"EMPTY"}
# base_url=${5:-"http://localhost:8000/v1"}
# output_dir=${6:-"${EVAL_DIR}/${version_name}"}

# Setting 2: use qwen 7b model with sft lora
# version_name=${1:-"eval_hyde_lora_75"}
# rewritten_file=${2:-"${EVAL_DIR}/answer/75_testingset_75updated_mul.json"}
# model_name=${3:-"hyde-lora"}
# api_key=${4:-"EMPTY"}
# base_url=${5:-"http://localhost:8000/v1"}
# output_dir=${6:-"${EVAL_DIR}/${version_name}"}

# Setting 3: use deepseek
# version_name=${1:-"eval_hyde_deepseek_75"}
# model_name=${3:-"deepseek-chat"}
# api_key=${4:-"sk-4768b45eb65f407790a619db44c37f32"}
# base_url=${5:-"https://api.deepseek.com"}
# output_dir=${6:-"${EVAL_DIR}/${version_name}"}

# Setting 4: use qwen 72b model
# version_name=${1:-"eval_hyde_qwen72b_75"}
# rewritten_file=${2:-"${EVAL_DIR}/answer/75_testingset_75updated_mul.json"}
# model_name=${3:-"Qwen/Qwen2___5-72B-Instruct-AWQ"}
# api_key=${4:-"EMPTY"}
# base_url=${5:-"http://localhost:8000/v1"}
# output_dir=${6:-"${EVAL_DIR}/${version_name}"}


run_step1 ${version_name} ${rewritten_file} ${model_name} ${api_key} ${base_url} ${output_dir}