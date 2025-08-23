#!/bin/bash

# =================================================================
# Retriever experiemnts
# =================================================================
# 
# Check Retriever performance
#
# See step2.py for input and output format
# params: version_name input_file

EVAL_DIR="/root/autodl-tmp/cjj/RAG_Agent/experiments/retriever"
LOTUS_ENV_NAME="lotusenv"

activate_conda_env() {
    local env_name=$1
    echo "Activating conda environment: $env_name"
    source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
    conda deactivate
    conda activate $env_name || { echo "Failed to activate conda environment: $env_name"; exit 1; }
}

run_step2() {
    local exp_name=$1
    local faiss_k=$2
    local bm25_k=$3
    local faiss_ts_k=$4
    local enable_expand=$5
    local input_file=$6
    local output_base_dir=$7
    local enable_hyde=$8

    output_dir="${output_base_dir}/${version_name}/${exp_name}"
    
    mkdir -p "$output_base_dir"
    
    [ ! -f "$input_file" ] && { echo "Error: Input file not found: ${input_file}"; return 1; }
    
    activate_conda_env "$LOTUS_ENV_NAME"
    
    echo "Running evaluation for ${version_name}/${exp_name}"
    echo "Input: ${input_file}"
    echo "Output directory: ${output_dir}"

    cmd="python step2.py \
    --input "${input_file}" \
    --output "${output_dir}" \
    --answer "${answer_file}" \
    --faiss_k ${faiss_k} \
    --bm25_k ${bm25_k} \
    --faiss_ts_k ${faiss_ts_k}" \

    if [ "$enable_expand" = true ]; then
        cmd="$cmd --enable_expand"
    fi

    if [ "$enable_hyde" = true ]; then
        cmd="$cmd --enable_hyde"
    fi

    echo $cmd

    $cmd

    echo "Results saved in: ${output_dir}"
    return 0
}

echo "=== Running step2 ==="


### Run with eval_all_retrieval.sh

input_file=${1:-"input_file"}
output_base_dir="${2:-${EVAL_DIR}}"
version_name=${3:-"result_expand_70"}
faiss_k=${4:-10}
answer_file=${5:-"/root/autodl-tmp/cjj/RAG_Agent/experiments/retriever/answer/75_testingset_75updated_mul.json"}

# run_step2 "faiss" 10 0 0 false ${input_file} ${output_base_dir} true


### Venn graph Experiment (single retriever) ###

# input_file="/root/autodl-tmp/RAG_Agent_vllm_cjj/eval/result_hyde/eval_hyde_lora_75_75/checkpoint-800/result_1.json"
# version_name=${3:-"result_single_retriever"}

# echo "=== Running ${version_name}_faiss ==="
# run_step2 "faiss" 10 0 0 false ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_bm25 ==="
# run_step2 "bm25" 0 10 0 false ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_ts ==="
# run_step2 "title_summary" 0 0 10 false ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_faiss_expand ==="
# run_step2 "faiss_expand" 10 0 0 true ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_hyde ==="
# WARNING: In this case, the ensembleRetriever should only use hypothetical answers but not origional query. So modify the code first and run this experiment.
# run_step2 "hyde" 10 0 0 false ${input_file} ${output_base_dir} true


### Main Experiment (get approximatly same number of chunks for every experiment)###

# input_file="/root/autodl-tmp/RAG_Agent_vllm_cjj/eval/result_hyde/eval_hyde_lora_75_75/checkpoint-800/result_1.json"
# version_name=${3:-"result_same_num_chunks"}

# echo "=== Running ${version_name}_faiss ==="
# run_step2 "faiss" 70 0 0 false ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_faiss_expand ==="
# run_step2 "faiss_expand" 65 0 0 true ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_faiss_bm25 ==="
# run_step2 "faiss_bm25" 35 35 0 false ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_faiss_ts ==="
# run_step2 "faiss_ts" 40 0 10 false ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_faiss_bm25_ts ==="
# run_step2 "faiss_bm25_ts" 25 25 10 false ${input_file} ${output_base_dir} false

# echo "=== Running ${version_name}_faiss_bm25_ts_hyde ==="
# run_step2 "faiss_bm25_ts_hyde" 10 10 10 true ${input_file} ${output_base_dir} true


### Hyde Experiments (compare 7b, 72b and 7b with lora adapter)###

# input_file="/root/autodl-tmp/RAG_Agent_vllm_cjj/eval/result_hyde/eval_hyde_lora_75_75/checkpoint-800/result_1.json"
# version_name=${3:-"result_hyde"}

# echo "=== Running ${version_name}_faiss_hyde(lora) ==="
# run_step2 "faiss_hyde(qwen7b-sft)" 30 0 0 false ${input_file} ${output_base_dir} true

# input_file="/root/autodl-tmp/RAG_Agent_vllm_cjj/eval/result_hyde/eval_hyde_qwen72b_75/result_1.json"

# echo "=== Running ${version_name}_faiss_hyde(qwen72b) ==="
# run_step2 "faiss_hyde(qwen72b)" 25 0 0 false ${input_file} ${output_base_dir} true

# input_file="/root/autodl-tmp/RAG_Agent_vllm_cjj/eval/result_hyde/eval_hyde_qwen7b_75/result_1.json"

# echo "=== Running ${version_name}_faiss_bm25_ts_hyde(qwen7b) ==="
# run_step2 "faiss_hyde(qwen7b)" 25 0 0 false ${input_file} ${output_base_dir} true


### Expand Experiments (compare expand and non-expand faiss) ###
input_file="/root/autodl-tmp/cjj/RAG_Agent/experiments/retriever/result_hyde/eval_hyde_lora_75_75/checkpoint-800/result_1.json"
run_step2 "faiss_noexpand" 70 0 0 false ${input_file} ${output_base_dir} false

input_file="/root/autodl-tmp/cjj/RAG_Agent/experiments/retriever/result_hyde/eval_hyde_lora_75_75/checkpoint-800/result_1.json"
run_step2 "faiss_expand" 70 0 0 true ${input_file} ${output_base_dir} false


echo "=== Evaluation Complete ==="
