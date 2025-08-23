# !/bin/bash

# =================================================================
# Evaluate all trained hyde models
# =================================================================
# 
# Check model checkpoints, do hyde first and evaluate the retrieving chunks.
# Run with step1.sh and step2.sh

OUT_DIR="/root/autodl-tmp/cjj/RAG_Agent/experiments/retriever/expand"
EVAL_DIR="/root/autodl-tmp/cjj/RAG_Agent/experiments/retriever"
ANSWER_FILE="${EVAL_DIR}/answer/75_testingset_75updated_mul.json"
MODEL_PATH=/root/autodl-tmp/LLaMA-Factory/cache/Qwen
MODEL=Qwen2___5-7B-Instruct
MAX_LENGTH=16000
ADAPTER_PATH=/root/autodl-tmp/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/lora/hyde3_epoch3
VLLM_PORT=8000
VLLM_ENV_NAME="vllm"

# CHECKPOINT_LIST=("checkpoint-200")
CHECKPOINT_LIST=("checkpoint-200" "checkpoint-400" "checkpoint-600" "checkpoint-800" "checkpoint-843")

# Function to wait for VLLM server to be ready
wait_for_server() {
    local max_attempts=50
    local attempt=1
    
    while ! curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo "Server failed to start after $max_attempts attempts"
            return 1
        fi
        echo "Waiting for VLLM server to start (attempt $attempt/$max_attempts)..."
        sleep 2
        ((attempt++))
    done
    return 0
}

# Function to kill VLLM server
kill_server() {
    # local pid=$(lsof -ti:${VLLM_PORT})
    # if [ ! -z "$pid" ]; then
    #     echo "Killing VLLM server (PID: $pid)"
    #     kill -2 $pid
    # fi
    vllm_pids=$(ps aux | grep vllm | grep -v grep | awk '{print $2}')
    if [ ! -z "$vllm_pids" ]; then
        echo "Killing VLLM server (PID: $vllm_pids)"
        for pid in $vllm_pids; do
            kill -2 $pid 2>/dev/null || true
        done
    fi

    sleep 10

    # check the pid again. If not killed, force kill it
    vllm_pids=$(ps aux | grep vllm | grep -v grep | awk '{print $2}')
    if [ ! -z "$vllm_pids" ]; then
        echo "Force killing remaining VLLM processes"
        for pid in $vllm_pids; do
            kill -9 $pid 2>/dev/null || true
        done
    fi
}

activate_conda_env() {
    local env_name=$1
    echo "Activating conda environment: $env_name"
    source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
    conda deactivate
    conda activate $env_name || { echo "Failed to activate conda environment: $env_name"; exit 1; }
}

kill_server

run_evaluation() {
    local lora_name=$1
    local faiss_k=$2    

    # Activate inference environment for VLLM
    activate_conda_env "$VLLM_ENV_NAME"

    # Start VLLM server
    echo "Starting VLLM server with LoRA path: ${lora_path}"
    lora_path="${ADAPTER_PATH}/${lora_name}"
    python -m vllm.entrypoints.openai.api_server --trust-remote-code \
      --model ${MODEL_PATH}/${MODEL} --host 0.0.0.0 \
      --port ${VLLM_PORT} --tensor-parallel-size 1 \
      --max-num-batched-tokens ${MAX_LENGTH} \
      --served-model-name hyde \
      --gpu-memory-utilization 0.6 \
      --trust-remote-code \
      --max-model-len ${MAX_LENGTH} \
      --max-seq-len-to-capture ${MAX_LENGTH} \
      --swap-space 8 \
      --enable-prefix-caching \
      --enable-lora \
      --lora-modules hyde-lora=${lora_path} &
    
    # Wait for server to start
    if ! wait_for_server; then
        echo "Failed to start VLLM server"
        return 1
    fi

    ./step1.sh ${lora_name} ${ANSWER_FILE} "hyde-lora" "EMPTY" "http://localhost:${VLLM_PORT}/v1" "${OUT_DIR}/${lora_name}"

    kill_server

    ./step2.sh "${OUT_DIR}/${lora_name}/result_1.json" ${OUT_DIR} ${lora_name} ${faiss_k} ${ANSWER_FILE}

}

mkdir -p "${OUT_DIR}"

for checkpoint in "${CHECKPOINT_LIST[@]}"; do
    run_evaluation ${checkpoint} 10
done
