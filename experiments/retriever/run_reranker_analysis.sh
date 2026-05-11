#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="experiments/reranker"
ANSWER_FILE="experiments/retriever/answer/75.json"

mkdir -p "$OUTPUT_DIR"

echo "Running reranker transition analysis (any)..."
python3 experiments/retriever/analyze_reranker_transitions.py \
  --answer-file "$ANSWER_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --correctness-mode any

echo "Running reranker transition analysis (full)..."
python3 experiments/retriever/analyze_reranker_transitions.py \
  --answer-file "$ANSWER_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --correctness-mode full

echo "Running reranker cutoff ablation..."
python3 experiments/retriever/reranker_ablation.py \
  --answer-file "$ANSWER_FILE" \
  --output-dir "$OUTPUT_DIR"

echo "Preparing reranker failure cases for question75_5..."
python3 experiments/retriever/prepare_reranker_failure_cases.py \
  --input src/test/test_questions/question75_5 \
  --answer-file "$ANSWER_FILE" \
  --output "$OUTPUT_DIR/question75_5_failure_cases.csv"

echo "Preparing reranker failure cases for question75_10..."
python3 experiments/retriever/prepare_reranker_failure_cases.py \
  --input src/test/test_questions/question75_10 \
  --answer-file "$ANSWER_FILE" \
  --output "$OUTPUT_DIR/question75_10_failure_cases.csv"

