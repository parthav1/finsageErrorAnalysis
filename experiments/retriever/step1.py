'''
Step 1: Hyde
    1. Perplexity: not computed with native Ollama API (set to -1); use OpenAI-compatible
       stack if you need token logprobs.
    2. Get hyde content via local Ollama (default model: qwen2.5:14b).

Requires: pip install ollama
          ollama pull qwen2.5:14b

Input
    JSON file with the following structure:
    [
        {
            "question": "original question",
            "rewritten": ["rewritten question 1", "rewritten question 2"]
        },
        ...
    ]

Output
    Json with the following structure:
    {
        "question": "original question",
        "rewritten": "rewritten question",
        "hyde": ["hyde 1", "hyde 2", ... ],
        "perplexity": 0.0 or -1 if unavailable
    }

'''

import os
import json
import logging
import sys
from tqdm import tqdm
import argparse

import ollama

logging.basicConfig(
    filemode='w',
    filename='step1.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Repo root: experiments/retriever/step1.py -> three levels up (same as step2.py)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.prompts.hyde import get_hypo_sys_prompt

def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(data, file_path):
    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_hyde_response(prompt: str, model: str, client) -> tuple[str, None]:
    """
    Call Ollama chat. Returns (content, None) — token logprobs are not used; perplexity stays -1.
    """
    err = None
    for _ in range(5):
        try:
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": get_hypo_sys_prompt(3)},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": 0,
                    "top_p": 1.0,
                },
            )
            content = response["message"]["content"]
            return content, None
        except Exception as e:
            err = e
            logging.warning(f"Ollama chat attempt failed: {e}")
            continue
    raise err

def hyde_rewritten(file_path, model, client, output_file: str):
    data = load_json_file(file_path)
    results = []

    for idx, entry in tqdm(enumerate(data)):
        try:
            rewritten = entry.get("rewritten", "")

            if rewritten == "":
                logging.warning(f"Skipping entry {idx}: No rewritten questions found")
                continue

            rewritten = [rewritten] if type(rewritten) == str else rewritten

            hydes = []
            ppls = []
            for q in rewritten:
                logging.info(f"Evaluation questions: {q}")
                content, _ = get_hyde_response(q, model, client)
                hyde = [chunk.strip() for chunk in content.split("ANSWER:")[1:]]
                hydes.append(hyde)
                if len(hyde) == 0:
                    logging.warning(f"No hyde content found for question: {q}")
                logging.info("Native Ollama client does not provide token logprobs; perplexity set to -1")
                ppls.append(-1.0)
            results.append({
                "question": entry.get("question"),
                "rewritten": rewritten,
                "hyde": hydes,
                "perplexity": ppls,
            })
            save_json_file(results, output_file)
        except Exception as e:
            logging.warning(f"Error processing entry {idx}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="HyDE generation using the local Ollama Python API (no OpenAI SDK)."
    )
    parser.add_argument('--input', type=str, default='answer/75.json', help='Input JSON file path')
    parser.add_argument('--output', type=str, default='hyde_results_75.json', help='Output file path')
    parser.add_argument(
        '--model_name',
        type=str,
        default='qwen2.5:14b',
        help='Ollama model tag (run: ollama pull <name>)',
    )
    parser.add_argument(
        '--host',
        type=str,
        default=os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434'),
        help='Ollama server URL (default: http://127.0.0.1:11434 or OLLAMA_HOST env)',
    )
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    model_name = args.model_name

    if not os.path.exists(input_file):
        logging.error(f"File not found: {input_file}")
        return

    client = ollama.Client(host=args.host)
    hyde_rewritten(input_file, model_name, client, output_file)
    logging.info(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    main()
