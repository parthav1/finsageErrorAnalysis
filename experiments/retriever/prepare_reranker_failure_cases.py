import argparse
import json
from pathlib import Path
from typing import Dict, List

from analyze_reranker_transitions import (
    DEFAULT_ANSWER_FILE,
    compute_metrics,
    extract_reranked_chunks,
    flatten_retrieved_chunks,
    iter_question_entries,
    load_answer_map,
    write_csv,
)


def brief_chunks(chunks: List[str], limit: int = 3, max_chars: int = 220) -> str:
    snippets = []
    for chunk in chunks[:limit]:
        snippet = chunk[:max_chars].replace("\n", " ")
        snippets.append(snippet)
    return " || ".join(snippets)


def collect_cases(input_path: Path, answer_map: Dict[str, Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for question_file, question_entry in iter_question_entries(input_path):
        question_text = question_entry.get("original_question", "")
        answer_entry = answer_map.get(question_text)
        if answer_entry is None:
            continue

        gold_chunks = answer_entry["gold_chunks"]
        retrieved_chunks = flatten_retrieved_chunks(question_entry)
        reranked_chunks = extract_reranked_chunks(question_entry)
        retrieval_metrics = compute_metrics(retrieved_chunks, gold_chunks)
        rerank_metrics = compute_metrics(reranked_chunks, gold_chunks)

        rows.append(
            {
                "source_file": str(question_file),
                "question_index": question_entry.get("question_index"),
                "question_zh": question_text,
                "question_en": " | ".join(question_entry.get("rewritten_question", [])),
                "gold_chunk_count": retrieval_metrics["gold_count"],
                "retrieved_chunk_count": retrieval_metrics["predicted_count"],
                "reranked_chunk_count": rerank_metrics["predicted_count"],
                "retrieval_recall": retrieval_metrics["recall"],
                "retrieval_precision": retrieval_metrics["precision"],
                "retrieval_f1": retrieval_metrics["f1"],
                "rerank_recall": rerank_metrics["recall"],
                "rerank_precision": rerank_metrics["precision"],
                "rerank_f1": rerank_metrics["f1"],
                "delta_recall": rerank_metrics["recall"] - retrieval_metrics["recall"],
                "delta_precision": rerank_metrics["precision"] - retrieval_metrics["precision"],
                "delta_f1": rerank_metrics["f1"] - retrieval_metrics["f1"],
                "gold_chunk_preview": brief_chunks(gold_chunks),
                "retrieved_match_preview": brief_chunks(retrieval_metrics["matched_chunks"]),
                "rerank_match_preview": brief_chunks(rerank_metrics["matched_chunks"]),
                "reranked_chunk_preview": brief_chunks(reranked_chunks),
            }
        )
    return rows


def add_annotation_columns(rows: List[Dict], dataset: str, bucket: str) -> List[Dict]:
    annotated = []
    for row in rows:
        annotated_row = {
            "dataset": dataset,
            "bucket": bucket,
            **row,
            "manual_label": "",
            "secondary_label": "",
            "notes": "",
        }
        annotated.append(annotated_row)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare reranker helped/hurt cases for manual annotation.")
    parser.add_argument(
        "--input",
        default="src/test/test_questions/question75_10",
        help="Question JSON file or directory to analyze.",
    )
    parser.add_argument(
        "--answer-file",
        default=str(DEFAULT_ANSWER_FILE),
        help="Gold answer JSON file with per-question content lists.",
    )
    parser.add_argument(
        "--output",
        default="experiments/reranker/reranker_failure_cases.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--top-helped",
        type=int,
        default=8,
        help="Number of top helped examples to export.",
    )
    parser.add_argument(
        "--top-hurt",
        type=int,
        default=8,
        help="Number of top hurt examples to export.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    dataset = input_path.name if input_path.is_dir() else input_path.stem
    answer_map = load_answer_map(Path(args.answer_file))

    rows = collect_cases(input_path, answer_map)
    if not rows:
        print(f"No matching questions found for {input_path}")
        return

    helped = sorted(
        [row for row in rows if row["delta_f1"] > 0],
        key=lambda row: (row["delta_f1"], row["delta_precision"]),
        reverse=True,
    )[: args.top_helped]
    hurt = sorted(
        [row for row in rows if row["delta_f1"] < 0 or (row["retrieval_recall"] > 0 and row["rerank_recall"] == 0)],
        key=lambda row: (row["delta_f1"], row["delta_recall"]),
    )[: args.top_hurt]

    output_rows = []
    output_rows.extend(add_annotation_columns(helped, dataset, "helped"))
    output_rows.extend(add_annotation_columns(hurt, dataset, "hurt"))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_rows, output_path)
    print(f"Saved annotation sheet to {output_path}")


if __name__ == "__main__":
    main()
