import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from analyze_reranker_transitions import (
    DEFAULT_ANSWER_FILE,
    compute_metrics,
    extract_reranked_chunks,
    flatten_retrieved_chunks,
    iter_question_entries,
    load_answer_map,
    normalize_text,
    unique_preserve_order,
    write_csv,
)


DEFAULT_DATASETS = [
    ("retrieval_vs_rerank@5", "src/test/test_questions/question75_5"),
    ("retrieval_vs_rerank@10", "src/test/test_questions/question75_10"),
]


def evaluate_dataset(input_path: Path, answer_map: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict]]:
    per_question_rows: List[Dict] = []

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

        per_question_rows.append(
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
                "retrieval_matched_count": retrieval_metrics["matched_count"],
                "rerank_matched_count": rerank_metrics["matched_count"],
            }
        )

    if not per_question_rows:
        return [], []

    question_count = len(per_question_rows)
    retrieval_only_row = {
        "setting": "retrieval_only",
        "question_count": question_count,
        "avg_recall": sum(row["retrieval_recall"] for row in per_question_rows) / question_count,
        "avg_precision": sum(row["retrieval_precision"] for row in per_question_rows) / question_count,
        "avg_f1": sum(row["retrieval_f1"] for row in per_question_rows) / question_count,
        "avg_chunk_count": sum(row["retrieved_chunk_count"] for row in per_question_rows) / question_count,
    }
    rerank_row = {
        "setting": input_path.name.replace("question75_", "rerank@"),
        "question_count": question_count,
        "avg_recall": sum(row["rerank_recall"] for row in per_question_rows) / question_count,
        "avg_precision": sum(row["rerank_precision"] for row in per_question_rows) / question_count,
        "avg_f1": sum(row["rerank_f1"] for row in per_question_rows) / question_count,
        "avg_chunk_count": sum(row["reranked_chunk_count"] for row in per_question_rows) / question_count,
    }

    return per_question_rows, [retrieval_only_row, rerank_row]


def parse_dataset_args(dataset_args: List[str]) -> List[Tuple[str, Path]]:
    parsed = []
    for dataset_arg in dataset_args:
        if "=" not in dataset_arg:
            raise ValueError(f"Expected dataset specification in label=path format, got: {dataset_arg}")
        label, path = dataset_arg.split("=", 1)
        parsed.append((label, Path(path)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reranker cutoff ablation tables.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[f"{label}={path}" for label, path in DEFAULT_DATASETS],
        help="Datasets in label=path format.",
    )
    parser.add_argument(
        "--answer-file",
        default=str(DEFAULT_ANSWER_FILE),
        help="Gold answer JSON file with per-question content lists.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/reranker",
        help="Directory to write ablation CSVs.",
    )
    args = parser.parse_args()

    answer_map = load_answer_map(Path(args.answer_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: List[Dict] = []

    for label, input_path in parse_dataset_args(args.datasets):
        per_question_rows, summary_rows = evaluate_dataset(input_path, answer_map)
        if not per_question_rows:
            print(f"No matching questions found for {label} ({input_path})")
            continue

        for row in summary_rows:
            tagged_row = {"dataset": label}
            tagged_row.update(row)
            combined_rows.append(tagged_row)

        summary_path = output_dir / f"{label}_ablation_summary.csv"
        per_question_path = output_dir / f"{label}_ablation_per_question.csv"
        write_csv(summary_rows, summary_path)
        write_csv(per_question_rows, per_question_path)
        print(f"Saved ablation summary to {summary_path}")
        print(f"Saved per-question ablation data to {per_question_path}")

    if combined_rows:
        combined_path = output_dir / "reranker_ablation_combined.csv"
        write_csv(combined_rows, combined_path)
        print(f"Saved combined ablation table to {combined_path}")


if __name__ == "__main__":
    main()
