import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_ANSWER_FILE = Path("experiments/retriever/answer/75.json")
DEFAULT_OUTPUT_DIR = Path("experiments/reranker")


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def load_answer_map(answer_file: Path) -> Dict[str, Dict]:
    with answer_file.open("r", encoding="utf-8") as file:
        answers = json.load(file)

    answer_map = {}
    for entry in answers:
        question = entry.get("question", "")
        answer_map[question] = {
            "question": question,
            "rewritten": entry.get("rewritten", ""),
            "answer": entry.get("answer", ""),
            "gold_chunks": [normalize_text(chunk) for chunk in entry.get("content", []) if normalize_text(chunk)],
        }
    return answer_map


def iter_question_entries(input_path: Path) -> Iterable[Tuple[Path, Dict]]:
    if input_path.is_file():
        question_files = [input_path]
    else:
        question_files = sorted(input_path.glob("*.json"))

    for question_file in question_files:
        with question_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, dict):
            questions = payload.get("questions", [])
        elif isinstance(payload, list):
            questions = payload
        else:
            questions = []

        for question in questions:
            yield question_file, question


def flatten_retrieved_chunks(question_entry: Dict) -> List[str]:
    chunks = []
    for sub_query_chunks in question_entry.get("all_retrieved_content", []):
        for chunk in sub_query_chunks:
            text = normalize_text(chunk.get("page_content", ""))
            if text:
                chunks.append(text)
    return chunks


def extract_reranked_chunks(question_entry: Dict) -> List[str]:
    chunks = []
    for chunk in question_entry.get("rag_info", []):
        text = normalize_text(chunk.get("chunk_content", ""))
        if text:
            chunks.append(text)
    return chunks


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def compute_metrics(predicted_chunks: List[str], gold_chunks: List[str]) -> Dict[str, object]:
    predicted_unique = unique_preserve_order(predicted_chunks)
    gold_set = set(gold_chunks)
    matched = [chunk for chunk in predicted_unique if chunk in gold_set]

    recall = len(matched) / len(gold_chunks) if gold_chunks else 0.0
    precision = len(matched) / len(predicted_unique) if predicted_unique else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "matched_chunks": matched,
        "matched_count": len(matched),
        "predicted_count": len(predicted_unique),
        "gold_count": len(gold_chunks),
    }


def is_correct(metrics: Dict[str, object], mode: str) -> bool:
    recall = float(metrics["recall"])
    if mode == "any":
        return recall > 0.0
    if mode == "full":
        return recall == 1.0
    raise ValueError(f"Unsupported correctness mode: {mode}")


def transition_label(before_correct: bool, after_correct: bool) -> str:
    return f'{"correct" if before_correct else "wrong"}->{"correct" if after_correct else "wrong"}'


def analyze_questions(input_path: Path, answer_map: Dict[str, Dict], correctness_mode: str) -> Tuple[List[Dict], List[Dict]]:
    per_question_rows = []

    for question_file, question_entry in iter_question_entries(input_path):
        question_text = question_entry.get("original_question", "")
        answer_entry = answer_map.get(question_text)
        if answer_entry is None:
            continue

        gold_chunks = answer_entry["gold_chunks"]
        retrieved_chunks = flatten_retrieved_chunks(question_entry)
        reranked_chunks = extract_reranked_chunks(question_entry)

        before_metrics = compute_metrics(retrieved_chunks, gold_chunks)
        after_metrics = compute_metrics(reranked_chunks, gold_chunks)

        before_correct = is_correct(before_metrics, correctness_mode)
        after_correct = is_correct(after_metrics, correctness_mode)

        per_question_rows.append(
            {
                "source_file": str(question_file),
                "question_index": question_entry.get("question_index"),
                "question_zh": question_text,
                "question_en": " | ".join(question_entry.get("rewritten_question", [])),
                "gold_chunk_count": before_metrics["gold_count"],
                "retrieved_chunk_count": before_metrics["predicted_count"],
                "reranked_chunk_count": after_metrics["predicted_count"],
                "retrieval_recall": before_metrics["recall"],
                "retrieval_precision": before_metrics["precision"],
                "retrieval_f1": before_metrics["f1"],
                "rerank_recall": after_metrics["recall"],
                "rerank_precision": after_metrics["precision"],
                "rerank_f1": after_metrics["f1"],
                "retrieval_correct": before_correct,
                "rerank_correct": after_correct,
                "transition": transition_label(before_correct, after_correct),
                "retrieval_matched_count": before_metrics["matched_count"],
                "rerank_matched_count": after_metrics["matched_count"],
                "retrieval_matched_chunks": json.dumps(before_metrics["matched_chunks"], ensure_ascii=False),
                "rerank_matched_chunks": json.dumps(after_metrics["matched_chunks"], ensure_ascii=False),
            }
        )

    if not per_question_rows:
        return [], []

    total_questions = len(per_question_rows)
    transition_counts: Dict[str, int] = {}
    for row in per_question_rows:
        transition = row["transition"]
        transition_counts[transition] = transition_counts.get(transition, 0) + 1

    summary_rows = []
    for transition in ["correct->correct", "correct->wrong", "wrong->correct", "wrong->wrong"]:
        count = int(transition_counts.get(transition, 0))
        summary_rows.append(
            {
                "transition": transition,
                "count": count,
                "fraction": count / total_questions if total_questions else 0.0,
            }
        )

    summary_rows.extend(
        [
            {
                "transition": "avg_retrieval_recall",
                "count": sum(row["retrieval_recall"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
            {
                "transition": "avg_retrieval_precision",
                "count": sum(row["retrieval_precision"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
            {
                "transition": "avg_retrieval_f1",
                "count": sum(row["retrieval_f1"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
            {
                "transition": "avg_rerank_recall",
                "count": sum(row["rerank_recall"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
            {
                "transition": "avg_rerank_precision",
                "count": sum(row["rerank_precision"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
            {
                "transition": "avg_rerank_f1",
                "count": sum(row["rerank_f1"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
            {
                "transition": "avg_retrieved_chunk_count",
                "count": sum(row["retrieved_chunk_count"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
            {
                "transition": "avg_reranked_chunk_count",
                "count": sum(row["reranked_chunk_count"] for row in per_question_rows) / total_questions,
                "fraction": None,
            },
        ]
    )

    return per_question_rows, summary_rows


def dataset_name(input_path: Path) -> str:
    return input_path.stem if input_path.is_file() else input_path.name


def write_csv(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval-to-rerank transitions against gold chunks.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "src/test/test_questions/question75_5",
            "src/test/test_questions/question75_10",
        ],
        help="One or more question JSON files or directories containing question JSON files.",
    )
    parser.add_argument(
        "--answer-file",
        default=str(DEFAULT_ANSWER_FILE),
        help="Gold answer JSON file with per-question content lists.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write per-question and summary CSVs.",
    )
    parser.add_argument(
        "--correctness-mode",
        choices=["any", "full"],
        default="any",
        help="Treat a question as correct if reranking keeps any gold chunk or all gold chunks.",
    )
    args = parser.parse_args()

    answer_map = load_answer_map(Path(args.answer_file))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_summary: List[Dict] = []

    for input_item in args.inputs:
        input_path = Path(input_item)
        per_question_rows, summary_rows = analyze_questions(input_path, answer_map, args.correctness_mode)
        if not per_question_rows:
            print(f"No matching questions found for {input_path}")
            continue

        name = dataset_name(input_path)
        per_question_path = output_dir / f"{name}_per_question_{args.correctness_mode}.csv"
        summary_path = output_dir / f"{name}_summary_{args.correctness_mode}.csv"

        write_csv(per_question_rows, per_question_path)
        write_csv(summary_rows, summary_path)

        print(f"Saved per-question analysis to {per_question_path}")
        print(f"Saved summary analysis to {summary_path}")

        for row in summary_rows:
            tagged_row = {"dataset": name}
            tagged_row.update(row)
            combined_summary.append(tagged_row)

    if combined_summary:
        combined_path = output_dir / f"combined_summary_{args.correctness_mode}.csv"
        write_csv(combined_summary, combined_path)
        print(f"Saved combined summary to {combined_path}")


if __name__ == "__main__":
    main()
